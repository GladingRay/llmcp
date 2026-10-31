//===- LowerToLinalg.cpp - Lower LLMCP ops to Linalg ----------------------===//
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/Passes.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCP.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::llmcp;
using llvm::StringRef;

namespace {

/// Materializes a typed tensor view of a weight memref.
Value toTensor(PatternRewriter &rewriter, Location loc, Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto tensorType =
      RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
  return rewriter.create<bufferization::ToTensorOp>(loc, tensorType, memref,
                                                    /*restrict=*/true);
}

/// Collects the dynamic dimensions of `type` from `source`.
SmallVector<Value> getDynamicSizes(PatternRewriter &rewriter, Location loc,
                                   RankedTensorType type, Value source) {
  SmallVector<Value> sizes;
  for (int64_t i = 0, e = type.getRank(); i < e; ++i) {
    if (type.isDynamicDim(i))
      sizes.push_back(rewriter.createOrFold<tensor::DimOp>(loc, source, i));
  }
  return sizes;
}

Value createEmpty(PatternRewriter &rewriter, Location loc,
                  RankedTensorType type, Value source) {
  return rewriter.create<tensor::EmptyOp>(
      loc, type, getDynamicSizes(rewriter, loc, type, source));
}

constexpr int64_t kVectorTile = 32;

Value emitQ8Matmul(PatternRewriter &rewriter, Location loc, Value input,
                   Value weight, RankedTensorType resultType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<MemRefType>(weight.getType());
  const int64_t reduction = inputType.getDimSize(1);
  const int64_t channels = resultType.getDimSize(1);
  if (inputType.getRank() != 2 || weightType.getRank() != 2 ||
      ShapedType::isDynamic(reduction) || ShapedType::isDynamic(channels) ||
      reduction % 32 != 0 || !weightType.getElementType().isInteger(8))
    return Value();

  auto f32 = rewriter.getF32Type();
  auto i8 = rewriter.getIntegerType(8);
  auto i16 = rewriter.getIntegerType(16);
  auto i32 = rewriter.getIntegerType(32);
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value blockSize = rewriter.create<arith::ConstantIndexOp>(loc, 32);
  Value blockCount =
      rewriter.create<arith::ConstantIndexOp>(loc, reduction / 32);
  Value channelsValue = rewriter.create<arith::ConstantIndexOp>(loc, channels);
  Value rows = rewriter.create<tensor::DimOp>(loc, input, 0);
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output =
      rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{rows});

  // Pre-quantize activations once: Aq[rows, K] and As[rows, K/32].
  Value quantized = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get({ShapedType::kDynamic, reduction}, i8),
      ValueRange{rows});
  Value scales = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get({ShapedType::kDynamic, reduction / 32}, f32),
      ValueRange{rows});
  auto rowQuantLoop = rewriter.create<scf::ForOp>(
      loc, zero, rows, one, ValueRange{});
  rowQuantLoop->setAttr("llmcp.parallel",
                        UnitAttr::get(rewriter.getContext()));
  rewriter.setInsertionPointToStart(rowQuantLoop.getBody());
  Value quantRow = rowQuantLoop.getInductionVar();
  auto quantBlockLoop = rewriter.create<scf::ForOp>(
      loc, zero, blockCount, one, ValueRange{});
  rewriter.setInsertionPointToStart(quantBlockLoop.getBody());
  Value quantBlock = quantBlockLoop.getInductionVar();
  Value blockBase =
      rewriter.create<arith::MulIOp>(loc, quantBlock, blockSize);
  Value maxInit =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  auto maxLoop = rewriter.create<scf::ForOp>(
      loc, zero, blockSize, one, ValueRange{maxInit});
  rewriter.setInsertionPointToStart(maxLoop.getBody());
  Value maxIndex = maxLoop.getInductionVar();
  Value sourceIndex =
      rewriter.create<arith::AddIOp>(loc, blockBase, maxIndex);
  Value value = rewriter.create<tensor::ExtractOp>(
      loc, input, ValueRange{quantRow, sourceIndex});
  if (value.getType().isBF16())
    value = rewriter.create<arith::ExtFOp>(loc, f32, value);
  Value magnitude = rewriter.create<math::AbsFOp>(loc, value);
  Value maxValue = rewriter.create<arith::MaxNumFOp>(
      loc, maxLoop.getRegionIterArg(0), magnitude);
  rewriter.create<scf::YieldOp>(loc, maxValue);
  rewriter.setInsertionPointAfter(maxLoop);

  Value maxAbs = maxLoop.getResult(0);
  Value scale = rewriter.create<arith::DivFOp>(
      loc, maxAbs,
      rewriter.create<arith::ConstantOp>(loc,
                                         rewriter.getF32FloatAttr(127.0f)));
  rewriter.create<memref::StoreOp>(loc, scale, scales,
                                   ValueRange{quantRow, quantBlock});
  Value zeroF =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  Value isZero = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                                maxAbs, zeroF);
  auto inverse = rewriter.create<scf::IfOp>(loc, TypeRange{f32}, isZero, true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&inverse.getThenRegion().front());
    rewriter.create<scf::YieldOp>(loc, zeroF);
  }
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&inverse.getElseRegion().front());
    Value oneF =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
    rewriter.create<scf::YieldOp>(
        loc, ValueRange{rewriter.create<arith::DivFOp>(loc, oneF, scale)});
  }
  rewriter.setInsertionPointAfter(inverse.getOperation());

  auto quantizeLoop = rewriter.create<scf::ForOp>(
      loc, zero, blockSize, one, ValueRange{});
  rewriter.setInsertionPointToStart(quantizeLoop.getBody());
  Value quantIndex = quantizeLoop.getInductionVar();
  Value quantSource =
      rewriter.create<arith::AddIOp>(loc, blockBase, quantIndex);
  Value quantValue = rewriter.create<tensor::ExtractOp>(
      loc, input, ValueRange{quantRow, quantSource});
  if (quantValue.getType().isBF16())
    quantValue = rewriter.create<arith::ExtFOp>(loc, f32, quantValue);
  Value scaled = rewriter.create<arith::MulFOp>(
      loc, quantValue, inverse.getResult(0));
  Value rounded = rewriter.create<math::RoundEvenOp>(loc, scaled);
  Value aInt = rewriter.create<arith::FPToSIOp>(loc, i32, rounded);
  Value aByte = rewriter.create<arith::TruncIOp>(loc, i8, aInt);
  rewriter.create<memref::StoreOp>(loc, aByte, quantized,
                                   ValueRange{quantRow, quantSource});
  rewriter.setInsertionPointAfter(quantizeLoop);
  rewriter.setInsertionPointAfter(quantBlockLoop);
  rewriter.setInsertionPointAfter(rowQuantLoop);

  // Q8_0 x Q8_1 dot product.
  auto columnLoop =
      rewriter.create<scf::ForOp>(loc, zero, channelsValue, one, ValueRange{});
  columnLoop->setAttr("llmcp.parallel", UnitAttr::get(rewriter.getContext()));
  rewriter.setInsertionPointToStart(columnLoop.getBody());
  Value column = columnLoop.getInductionVar();
  auto rowLoop =
      rewriter.create<scf::ForOp>(loc, zero, rows, one, ValueRange{});
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value row = rowLoop.getInductionVar();
  Value accInit =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  auto blockLoop = rewriter.create<scf::ForOp>(
      loc, zero, blockCount, one, ValueRange{accInit});
  rewriter.setInsertionPointToStart(blockLoop.getBody());
  Value block = blockLoop.getInductionVar();
  Value dotBlockBase =
      rewriter.create<arith::MulIOp>(loc, block, blockSize);
  Value scaleOffset = rewriter.create<arith::MulIOp>(
      loc, block, rewriter.create<arith::ConstantIndexOp>(loc, 34));
  Value lowByte = rewriter.create<memref::LoadOp>(
      loc, weight, ValueRange{column, scaleOffset});
  Value highOffset = rewriter.create<arith::AddIOp>(loc, scaleOffset, one);
  Value highByte = rewriter.create<memref::LoadOp>(
      loc, weight, ValueRange{column, highOffset});
  Value low = rewriter.create<arith::ExtUIOp>(loc, i16, lowByte);
  Value high = rewriter.create<arith::ExtUIOp>(loc, i16, highByte);
  Value shift =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(8));
  Value shifted = rewriter.create<arith::ShLIOp>(loc, high, shift);
  Value bits = rewriter.create<arith::OrIOp>(loc, low, shifted);
  Value half =
      rewriter.create<arith::BitcastOp>(loc, rewriter.getF16Type(), bits);
  Value weightScale = rewriter.create<arith::ExtFOp>(loc, f32, half);

  Value weightedDot;
#if defined(__aarch64__)
  auto i8x16 = VectorType::get({16}, i8);
  auto i32x4 = VectorType::get({4}, i32);
  AffineMap rowMap =
      AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                     rewriter.getContext());
  SmallVector<bool> inBounds = {true};
  Value zeroVec = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(i32x4, rewriter.getI32IntegerAttr(0)));
  Value aVec0 = rewriter.create<vector::TransferReadOp>(
      loc, i8x16, quantized, ValueRange{row, dotBlockBase}, rowMap, inBounds);
  Value aVec1 = rewriter.create<vector::TransferReadOp>(
      loc, i8x16, quantized,
      ValueRange{row, rewriter.create<arith::AddIOp>(
                          loc, dotBlockBase,
                          rewriter.create<arith::ConstantIndexOp>(loc, 16))},
      rowMap, inBounds);
  Value wBase = rewriter.create<arith::AddIOp>(
      loc, scaleOffset, rewriter.create<arith::ConstantIndexOp>(loc, 2));
  Value wVec0 = rewriter.create<vector::TransferReadOp>(
      loc, i8x16, weight, ValueRange{column, wBase}, rowMap, inBounds);
  Value wVec1 = rewriter.create<vector::TransferReadOp>(
      loc, i8x16, weight,
      ValueRange{column, rewriter.create<arith::AddIOp>(
                              loc, wBase,
                              rewriter.create<arith::ConstantIndexOp>(loc, 16))},
      rowMap, inBounds);
  auto sdot0 = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, i32x4, StringAttr::get(rewriter.getContext(),
                                  "llvm.aarch64.neon.sdot"),
      ValueRange{zeroVec, aVec0, wVec0});
  auto sdot1 = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, i32x4, StringAttr::get(rewriter.getContext(),
                                  "llvm.aarch64.neon.sdot"),
      ValueRange{sdot0.getResult(0), aVec1, wVec1});
  Value dot = rewriter.create<vector::ReductionOp>(
      loc, vector::CombiningKind::ADD, sdot1.getResult(0));
#else
  Value dotInit =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  auto dotLoop = rewriter.create<scf::ForOp>(
      loc, zero, blockSize, one, ValueRange{dotInit});
  rewriter.setInsertionPointToStart(dotLoop.getBody());
  Value dotIndex = dotLoop.getInductionVar();
  Value dotSource = rewriter.create<arith::AddIOp>(loc, dotBlockBase, dotIndex);
  Value aByte = rewriter.create<memref::LoadOp>(
      loc, quantized, ValueRange{row, dotSource});
  Value aInt = rewriter.create<arith::ExtSIOp>(loc, i32, aByte);
  Value weightIndex = rewriter.create<arith::AddIOp>(
      loc, scaleOffset,
      rewriter.create<arith::AddIOp>(
          loc, rewriter.create<arith::ConstantIndexOp>(loc, 2), dotIndex));
  Value wByte =
      rewriter.create<memref::LoadOp>(loc, weight, ValueRange{column, weightIndex});
  Value wInt = rewriter.create<arith::ExtSIOp>(loc, i32, wByte);
  Value product = rewriter.create<arith::MulIOp>(loc, aInt, wInt);
  Value dotSum = rewriter.create<arith::AddIOp>(
      loc, dotLoop.getRegionIterArg(0), product);
  rewriter.create<scf::YieldOp>(loc, dotSum);
  rewriter.setInsertionPointAfter(dotLoop);
  Value dot = dotLoop.getResult(0);
#endif

  Value activationScale = rewriter.create<memref::LoadOp>(
      loc, scales, ValueRange{row, block});
  Value dotF = rewriter.create<arith::SIToFPOp>(loc, f32, dot);
  Value contribution = rewriter.create<arith::MulFOp>(
      loc, dotF,
      rewriter.create<arith::MulFOp>(loc, activationScale, weightScale));
  Value acc =
      rewriter.create<arith::AddFOp>(loc, blockLoop.getRegionIterArg(0),
                                     contribution);
  rewriter.create<scf::YieldOp>(loc, acc);
  rewriter.setInsertionPointAfter(blockLoop);

  Value result = blockLoop.getResult(0);
  if (resultType.getElementType().isBF16())
    result =
        rewriter.create<arith::TruncFOp>(loc, rewriter.getBF16Type(), result);
  rewriter.create<memref::StoreOp>(loc, result, output,
                                   ValueRange{row, column});
  rewriter.setInsertionPointAfter(rowLoop);
  rewriter.setInsertionPointAfter(columnLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    true);
}

Value emitVectorMatmul(PatternRewriter &rewriter, Location loc, Value input,
                       Value weight, RankedTensorType resultType,
                       bool transposeB) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<RankedTensorType>(weight.getType());
  const int64_t channels = resultType.getDimSize(1);
  const int64_t reduction = inputType.getDimSize(1);
  if (ShapedType::isDynamic(channels) || ShapedType::isDynamic(reduction) ||
      channels % kVectorTile != 0 || weightType.getRank() != 2)
    return Value();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value tile = rewriter.create<arith::ConstantIndexOp>(loc, kVectorTile);
  Value channelsValue = rewriter.create<arith::ConstantIndexOp>(loc, channels);
  Value reductionValue =
      rewriter.create<arith::ConstantIndexOp>(loc, reduction);
  Value rows = rewriter.create<tensor::DimOp>(loc, input, 0);

  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output =
      rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{rows});
  auto f32 = rewriter.getF32Type();
  auto vectorType = VectorType::get({kVectorTile}, f32);

  auto columnLoop =
      rewriter.create<scf::ForOp>(loc, zero, channelsValue, tile, ValueRange{});
  rewriter.setInsertionPointToStart(columnLoop.getBody());
  columnLoop->setAttr("llmcp.parallel", UnitAttr::get(rewriter.getContext()));
  Value column = columnLoop.getInductionVar();
  auto rowLoop =
      rewriter.create<scf::ForOp>(loc, zero, rows, one, ValueRange{});
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value row = rowLoop.getInductionVar();

  Value zeroF =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  Value accInit = rewriter.create<vector::BroadcastOp>(loc, vectorType, zeroF);
  auto reductionLoop = rewriter.create<scf::ForOp>(loc, zero, reductionValue,
                                                   one, ValueRange{accInit});
  rewriter.setInsertionPointToStart(reductionLoop.getBody());
  Value reductionIndex = reductionLoop.getInductionVar();
  Value a = rewriter.create<tensor::ExtractOp>(loc, input,
                                               ValueRange{row, reductionIndex});
  if (a.getType().isBF16())
    a = rewriter.create<arith::ExtFOp>(loc, f32, a);

  auto bf16Vector = VectorType::get({kVectorTile}, weightType.getElementType());
  AffineMap bMap = transposeB
                       ? AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                                        rewriter.getContext())
                       : AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0)},
                                        rewriter.getContext());
  Value bIndices0 = transposeB ? reductionIndex : column;
  Value bIndices1 = transposeB ? column : reductionIndex;
  SmallVector<bool> inBounds = {true};
  Value bVec = rewriter.create<vector::TransferReadOp>(
      loc, bf16Vector, weight, ValueRange{bIndices0, bIndices1}, bMap,
      inBounds);
  if (cast<VectorType>(bVec.getType()).getElementType().isBF16())
    bVec = rewriter.create<arith::ExtFOp>(loc, vectorType, bVec);

  Value aVec = rewriter.create<vector::BroadcastOp>(loc, vectorType, a);
  Value acc = rewriter.create<vector::FMAOp>(loc, aVec, bVec,
                                             reductionLoop.getRegionIterArg(0));
  rewriter.create<scf::YieldOp>(loc, acc);
  rewriter.setInsertionPointAfter(reductionLoop);

  Value result = reductionLoop.getResult(0);
  if (resultType.getElementType().isBF16())
    result = rewriter.create<arith::TruncFOp>(
        loc, VectorType::get({kVectorTile}, rewriter.getBF16Type()), result);
  AffineMap outputMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                                       rewriter.getContext());
  rewriter.create<vector::TransferWriteOp>(
      loc, result, output, ValueRange{row, column}, outputMap, inBounds);
  rewriter.setInsertionPointAfter(rowLoop);
  rewriter.setInsertionPointAfter(columnLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    /*restrict=*/true);
}

Value emitVectorDotMatmul(PatternRewriter &rewriter, Location loc, Value input,
                          Value weight, RankedTensorType resultType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<RankedTensorType>(weight.getType());
  const int64_t channels = resultType.getDimSize(1);
  const int64_t reduction = inputType.getDimSize(1);
  if (ShapedType::isDynamic(channels) || ShapedType::isDynamic(reduction) ||
      reduction % kVectorTile != 0 || weightType.getRank() != 2)
    return Value();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value tile = rewriter.create<arith::ConstantIndexOp>(loc, kVectorTile);
  Value channelsValue = rewriter.create<arith::ConstantIndexOp>(loc, channels);
  Value reductionValue =
      rewriter.create<arith::ConstantIndexOp>(loc, reduction);
  Value rows = rewriter.create<tensor::DimOp>(loc, input, 0);
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output =
      rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{rows});
  auto f32 = rewriter.getF32Type();
  auto vectorType = VectorType::get({kVectorTile}, f32);
  auto bf16Vector = VectorType::get({kVectorTile}, inputType.getElementType());
  SmallVector<bool> inBounds = {true};
  AffineMap rowMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                                    rewriter.getContext());

  auto columnLoop =
      rewriter.create<scf::ForOp>(loc, zero, channelsValue, one, ValueRange{});
  rewriter.setInsertionPointToStart(columnLoop.getBody());
  columnLoop->setAttr("llmcp.parallel", UnitAttr::get(rewriter.getContext()));
  Value column = columnLoop.getInductionVar();
  auto rowLoop =
      rewriter.create<scf::ForOp>(loc, zero, rows, one, ValueRange{});
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value row = rowLoop.getInductionVar();

  Value zeroF =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  auto reductionLoop = rewriter.create<scf::ForOp>(loc, zero, reductionValue,
                                                   tile, ValueRange{zeroF});
  rewriter.setInsertionPointToStart(reductionLoop.getBody());
  Value block = reductionLoop.getInductionVar();
  Value aVec = rewriter.create<vector::TransferReadOp>(
      loc, bf16Vector, input, ValueRange{row, block}, rowMap, inBounds);
  Value bVec = rewriter.create<vector::TransferReadOp>(
      loc, bf16Vector, weight, ValueRange{column, block}, rowMap, inBounds);
  Value aF32 = rewriter.create<arith::ExtFOp>(loc, vectorType, aVec);
  Value bF32 = rewriter.create<arith::ExtFOp>(loc, vectorType, bVec);
  Value product = rewriter.create<arith::MulFOp>(loc, aF32, bF32);
  Value dot = rewriter.create<vector::ReductionOp>(
      loc, vector::CombiningKind::ADD, product);
  Value sum = rewriter.create<arith::AddFOp>(
      loc, reductionLoop.getRegionIterArg(0), dot);
  rewriter.create<scf::YieldOp>(loc, sum);
  rewriter.setInsertionPointAfter(reductionLoop);

  Value result = reductionLoop.getResult(0);
  if (resultType.getElementType().isBF16())
    result =
        rewriter.create<arith::TruncFOp>(loc, rewriter.getBF16Type(), result);
  rewriter.create<memref::StoreOp>(loc, result, output,
                                   ValueRange{row, column});
  rewriter.setInsertionPointAfter(rowLoop);
  rewriter.setInsertionPointAfter(columnLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    /*restrict=*/true);
}

#if defined(__aarch64__)
Value emitBfDotMatmul(PatternRewriter &rewriter, Location loc, Value input,
                      Value weight, RankedTensorType resultType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<RankedTensorType>(weight.getType());
  const int64_t channels = resultType.getDimSize(1);
  const int64_t reduction = inputType.getDimSize(1);
  if (ShapedType::isDynamic(channels) || ShapedType::isDynamic(reduction) ||
      reduction % 8 != 0 ||
      inputType.getElementType() != rewriter.getBF16Type() ||
      weightType.getElementType() != rewriter.getBF16Type())
    return Value();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value blockSize = rewriter.create<arith::ConstantIndexOp>(loc, 8);
  Value channelsValue = rewriter.create<arith::ConstantIndexOp>(loc, channels);
  Value reductionValue =
      rewriter.create<arith::ConstantIndexOp>(loc, reduction);
  Value rows = rewriter.create<tensor::DimOp>(loc, input, 0);
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output =
      rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{rows});

  auto f32 = rewriter.getF32Type();
  auto bf16 = rewriter.getBF16Type();
  auto bf16Vector = VectorType::get({8}, bf16);
  auto f32Vector = VectorType::get({4}, f32);
  SmallVector<bool> inBounds = {true};
  AffineMap rowMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                                    rewriter.getContext());

  auto columnLoop =
      rewriter.create<scf::ForOp>(loc, zero, channelsValue, one, ValueRange{});
  rewriter.setInsertionPointToStart(columnLoop.getBody());
  columnLoop->setAttr("llmcp.parallel", UnitAttr::get(rewriter.getContext()));
  Value column = columnLoop.getInductionVar();
  auto rowLoop =
      rewriter.create<scf::ForOp>(loc, zero, rows, one, ValueRange{});
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  Value row = rowLoop.getInductionVar();

  Value zeroF =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  Value accInit = rewriter.create<vector::BroadcastOp>(loc, f32Vector, zeroF);
  auto reductionLoop = rewriter.create<scf::ForOp>(
      loc, zero, reductionValue, blockSize, ValueRange{accInit});
  rewriter.setInsertionPointToStart(reductionLoop.getBody());
  Value block = reductionLoop.getInductionVar();
  Value aVec = rewriter.create<vector::TransferReadOp>(
      loc, bf16Vector, input, ValueRange{row, block}, rowMap, inBounds);
  Value bVec = rewriter.create<vector::TransferReadOp>(
      loc, bf16Vector, weight, ValueRange{column, block}, rowMap, inBounds);
  auto bfDot = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, f32Vector,
      StringAttr::get(rewriter.getContext(), "llvm.aarch64.neon.bfdot"),
      ValueRange{reductionLoop.getRegionIterArg(0), aVec, bVec});
  Value acc = bfDot.getResult(0);
  rewriter.create<scf::YieldOp>(loc, acc);
  rewriter.setInsertionPointAfter(reductionLoop);

  Value sum = rewriter.create<vector::ReductionOp>(
      loc, vector::CombiningKind::ADD, reductionLoop.getResult(0));
  Value result = sum;
  if (resultType.getElementType().isBF16())
    result = rewriter.create<arith::TruncFOp>(loc, bf16, result);
  rewriter.create<memref::StoreOp>(loc, result, output,
                                   ValueRange{row, column});
  rewriter.setInsertionPointAfter(rowLoop);
  rewriter.setInsertionPointAfter(columnLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    /*restrict=*/true);
}
#endif

Value castToF32(PatternRewriter &rewriter, Location loc, Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto f32Type = rewriter.getF32Type();
  auto resultType = RankedTensorType::get(inputType.getShape(), f32Type);
  Value empty = createEmpty(rewriter, loc, resultType, input);
  SmallVector<AffineMap> maps = {rewriter.getMultiDimIdentityMap(2),
                                 rewriter.getMultiDimIdentityMap(2)};
  SmallVector<utils::IteratorType> iterators(2, utils::IteratorType::parallel);
  auto cast = rewriter.create<linalg::GenericOp>(
      loc, resultType, ValueRange{input}, ValueRange{empty}, maps, iterators,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        Value value = builder.create<arith::ExtFOp>(loc, f32Type, args[0]);
        builder.create<linalg::YieldOp>(loc, value);
      });
  return cast.getResult(0);
}

class LinearOpLowering : public OpRewritePattern<LinearOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LinearOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getOutput().getType());
    auto input = op.getInput();
    if (op.getWeightFormat() == "q8_0") {
      if (Value q8 =
              emitQ8Matmul(rewriter, loc, input, op.getWeight(), resultType)) {
        rewriter.replaceOp(op, q8);
        return success();
      }
    }
    Value weight = toTensor(rewriter, loc, op.getWeight());

    if (!op.getTransposeB()) {
#if defined(__aarch64__)
      if (Value vectorized =
              emitBfDotMatmul(rewriter, loc, input, weight, resultType)) {
        rewriter.replaceOp(op, vectorized);
        return success();
      }
#endif
      if (Value vectorized =
              emitVectorDotMatmul(rewriter, loc, input, weight, resultType)) {
        rewriter.replaceOp(op, vectorized);
        return success();
      }
    }
    if (Value vectorized = emitVectorMatmul(rewriter, loc, input, weight,
                                            resultType, op.getTransposeB())) {
      rewriter.replaceOp(op, vectorized);
      return success();
    }

    auto f32Type = rewriter.getF32Type();
    auto accType = RankedTensorType::get(resultType.getShape(), f32Type);
    Value acc = createEmpty(rewriter, loc, accType, input);
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value accInit =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{acc})
            .getResult(0);

    Value matmul;
    if (op.getTransposeB()) {
      auto mm = rewriter.create<linalg::MatmulOp>(loc, TypeRange{accType},
                                                  ValueRange{input, weight},
                                                  ValueRange{accInit});
      matmul = mm.getResult(0);
    } else {
      auto mm = rewriter.create<linalg::MatmulTransposeBOp>(
          loc, TypeRange{accType}, ValueRange{input, weight},
          ValueRange{accInit});
      matmul = mm.getResult(0);
    }

    Value output = createEmpty(rewriter, loc, resultType, input);
    SmallVector<AffineMap> maps = {rewriter.getMultiDimIdentityMap(2),
                                   rewriter.getMultiDimIdentityMap(2)};
    SmallVector<utils::IteratorType> iterators(2,
                                               utils::IteratorType::parallel);
    auto cast = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{matmul}, ValueRange{output}, maps,
        iterators, [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value value = builder.create<arith::TruncFOp>(
              loc, resultType.getElementType(), args[0]);
          builder.create<linalg::YieldOp>(loc, value);
        });
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

class RmsNormOpLowering : public OpRewritePattern<RmsNormOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RmsNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !resultType || inputType.getRank() != 2)
      return failure();

    const int64_t hidden = inputType.getDimSize(1);
    if (ShapedType::isDynamic(hidden))
      return failure();

    auto f32 = rewriter.getF32Type();
    auto bf16 = rewriter.getBF16Type();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value hiddenValue = rewriter.create<arith::ConstantIndexOp>(loc, hidden);
    Value rows = rewriter.create<tensor::DimOp>(loc, op.getInput(), 0);

    auto outputType = MemRefType::get({ShapedType::kDynamic, hidden},
                                      resultType.getElementType());
    Value output =
        rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{rows});

    // One parallel region per RMSNorm: each row performs the reduction,
    // rstd computation, and scaling without leaving the region.
    auto parallel = rewriter.create<scf::ParallelOp>(
        loc, ValueRange{zero}, ValueRange{rows}, ValueRange{one});
    rewriter.setInsertionPointToStart(parallel.getBody());
    Value row = parallel.getInductionVars().front();
    Value sumInit =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    auto sumLoop = rewriter.create<scf::ForOp>(loc, zero, hiddenValue, one,
                                               ValueRange{sumInit});
    rewriter.setInsertionPointToStart(sumLoop.getBody());
    Value d = sumLoop.getInductionVar();
    Value x = rewriter.create<tensor::ExtractOp>(loc, op.getInput(),
                                                 ValueRange{row, d});
    if (x.getType().isBF16())
      x = rewriter.create<arith::ExtFOp>(loc, f32, x);
    Value square = rewriter.create<arith::MulFOp>(loc, x, x);
    Value sum = rewriter.create<arith::AddFOp>(loc, sumLoop.getRegionIterArg(0),
                                               square);
    rewriter.create<scf::YieldOp>(loc, sum);
    rewriter.setInsertionPointAfter(sumLoop);

    Value dimValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(static_cast<float>(hidden)));
    Value mean =
        rewriter.create<arith::DivFOp>(loc, sumLoop.getResult(0), dimValue);
    Value eps = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(op.getEpsAttr().getValueAsDouble()));
    Value variance = rewriter.create<arith::AddFOp>(loc, mean, eps);
    Value rstd = rewriter.create<math::RsqrtOp>(loc, variance);

    auto scaleLoop =
        rewriter.create<scf::ForOp>(loc, zero, hiddenValue, one, ValueRange{});
    rewriter.setInsertionPointToStart(scaleLoop.getBody());
    Value scaleD = scaleLoop.getInductionVar();
    Value value = rewriter.create<tensor::ExtractOp>(loc, op.getInput(),
                                                     ValueRange{row, scaleD});
    if (value.getType().isBF16())
      value = rewriter.create<arith::ExtFOp>(loc, f32, value);
    Value weight = rewriter.create<memref::LoadOp>(loc, op.getWeight(),
                                                   ValueRange{scaleD});
    if (weight.getType().isBF16())
      weight = rewriter.create<arith::ExtFOp>(loc, f32, weight);
    if (op.getZeroCentered()) {
      Value oneF = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(1.0f));
      weight = rewriter.create<arith::AddFOp>(loc, weight, oneF);
    }
    Value normalized = rewriter.create<arith::MulFOp>(
        loc, rewriter.create<arith::MulFOp>(loc, value, rstd), weight);
    Value result = normalized;
    if (resultType.getElementType().isBF16())
      result = rewriter.create<arith::TruncFOp>(loc, bf16, result);
    rewriter.create<memref::StoreOp>(loc, result, output,
                                     ValueRange{row, scaleD});
    rewriter.setInsertionPointAfter(scaleLoop);
    rewriter.setInsertionPointAfter(parallel);

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, output, /*restrict=*/true);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

class AddOpLowering : public OpRewritePattern<AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value output = createEmpty(rewriter, loc, resultType, op.getLhs());
    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(2));
    SmallVector<utils::IteratorType> iterators(2,
                                               utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{output}, maps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value value = builder.create<arith::AddFOp>(loc, args[0], args[1]);
          builder.create<linalg::YieldOp>(loc, value);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

class SiluMulOpLowering : public OpRewritePattern<SiluMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SiluMulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto f32Type = rewriter.getF32Type();
    Value output = createEmpty(rewriter, loc, resultType, op.getGate());
    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(2));
    SmallVector<utils::IteratorType> iterators(2,
                                               utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getGate(), op.getUp()},
        ValueRange{output}, maps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value gate = args[0];
          Value up = args[1];
          if (gate.getType().isBF16()) {
            gate = builder.create<arith::ExtFOp>(loc, f32Type, gate);
            up = builder.create<arith::ExtFOp>(loc, f32Type, up);
          }
          Value negated = builder.create<arith::NegFOp>(loc, gate);
          Value exponent = builder.create<math::ExpOp>(loc, negated);
          Value one = builder.create<arith::ConstantOp>(
              loc, builder.getF32FloatAttr(1.0f));
          Value denominator = builder.create<arith::AddFOp>(loc, one, exponent);
          Value sigmoid = builder.create<arith::DivFOp>(loc, one, denominator);
          Value silu = builder.create<arith::MulFOp>(loc, gate, sigmoid);
          Value value = builder.create<arith::MulFOp>(loc, silu, up);
          Value truncated = builder.create<arith::TruncFOp>(
              loc, resultType.getElementType(), value);
          builder.create<linalg::YieldOp>(loc, truncated);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

class EmbeddingOpLowering : public OpRewritePattern<EmbeddingOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(EmbeddingOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value table = toTensor(rewriter, loc, op.getTable());
    Value output = createEmpty(rewriter, loc, resultType, op.getTokens());
    SmallVector<AffineMap> maps = {
        AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0)},
                       rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(2)};
    SmallVector<utils::IteratorType> iterators(2,
                                               utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getTokens()}, ValueRange{output}, maps,
        iterators, [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value row = builder.create<linalg::IndexOp>(loc, 0);
          Value column = builder.create<linalg::IndexOp>(loc, 1);
          Value token = builder.create<tensor::ExtractOp>(loc, op.getTokens(),
                                                          ValueRange{row});
          Value value = builder.create<tensor::ExtractOp>(
              loc, table, ValueRange{token, column});
          builder.create<linalg::YieldOp>(loc, value);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

class GatedRmsNormOpLowering : public OpRewritePattern<GatedRmsNormOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GatedRmsNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !resultType || inputType.getRank() != 2)
      return failure();

    const int64_t hidden = inputType.getDimSize(1);
    const int64_t group = op.getGroupSize();
    if (ShapedType::isDynamic(hidden) || group <= 0 || hidden % group != 0)
      return failure();
    const int64_t groups = hidden / group;

    const int64_t batchDim = inputType.getDimSize(0);
    auto expandedType = RankedTensorType::get({batchDim, groups, group},
                                              inputType.getElementType());
    Value batchSize =
        rewriter.createOrFold<tensor::DimOp>(loc, op.getInput(), 0);
    SmallVector<ReassociationIndices> reassociation = {{0}, {1, 2}};
    SmallVector<OpFoldResult> expandedShape = {
        batchSize, rewriter.getIndexAttr(groups), rewriter.getIndexAttr(group)};

    auto expand = [&](Value value) -> Value {
      return rewriter.create<tensor::ExpandShapeOp>(
          loc, expandedType, value, reassociation, expandedShape);
    };
    Value input = expand(op.getInput());
    Value gate = expand(op.getGate());
    Value weight = toTensor(rewriter, loc, op.getWeight());

    auto f32Type = rewriter.getF32Type();
    auto f32ExpandedType =
        RankedTensorType::get({batchDim, groups, group}, f32Type);
    Value inputF32 = createEmpty(rewriter, loc, f32ExpandedType, input);
    Value gateF32 = createEmpty(rewriter, loc, f32ExpandedType, input);

    SmallVector<AffineMap> castMaps = {rewriter.getMultiDimIdentityMap(3),
                                       rewriter.getMultiDimIdentityMap(3)};
    SmallVector<utils::IteratorType> castIters(3,
                                               utils::IteratorType::parallel);
    auto castGeneric = rewriter.create<linalg::GenericOp>(
        loc, f32ExpandedType, ValueRange{input}, ValueRange{inputF32}, castMaps,
        castIters, [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value value = builder.create<arith::ExtFOp>(loc, f32Type, args[0]);
          builder.create<linalg::YieldOp>(loc, value);
        });
    auto gateGeneric = rewriter.create<linalg::GenericOp>(
        loc, f32ExpandedType, ValueRange{gate}, ValueRange{gateF32}, castMaps,
        castIters, [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value value = builder.create<arith::ExtFOp>(loc, f32Type, args[0]);
          builder.create<linalg::YieldOp>(loc, value);
        });

    auto groupType = RankedTensorType::get({batchDim, groups}, f32Type);
    Value rowEmpty = createEmpty(rewriter, loc, groupType, op.getInput());
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value sumInit =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{rowEmpty})
            .getResult(0);

    SmallVector<AffineMap> sumMaps = {
        rewriter.getMultiDimIdentityMap(3),
        AffineMap::get(
            3, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
            rewriter.getContext())};
    SmallVector<utils::IteratorType> sumIters = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::reduction};
    auto sum = rewriter.create<linalg::GenericOp>(
        loc, groupType, ValueRange{castGeneric.getResult(0)},
        ValueRange{sumInit}, sumMaps, sumIters,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value square = builder.create<arith::MulFOp>(loc, args[0], args[0]);
          Value value = builder.create<arith::AddFOp>(loc, args[1], square);
          builder.create<linalg::YieldOp>(loc, value);
        });

    Value rstdEmpty = createEmpty(rewriter, loc, groupType, op.getInput());
    SmallVector<AffineMap> statMaps = {rewriter.getMultiDimIdentityMap(2),
                                       rewriter.getMultiDimIdentityMap(2)};
    SmallVector<utils::IteratorType> statIters(2,
                                               utils::IteratorType::parallel);
    auto rstd = rewriter.create<linalg::GenericOp>(
        loc, groupType, ValueRange{sum.getResult(0)}, ValueRange{rstdEmpty},
        statMaps, statIters,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value dim = builder.create<arith::ConstantOp>(
              loc, builder.getF32FloatAttr(static_cast<float>(group)));
          Value mean = builder.create<arith::DivFOp>(loc, args[0], dim);
          Value eps = builder.create<arith::ConstantOp>(
              loc, builder.getF32FloatAttr(op.getEpsAttr().getValueAsDouble()));
          Value variance = builder.create<arith::AddFOp>(loc, mean, eps);
          Value value = builder.create<math::RsqrtOp>(loc, variance);
          builder.create<linalg::YieldOp>(loc, value);
        });

    Value outputExpanded = createEmpty(rewriter, loc, expandedType, input);
    SmallVector<AffineMap> maps = {
        rewriter.getMultiDimIdentityMap(3), rewriter.getMultiDimIdentityMap(3),
        AffineMap::get(3, 0, {rewriter.getAffineDimExpr(2)},
                       rewriter.getContext()),
        AffineMap::get(
            3, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
            rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(3)};
    SmallVector<utils::IteratorType> iterators(3,
                                               utils::IteratorType::parallel);
    auto scaled = rewriter.create<linalg::GenericOp>(
        loc, expandedType,
        ValueRange{castGeneric.getResult(0), gateGeneric.getResult(0), weight,
                   rstd.getResult(0)},
        ValueRange{outputExpanded}, maps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value gate = args[1];
          Value negated = builder.create<arith::NegFOp>(loc, gate);
          Value exponent = builder.create<math::ExpOp>(loc, negated);
          Value one = builder.create<arith::ConstantOp>(
              loc, builder.getF32FloatAttr(1.0f));
          Value denominator = builder.create<arith::AddFOp>(loc, one, exponent);
          Value sigmoid = builder.create<arith::DivFOp>(loc, one, denominator);
          Value silu = builder.create<arith::MulFOp>(loc, gate, sigmoid);
          Value normalized = builder.create<arith::MulFOp>(loc, args[0], silu);
          Value value = builder.create<arith::MulFOp>(loc, normalized, args[2]);
          Value scaledValue =
              builder.create<arith::MulFOp>(loc, value, args[3]);
          Value truncated = builder.create<arith::TruncFOp>(
              loc, resultType.getElementType(), scaledValue);
          builder.create<linalg::YieldOp>(loc, truncated);
        });

    SmallVector<ReassociationIndices> collapseReassociation = {{0}, {1, 2}};
    auto collapsed = rewriter.create<tensor::CollapseShapeOp>(
        loc, resultType, scaled.getResult(0), collapseReassociation);
    rewriter.replaceOp(op, collapsed.getResult());
    return success();
  }
};

class LmHeadOpLowering : public OpRewritePattern<LmHeadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LmHeadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getLogits().getType());
    if (op.getWeightFormat() == "q8_0") {
      if (Value q8 = emitQ8Matmul(rewriter, loc, op.getHidden(), op.getTable(),
                                  resultType)) {
        rewriter.replaceOp(op, q8);
        return success();
      }
    }
    Value table = toTensor(rewriter, loc, op.getTable());
#if defined(__aarch64__)
    if (Value vectorized =
            emitBfDotMatmul(rewriter, loc, op.getHidden(), table, resultType)) {
      rewriter.replaceOp(op, vectorized);
      return success();
    }
#endif
    if (Value vectorized = emitVectorDotMatmul(rewriter, loc, op.getHidden(),
                                               table, resultType)) {
      rewriter.replaceOp(op, vectorized);
      return success();
    }
    if (Value vectorized =
            emitVectorMatmul(rewriter, loc, op.getHidden(), table, resultType,
                             /*transposeB=*/false)) {
      rewriter.replaceOp(op, vectorized);
      return success();
    }
    Value output = createEmpty(rewriter, loc, resultType, op.getHidden());
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value outputInit =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{output})
            .getResult(0);
    auto matmul = rewriter.create<linalg::MatmulTransposeBOp>(
        loc, TypeRange{resultType}, ValueRange{op.getHidden(), table},
        ValueRange{outputInit});
    rewriter.replaceOp(op, matmul.getResult(0));
    return success();
  }
};

struct LowerToLinalgPass
    : public PassWrapper<LowerToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLinalgPass)

  StringRef getArgument() const final { return "llmcp-lower-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower llmcp.linear and llmcp.rms_norm to linalg";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    context->getOrLoadDialect<bufferization::BufferizationDialect>();
    context->getOrLoadDialect<linalg::LinalgDialect>();
    context->getOrLoadDialect<tensor::TensorDialect>();
    context->getOrLoadDialect<arith::ArithDialect>();
    context->getOrLoadDialect<math::MathDialect>();
    context->getOrLoadDialect<vector::VectorDialect>();
    context->getOrLoadDialect<LLVM::LLVMDialect>();

    RewritePatternSet patterns(context);
    populateCausalConv1dLoweringPatterns(patterns);
    populateGatedDeltaNetLoweringPatterns(patterns);
    populateAttentionLoweringPatterns(patterns);
    patterns.add<LinearOpLowering, RmsNormOpLowering, AddOpLowering,
                 SiluMulOpLowering, EmbeddingOpLowering, GatedRmsNormOpLowering,
                 LmHeadOpLowering>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::llmcp::createLowerToLinalgPass() {
  return std::make_unique<LowerToLinalgPass>();
}

void mlir::llmcp::registerLowerToLinalgPass() {
  PassRegistration<LowerToLinalgPass>();
}
