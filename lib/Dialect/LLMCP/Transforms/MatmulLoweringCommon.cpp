//===- MatmulLoweringCommon.cpp - Shared matmul lowering ------------------===//
//
// Kernel emitters and the shared prefill/decode loop scheduler.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/MatmulLoweringInternal.h"

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
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::llmcp;
using llvm::StringRef;

namespace mlir {
namespace llmcp {
namespace {

Value toTensor(PatternRewriter &rewriter, Location loc, Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());
  auto tensorType =
      RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
  return rewriter.create<bufferization::ToTensorOp>(loc, tensorType, memref,
                                                    /*restrict=*/true);
}

constexpr int64_t kVectorTile = 32;

struct MatmulLoops {
  scf::ForOp rowLoop;
  scf::ForOp columnLoop;
  Value row;
  Value column;
};

/// Creates a matmul loop nest. Decode is a GEMV: the row loop is omitted and
/// the column loop is parallelized. Prefill and generic paths keep all rows and
/// parallelize output columns.
MatmulLoops createMatmulLoops(PatternRewriter &rewriter, Location loc,
                              Value zero, Value rows, Value one,
                              Value columns, Value step, bool decode) {
  MatmulLoops loops;
  loops.columnLoop = rewriter.create<scf::ForOp>(loc, zero, columns, step,
                                                ValueRange{});
  loops.columnLoop->setAttr("llmcp.parallel",
                            UnitAttr::get(rewriter.getContext()));
  rewriter.setInsertionPointToStart(loops.columnLoop.getBody());
  loops.column = loops.columnLoop.getInductionVar();
  if (decode) {
    loops.row = zero;
    return loops;
  }
  loops.rowLoop = rewriter.create<scf::ForOp>(loc, zero, rows, one,
                                             ValueRange{});
  rewriter.setInsertionPointToStart(loops.rowLoop.getBody());
  loops.row = loops.rowLoop.getInductionVar();
  return loops;
}

Value applyActivation(PatternRewriter &rewriter, Location loc, Value value,
                      StringRef activation) {
  if (activation == "none")
    return value;
  auto f32 = rewriter.getF32Type();
  Value one =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
  if (activation == "silu") {
    Value sigmoid = rewriter.create<arith::DivFOp>(
        loc, one,
        rewriter.create<arith::AddFOp>(
            loc, one,
            rewriter.create<math::ExpOp>(
                loc, rewriter.create<arith::NegFOp>(loc, value))));
    return rewriter.create<arith::MulFOp>(loc, value, sigmoid);
  }
  if (activation == "sigmoid")
    return rewriter.create<arith::DivFOp>(
        loc, one,
        rewriter.create<arith::AddFOp>(
            loc, one,
            rewriter.create<math::ExpOp>(
                loc, rewriter.create<arith::NegFOp>(loc, value))));
  return rewriter.create<math::Log1pOp>(
      loc, rewriter.create<math::ExpOp>(loc, value));
}

Value emitQ8Matmul(PatternRewriter &rewriter, Location loc, Value input,
                   Value weight, RankedTensorType resultType, bool decode,
                   StringRef activation) {
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
  Value outputRows = decode ? one : rows;
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output =
      rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{outputRows});

  // Pre-quantize activations once: Aq[rows, K] and As[rows, K/32].
  Value quantized = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get({ShapedType::kDynamic, reduction}, i8),
      ValueRange{outputRows});
  Value scales = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get({ShapedType::kDynamic, reduction / 32}, f32),
      ValueRange{outputRows});
  auto rowQuantLoop =
      rewriter.create<scf::ForOp>(loc, zero, outputRows, one, ValueRange{});
  rowQuantLoop->setAttr("llmcp.parallel", UnitAttr::get(rewriter.getContext()));
  rewriter.setInsertionPointToStart(rowQuantLoop.getBody());
  Value quantRow = rowQuantLoop.getInductionVar();
  auto quantBlockLoop =
      rewriter.create<scf::ForOp>(loc, zero, blockCount, one, ValueRange{});
  rewriter.setInsertionPointToStart(quantBlockLoop.getBody());
  Value quantBlock = quantBlockLoop.getInductionVar();
  Value blockBase = rewriter.create<arith::MulIOp>(loc, quantBlock, blockSize);
  Value maxInit =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  auto maxLoop = rewriter.create<scf::ForOp>(loc, zero, blockSize, one,
                                             ValueRange{maxInit});
  rewriter.setInsertionPointToStart(maxLoop.getBody());
  Value maxIndex = maxLoop.getInductionVar();
  Value sourceIndex = rewriter.create<arith::AddIOp>(loc, blockBase, maxIndex);
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

  auto quantizeLoop =
      rewriter.create<scf::ForOp>(loc, zero, blockSize, one, ValueRange{});
  rewriter.setInsertionPointToStart(quantizeLoop.getBody());
  Value quantIndex = quantizeLoop.getInductionVar();
  Value quantSource =
      rewriter.create<arith::AddIOp>(loc, blockBase, quantIndex);
  Value quantValue = rewriter.create<tensor::ExtractOp>(
      loc, input, ValueRange{quantRow, quantSource});
  if (quantValue.getType().isBF16())
    quantValue = rewriter.create<arith::ExtFOp>(loc, f32, quantValue);
  Value scaled =
      rewriter.create<arith::MulFOp>(loc, quantValue, inverse.getResult(0));
  Value rounded = rewriter.create<math::RoundEvenOp>(loc, scaled);
  Value aInt = rewriter.create<arith::FPToSIOp>(loc, i32, rounded);
  Value aByte = rewriter.create<arith::TruncIOp>(loc, i8, aInt);
  rewriter.create<memref::StoreOp>(loc, aByte, quantized,
                                   ValueRange{quantRow, quantSource});
  rewriter.setInsertionPointAfter(quantizeLoop);
  rewriter.setInsertionPointAfter(quantBlockLoop);
  rewriter.setInsertionPointAfter(rowQuantLoop);

  // Q8_0 x Q8_1 dot product.
  MatmulLoops loops =
      createMatmulLoops(rewriter, loc, zero, rows, one, channelsValue, one, decode);
  Value row = loops.row;
  Value column = loops.column;
  Value accInit =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  auto blockLoop = rewriter.create<scf::ForOp>(loc, zero, blockCount, one,
                                               ValueRange{accInit});
  rewriter.setInsertionPointToStart(blockLoop.getBody());
  Value block = blockLoop.getInductionVar();
  Value dotBlockBase = rewriter.create<arith::MulIOp>(loc, block, blockSize);
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
  AffineMap rowMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
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
      loc, i32x4,
      StringAttr::get(rewriter.getContext(), "llvm.aarch64.neon.sdot"),
      ValueRange{zeroVec, aVec0, wVec0});
  auto sdot1 = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, i32x4,
      StringAttr::get(rewriter.getContext(), "llvm.aarch64.neon.sdot"),
      ValueRange{sdot0.getResult(0), aVec1, wVec1});
  Value dot = rewriter.create<vector::ReductionOp>(
      loc, vector::CombiningKind::ADD, sdot1.getResult(0));
#else
  Value dotInit =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  auto dotLoop = rewriter.create<scf::ForOp>(loc, zero, blockSize, one,
                                             ValueRange{dotInit});
  rewriter.setInsertionPointToStart(dotLoop.getBody());
  Value dotIndex = dotLoop.getInductionVar();
  Value dotSource = rewriter.create<arith::AddIOp>(loc, dotBlockBase, dotIndex);
  Value aByte = rewriter.create<memref::LoadOp>(loc, quantized,
                                                ValueRange{row, dotSource});
  Value aInt = rewriter.create<arith::ExtSIOp>(loc, i32, aByte);
  Value weightIndex = rewriter.create<arith::AddIOp>(
      loc, scaleOffset,
      rewriter.create<arith::AddIOp>(
          loc, rewriter.create<arith::ConstantIndexOp>(loc, 2), dotIndex));
  Value wByte = rewriter.create<memref::LoadOp>(
      loc, weight, ValueRange{column, weightIndex});
  Value wInt = rewriter.create<arith::ExtSIOp>(loc, i32, wByte);
  Value product = rewriter.create<arith::MulIOp>(loc, aInt, wInt);
  Value dotSum =
      rewriter.create<arith::AddIOp>(loc, dotLoop.getRegionIterArg(0), product);
  rewriter.create<scf::YieldOp>(loc, dotSum);
  rewriter.setInsertionPointAfter(dotLoop);
  Value dot = dotLoop.getResult(0);
#endif

  Value activationScale =
      rewriter.create<memref::LoadOp>(loc, scales, ValueRange{row, block});
  Value dotF = rewriter.create<arith::SIToFPOp>(loc, f32, dot);
  Value contribution = rewriter.create<arith::MulFOp>(
      loc, dotF,
      rewriter.create<arith::MulFOp>(loc, activationScale, weightScale));
  Value acc = rewriter.create<arith::AddFOp>(loc, blockLoop.getRegionIterArg(0),
                                             contribution);
  rewriter.create<scf::YieldOp>(loc, acc);
  rewriter.setInsertionPointAfter(blockLoop);

  Value result = blockLoop.getResult(0);
  result = applyActivation(rewriter, loc, result, activation);
  if (resultType.getElementType().isBF16())
    result =
        rewriter.create<arith::TruncFOp>(loc, rewriter.getBF16Type(), result);
  rewriter.create<memref::StoreOp>(loc, result, output,
                                   ValueRange{row, column});
  rewriter.setInsertionPointAfter(loops.columnLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    true);
}

Value emitVectorMatmul(PatternRewriter &rewriter, Location loc, Value input,
                       Value weight, RankedTensorType resultType,
                       bool transposeB, bool decode, StringRef activation) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<RankedTensorType>(weight.getType());
  const int64_t channels = resultType.getDimSize(1);
  const int64_t reduction = inputType.getDimSize(1);
  if (ShapedType::isDynamic(channels) || ShapedType::isDynamic(reduction) ||
      channels % kVectorTile != 0 || weightType.getRank() != 2 ||
      activation != "none")
    return Value();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value tile = rewriter.create<arith::ConstantIndexOp>(loc, kVectorTile);
  Value channelsValue = rewriter.create<arith::ConstantIndexOp>(loc, channels);
  Value reductionValue =
      rewriter.create<arith::ConstantIndexOp>(loc, reduction);
  Value rows = rewriter.create<tensor::DimOp>(loc, input, 0);
  Value outputRows = decode ? one : rows;

  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output =
      rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{outputRows});
  auto f32 = rewriter.getF32Type();
  auto vectorType = VectorType::get({kVectorTile}, f32);

  MatmulLoops loops =
      createMatmulLoops(rewriter, loc, zero, rows, one, channelsValue, tile, decode);
  Value row = loops.row;
  Value column = loops.column;

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
  rewriter.setInsertionPointAfter(loops.columnLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    /*restrict=*/true);
}

Value emitVectorDotMatmul(PatternRewriter &rewriter, Location loc, Value input,
                          Value weight, RankedTensorType resultType, bool decode,
                   StringRef activation) {
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
  Value outputRows = decode ? one : rows;
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output =
      rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{outputRows});
  auto f32 = rewriter.getF32Type();
  auto vectorType = VectorType::get({kVectorTile}, f32);
  auto bf16Vector = VectorType::get({kVectorTile}, inputType.getElementType());
  SmallVector<bool> inBounds = {true};
  AffineMap rowMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                                    rewriter.getContext());

  MatmulLoops loops =
      createMatmulLoops(rewriter, loc, zero, rows, one, channelsValue, one, decode);
  Value row = loops.row;
  Value column = loops.column;

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
  result = applyActivation(rewriter, loc, result, activation);
  if (resultType.getElementType().isBF16())
    result =
        rewriter.create<arith::TruncFOp>(loc, rewriter.getBF16Type(), result);
  rewriter.create<memref::StoreOp>(loc, result, output,
                                   ValueRange{row, column});
  rewriter.setInsertionPointAfter(loops.columnLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    /*restrict=*/true);
}

#if defined(__aarch64__)
Value emitBfDotMatmul(PatternRewriter &rewriter, Location loc, Value input,
                      Value weight, RankedTensorType resultType, bool decode,
                   StringRef activation) {
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
  Value outputRows = decode ? one : rows;
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output =
      rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{outputRows});

  auto f32 = rewriter.getF32Type();
  auto bf16 = rewriter.getBF16Type();
  auto bf16Vector = VectorType::get({8}, bf16);
  auto f32Vector = VectorType::get({4}, f32);
  SmallVector<bool> inBounds = {true};
  AffineMap rowMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                                    rewriter.getContext());

  MatmulLoops loops =
      createMatmulLoops(rewriter, loc, zero, rows, one, channelsValue, one, decode);
  Value row = loops.row;
  Value column = loops.column;

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
  Value result = applyActivation(rewriter, loc, sum, activation);
  if (resultType.getElementType().isBF16())
    result = rewriter.create<arith::TruncFOp>(loc, bf16, result);
  rewriter.create<memref::StoreOp>(loc, result, output,
                                   ValueRange{row, column});
  rewriter.setInsertionPointAfter(loops.columnLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    /*restrict=*/true);
}
#endif

} // namespace

/// Prefill bf16 GEMM with 4-row register tiles and 32-column / 64-K cache
/// blocks.
Value emitPrefillBfDotBlocked(PatternRewriter &rewriter, Location loc,
                              Value input, Value weight,
                              RankedTensorType resultType,
                              StringRef activation) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<RankedTensorType>(weight.getType());
  const int64_t channels = resultType.getDimSize(1);
  const int64_t reduction = inputType.getDimSize(1);
  constexpr int64_t kColumnBlock = 32;
  constexpr int64_t kReductionBlock = 64;
  constexpr int64_t kRowTile = 4;
  if (ShapedType::isDynamic(channels) || ShapedType::isDynamic(reduction) ||
      channels % kColumnBlock != 0 || reduction % 8 != 0 ||
      inputType.getElementType() != rewriter.getBF16Type() ||
      weightType.getElementType() != rewriter.getBF16Type())
    return Value();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value columnBlock =
      rewriter.create<arith::ConstantIndexOp>(loc, kColumnBlock);
  Value reductionBlock =
      rewriter.create<arith::ConstantIndexOp>(loc, kReductionBlock);
  Value rowTile = rewriter.create<arith::ConstantIndexOp>(loc, kRowTile);
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
  Value zeroF =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
  Value zeroVec = rewriter.create<vector::BroadcastOp>(loc, f32Vector, zeroF);

  auto emitDot = [&](Value accumulator, Value row, Value column, Value block) {
    Value aVec = rewriter.create<vector::TransferReadOp>(
        loc, bf16Vector, input, ValueRange{row, block}, rowMap, inBounds);
    Value bVec = rewriter.create<vector::TransferReadOp>(
        loc, bf16Vector, weight, ValueRange{column, block}, rowMap, inBounds);
    return rewriter
        .create<LLVM::CallIntrinsicOp>(
            loc, f32Vector,
            StringAttr::get(rewriter.getContext(),
                            "llvm.aarch64.neon.bfdot"),
            ValueRange{accumulator, aVec, bVec})
        .getResult(0);
  };
  auto reduce = [&](Value vector) {
    return rewriter.create<vector::ReductionOp>(
        loc, vector::CombiningKind::ADD, vector);
  };
  auto store = [&](Value value, Value row, Value column) {
    value = applyActivation(rewriter, loc, value, activation);
    if (resultType.getElementType().isBF16())
      value = rewriter.create<arith::TruncFOp>(loc, bf16, value);
    rewriter.create<memref::StoreOp>(loc, value, output,
                                     ValueRange{row, column});
  };

  auto columnBlockLoop = rewriter.create<scf::ForOp>(
      loc, zero, channelsValue, columnBlock, ValueRange{});
  columnBlockLoop->setAttr("llmcp.parallel",
                           UnitAttr::get(rewriter.getContext()));
  rewriter.setInsertionPointToStart(columnBlockLoop.getBody());
  Value columnStart = columnBlockLoop.getInductionVar();
  Value columnEnd = rewriter.create<arith::MinUIOp>(
      loc,
      rewriter.create<arith::AddIOp>(loc, columnStart, columnBlock),
      channelsValue);
  Value mainRows = rewriter.create<arith::SubIOp>(
      loc, rows, rewriter.create<arith::RemUIOp>(loc, rows, rowTile));

  auto rowGroupLoop = rewriter.create<scf::ForOp>(
      loc, zero, mainRows, rowTile, ValueRange{});
  rewriter.setInsertionPointToStart(rowGroupLoop.getBody());
  Value rowStart = rowGroupLoop.getInductionVar();

  auto columnLoop = rewriter.create<scf::ForOp>(
      loc, columnStart, columnEnd, one, ValueRange{});
  rewriter.setInsertionPointToStart(columnLoop.getBody());
  Value column = columnLoop.getInductionVar();
  auto reductionBlockLoop = rewriter.create<scf::ForOp>(
      loc, zero, reductionValue, reductionBlock,
      ValueRange{zeroVec, zeroVec, zeroVec, zeroVec});
  rewriter.setInsertionPointToStart(reductionBlockLoop.getBody());
  Value reductionStart = reductionBlockLoop.getInductionVar();
  Value reductionEnd = rewriter.create<arith::MinUIOp>(
      loc,
      rewriter.create<arith::AddIOp>(loc, reductionStart, reductionBlock),
      reductionValue);
  Value step = rewriter.create<arith::ConstantIndexOp>(loc, 8);
  auto dotLoop = rewriter.create<scf::ForOp>(
      loc, reductionStart, reductionEnd, step,
      ValueRange{reductionBlockLoop.getRegionIterArgs()});
  rewriter.setInsertionPointToStart(dotLoop.getBody());
  Value block = dotLoop.getInductionVar();
  ValueRange accumulators = dotLoop.getRegionIterArgs();
  SmallVector<Value, 4> next;
  for (int64_t row = 0; row < kRowTile; ++row) {
    Value rowIndex = rewriter.create<arith::AddIOp>(
        loc, rowStart, rewriter.create<arith::ConstantIndexOp>(loc, row));
    next.push_back(emitDot(accumulators[row], rowIndex, column, block));
  }
  rewriter.create<scf::YieldOp>(loc, ValueRange(next));
  rewriter.setInsertionPointAfter(dotLoop);
  rewriter.create<scf::YieldOp>(loc,
                                ValueRange{dotLoop.getResults()});
  rewriter.setInsertionPointAfter(reductionBlockLoop);
  for (int64_t row = 0; row < kRowTile; ++row) {
    Value rowIndex = rewriter.create<arith::AddIOp>(
        loc, rowStart, rewriter.create<arith::ConstantIndexOp>(loc, row));
    store(reduce(reductionBlockLoop.getResult(row)), rowIndex, column);
  }
  rewriter.setInsertionPointAfter(columnLoop);
  rewriter.setInsertionPointAfter(rowGroupLoop);

  // Tail rows that do not fill a full register tile.
  auto tailRowLoop = rewriter.create<scf::ForOp>(
      loc, mainRows, rows, one, ValueRange{});
  rewriter.setInsertionPointToStart(tailRowLoop.getBody());
  Value tailRow = tailRowLoop.getInductionVar();
  auto tailColumnLoop = rewriter.create<scf::ForOp>(
      loc, columnStart, columnEnd, one, ValueRange{});
  rewriter.setInsertionPointToStart(tailColumnLoop.getBody());
  Value tailColumn = tailColumnLoop.getInductionVar();
  auto tailReductionBlockLoop = rewriter.create<scf::ForOp>(
      loc, zero, reductionValue, reductionBlock, ValueRange{zeroVec});
  rewriter.setInsertionPointToStart(tailReductionBlockLoop.getBody());
  Value tailReductionStart = tailReductionBlockLoop.getInductionVar();
  Value tailReductionEnd = rewriter.create<arith::MinUIOp>(
      loc,
      rewriter.create<arith::AddIOp>(loc, tailReductionStart, reductionBlock),
      reductionValue);
  auto tailDotLoop = rewriter.create<scf::ForOp>(
      loc, tailReductionStart, tailReductionEnd, step,
      ValueRange{tailReductionBlockLoop.getRegionIterArg(0)});
  rewriter.setInsertionPointToStart(tailDotLoop.getBody());
  Value tailBlock = tailDotLoop.getInductionVar();
  Value tailNext =
      emitDot(tailDotLoop.getRegionIterArg(0), tailRow, tailColumn, tailBlock);
  rewriter.create<scf::YieldOp>(loc, tailNext);
  rewriter.setInsertionPointAfter(tailDotLoop);
  rewriter.create<scf::YieldOp>(loc,
                                ValueRange{tailDotLoop.getResult(0)});
  rewriter.setInsertionPointAfter(tailReductionBlockLoop);
  store(reduce(tailReductionBlockLoop.getResult(0)), tailRow, tailColumn);
  rewriter.setInsertionPointAfter(tailColumnLoop);
  rewriter.setInsertionPointAfter(tailRowLoop);
  rewriter.setInsertionPointAfter(columnBlockLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    /*restrict=*/true);
}

/// Prefill Q8 GEMM with 4-row register tiles and 32-column / 128-K cache
/// blocks.
Value emitPrefillQ8Blocked(PatternRewriter &rewriter, Location loc, Value input,
                           Value weight, RankedTensorType resultType,
                           StringRef activation) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<MemRefType>(weight.getType());
  const int64_t reduction = inputType.getDimSize(1);
  const int64_t channels = resultType.getDimSize(1);
  constexpr int64_t kColumnBlock = 32;
  constexpr int64_t kColumnTile = 4;
  constexpr int64_t kReductionBlock = 128;
  constexpr int64_t kRowTile = 4;
  if (inputType.getRank() != 2 || weightType.getRank() != 2 ||
      ShapedType::isDynamic(reduction) || ShapedType::isDynamic(channels) ||
      reduction % 32 != 0 || channels % kColumnBlock != 0 ||
      !weightType.getElementType().isInteger(8) ||
      !inputType.getElementType().isBF16())
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
  Value reductionValue =
      rewriter.create<arith::ConstantIndexOp>(loc, reduction);
  Value rows = rewriter.create<tensor::DimOp>(loc, input, 0);
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output =
      rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{rows});
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

  const int64_t rowTile = kRowTile;
  Value rowTileValue = rewriter.create<arith::ConstantIndexOp>(loc, rowTile);
  Value columnTileValue =
      rewriter.create<arith::ConstantIndexOp>(loc, kColumnTile);
  Value columnBlock =
      rewriter.create<arith::ConstantIndexOp>(loc, kColumnBlock);
  Value reductionBlock =
      rewriter.create<arith::ConstantIndexOp>(loc, kReductionBlock);
  Value mainRows = rewriter.create<arith::SubIOp>(
      loc, rows, rewriter.create<arith::RemUIOp>(loc, rows, rowTileValue));

  auto i32x4 = VectorType::get({4}, i32);
  auto i8x16 = VectorType::get({16}, i8);
  AffineMap rowMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                                    rewriter.getContext());
  SmallVector<bool> inBounds = {true};
  Value zeroI32 = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(i32x4, rewriter.getI32IntegerAttr(0)));
  Value zeroVecF =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));

  auto emitDot = [&](Value accumulator, Value row, Value column,
                     Value qblock) -> Value {
    Value qblockIndex = rewriter.create<arith::DivUIOp>(loc, qblock,
                                                        blockSize);
    Value scaleOffset = rewriter.create<arith::MulIOp>(
        loc, qblockIndex, rewriter.create<arith::ConstantIndexOp>(loc, 34));
    Value lowByte = rewriter.create<memref::LoadOp>(
        loc, weight, ValueRange{column, scaleOffset});
    Value highByte = rewriter.create<memref::LoadOp>(
        loc, weight,
        ValueRange{column, rewriter.create<arith::AddIOp>(
                               loc, scaleOffset, one)});
    Value low = rewriter.create<arith::ExtUIOp>(loc, i16, lowByte);
    Value high = rewriter.create<arith::ExtUIOp>(loc, i16, highByte);
    Value shift =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(8));
    Value bits = rewriter.create<arith::OrIOp>(
        loc, low, rewriter.create<arith::ShLIOp>(loc, high, shift));
    Value half =
        rewriter.create<arith::BitcastOp>(loc, rewriter.getF16Type(), bits);
    Value weightScale = rewriter.create<arith::ExtFOp>(loc, f32, half);

    Value qBase = qblock;
    Value wBase = rewriter.create<arith::AddIOp>(
        loc, scaleOffset, rewriter.create<arith::ConstantIndexOp>(loc, 2));
    Value aVec0 = rewriter.create<vector::TransferReadOp>(
        loc, i8x16, quantized, ValueRange{row, qBase}, rowMap, inBounds);
    Value aVec1 = rewriter.create<vector::TransferReadOp>(
        loc, i8x16, quantized,
        ValueRange{row, rewriter.create<arith::AddIOp>(
                            loc, qBase,
                            rewriter.create<arith::ConstantIndexOp>(loc, 16))},
        rowMap, inBounds);
    Value wVec0 = rewriter.create<vector::TransferReadOp>(
        loc, i8x16, weight, ValueRange{column, wBase}, rowMap, inBounds);
    Value wVec1 = rewriter.create<vector::TransferReadOp>(
        loc, i8x16, weight,
        ValueRange{column, rewriter.create<arith::AddIOp>(
                               loc, wBase,
                               rewriter.create<arith::ConstantIndexOp>(loc, 16))},
        rowMap, inBounds);
    auto sdot0 = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, i32x4,
        StringAttr::get(rewriter.getContext(), "llvm.aarch64.neon.sdot"),
        ValueRange{zeroI32, aVec0, wVec0});
    auto sdot1 = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, i32x4,
        StringAttr::get(rewriter.getContext(), "llvm.aarch64.neon.sdot"),
        ValueRange{sdot0.getResult(0), aVec1, wVec1});
    Value dot = rewriter.create<vector::ReductionOp>(
        loc, vector::CombiningKind::ADD, sdot1.getResult(0));
    Value activationScale = rewriter.create<memref::LoadOp>(
        loc, scales, ValueRange{row, qblockIndex});
    Value dotF = rewriter.create<arith::SIToFPOp>(loc, f32, dot);
    return rewriter.create<arith::AddFOp>(
        loc, accumulator,
        rewriter.create<arith::MulFOp>(
            loc, dotF,
            rewriter.create<arith::MulFOp>(loc, activationScale,
                                           weightScale)));
  };
  auto store = [&](Value value, Value row, Value column) {
    value = applyActivation(rewriter, loc, value, activation);
    if (resultType.getElementType().isBF16())
      value =
          rewriter.create<arith::TruncFOp>(loc, rewriter.getBF16Type(), value);
    rewriter.create<memref::StoreOp>(loc, value, output,
                                     ValueRange{row, column});
  };

  auto columnBlockLoop = rewriter.create<scf::ForOp>(
      loc, zero, channelsValue, columnBlock, ValueRange{});
  columnBlockLoop->setAttr("llmcp.parallel",
                           UnitAttr::get(rewriter.getContext()));
  rewriter.setInsertionPointToStart(columnBlockLoop.getBody());
  Value columnStart = columnBlockLoop.getInductionVar();
  Value columnEnd = rewriter.create<arith::MinUIOp>(
      loc, rewriter.create<arith::AddIOp>(loc, columnStart, columnBlock),
      channelsValue);

  auto rowGroupLoop = rewriter.create<scf::ForOp>(
      loc, zero, mainRows, rowTileValue, ValueRange{});
  rewriter.setInsertionPointToStart(rowGroupLoop.getBody());
  Value rowStart = rowGroupLoop.getInductionVar();
  auto columnTileLoop = rewriter.create<scf::ForOp>(
      loc, columnStart, columnEnd, columnTileValue, ValueRange{});
  rewriter.setInsertionPointToStart(columnTileLoop.getBody());
  Value columnStartTile = columnTileLoop.getInductionVar();

  auto reductionBlockLoop = rewriter.create<scf::ForOp>(
      loc, zero, reductionValue, reductionBlock,
      ValueRange{SmallVector<Value, 16>(16, zeroVecF)});
  rewriter.setInsertionPointToStart(reductionBlockLoop.getBody());
  Value reductionStart = reductionBlockLoop.getInductionVar();
  Value reductionEnd = rewriter.create<arith::MinUIOp>(
      loc, rewriter.create<arith::AddIOp>(loc, reductionStart, reductionBlock),
      reductionValue);
  auto qblockLoop = rewriter.create<scf::ForOp>(
      loc, reductionStart, reductionEnd, blockSize,
      ValueRange{reductionBlockLoop.getRegionIterArgs()});
  rewriter.setInsertionPointToStart(qblockLoop.getBody());
  Value qblock = qblockLoop.getInductionVar();
  SmallVector<Value, 16> next;
  for (int64_t row = 0; row < kRowTile; ++row) {
    Value rowIndex = rewriter.create<arith::AddIOp>(
        loc, rowStart, rewriter.create<arith::ConstantIndexOp>(loc, row));
    for (int64_t column = 0; column < kColumnTile; ++column) {
      Value columnIndex = rewriter.create<arith::AddIOp>(
          loc, columnStartTile,
          rewriter.create<arith::ConstantIndexOp>(loc, column));
      next.push_back(emitDot(qblockLoop.getRegionIterArg(row * kColumnTile +
                                                         column),
                             rowIndex, columnIndex, qblock));
    }
  }
  rewriter.create<scf::YieldOp>(loc, ValueRange(next));
  rewriter.setInsertionPointAfter(qblockLoop);
  rewriter.create<scf::YieldOp>(loc, ValueRange{qblockLoop.getResults()});
  rewriter.setInsertionPointAfter(reductionBlockLoop);
  for (int64_t row = 0; row < kRowTile; ++row) {
    Value rowIndex = rewriter.create<arith::AddIOp>(
        loc, rowStart, rewriter.create<arith::ConstantIndexOp>(loc, row));
    for (int64_t column = 0; column < kColumnTile; ++column) {
      Value columnIndex = rewriter.create<arith::AddIOp>(
          loc, columnStartTile,
          rewriter.create<arith::ConstantIndexOp>(loc, column));
      store(reductionBlockLoop.getResult(row * kColumnTile + column), rowIndex,
            columnIndex);
    }
  }
  rewriter.setInsertionPointAfter(columnTileLoop);
  rewriter.setInsertionPointAfter(rowGroupLoop);

  auto tailRowLoop = rewriter.create<scf::ForOp>(
      loc, mainRows, rows, one, ValueRange{});
  rewriter.setInsertionPointToStart(tailRowLoop.getBody());
  Value tailRow = tailRowLoop.getInductionVar();
  auto tailColumnTileLoop = rewriter.create<scf::ForOp>(
      loc, columnStart, columnEnd, columnTileValue, ValueRange{});
  rewriter.setInsertionPointToStart(tailColumnTileLoop.getBody());
  Value tailColumnStart = tailColumnTileLoop.getInductionVar();
  auto tailReductionBlockLoop = rewriter.create<scf::ForOp>(
      loc, zero, reductionValue, reductionBlock,
      ValueRange{SmallVector<Value, 4>(4, zeroVecF)});
  rewriter.setInsertionPointToStart(tailReductionBlockLoop.getBody());
  Value tailReductionStart = tailReductionBlockLoop.getInductionVar();
  Value tailReductionEnd = rewriter.create<arith::MinUIOp>(
      loc,
      rewriter.create<arith::AddIOp>(loc, tailReductionStart, reductionBlock),
      reductionValue);
  auto tailQblockLoop = rewriter.create<scf::ForOp>(
      loc, tailReductionStart, tailReductionEnd, blockSize,
      ValueRange{tailReductionBlockLoop.getRegionIterArgs()});
  rewriter.setInsertionPointToStart(tailQblockLoop.getBody());
  Value tailQblock = tailQblockLoop.getInductionVar();
  SmallVector<Value, 4> tailNext;
  for (int64_t column = 0; column < kColumnTile; ++column) {
    Value columnIndex = rewriter.create<arith::AddIOp>(
        loc, tailColumnStart,
        rewriter.create<arith::ConstantIndexOp>(loc, column));
    tailNext.push_back(emitDot(tailQblockLoop.getRegionIterArg(column), tailRow,
                               columnIndex, tailQblock));
  }
  rewriter.create<scf::YieldOp>(loc, ValueRange(tailNext));
  rewriter.setInsertionPointAfter(tailQblockLoop);
  rewriter.create<scf::YieldOp>(loc, ValueRange{tailQblockLoop.getResults()});
  rewriter.setInsertionPointAfter(tailReductionBlockLoop);
  for (int64_t column = 0; column < kColumnTile; ++column) {
    Value columnIndex = rewriter.create<arith::AddIOp>(
        loc, tailColumnStart,
        rewriter.create<arith::ConstantIndexOp>(loc, column));
    store(tailReductionBlockLoop.getResult(column), tailRow, columnIndex);
  }
  rewriter.setInsertionPointAfter(tailColumnTileLoop);
  rewriter.setInsertionPointAfter(tailRowLoop);
  rewriter.setInsertionPointAfter(columnBlockLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    /*restrict=*/true);
}

Value lowerQ8MmaPrefill(PatternRewriter &rewriter, Location loc, Value input,
                         Value weight, RankedTensorType resultType,
                         StringRef activation) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<MemRefType>(weight.getType());
  const int64_t reduction = inputType.getDimSize(1);
  const int64_t channels = resultType.getDimSize(1);
  // 4x4 output tile: four SMMLA 2x2 accumulators plus f32 accumulators.
  constexpr int64_t kATile = 4;
  constexpr int64_t kBTile = 4;
  // Cache blocks: 16x128 i8 A/B tiles fit comfortably in the 128 KiB L1D.
  constexpr int64_t kABlock = 16;
  constexpr int64_t kBBlock = 16;
  constexpr int64_t kKBlock = 128;
  if (inputType.getRank() != 2 || weightType.getRank() != 2 ||
      ShapedType::isDynamic(reduction) || ShapedType::isDynamic(channels) ||
      reduction % 32 != 0 || channels % kBTile != 0 ||
      !inputType.getElementType().isBF16() ||
      !weightType.getElementType().isInteger(8))
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
  Value reductionValue =
      rewriter.create<arith::ConstantIndexOp>(loc, reduction);
  Value rows = rewriter.create<tensor::DimOp>(loc, input, 0);
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output =
      rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{rows});
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

  auto i8x8 = VectorType::get({8}, i8);
  auto i8x16 = VectorType::get({16}, i8);
  auto i32x4 = VectorType::get({4}, i32);
  auto f32x4 = VectorType::get({4}, f32);
  SmallVector<bool> inBounds = {true};
  AffineMap rowMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                                    rewriter.getContext());
  AffineMap identity2 = rewriter.getMultiDimIdentityMap(2);
  SmallVector<int64_t, 16> concatMask;
  for (int64_t i = 0; i < 16; ++i)
    concatMask.push_back(i);
  Value zeroI32 = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(i32x4, rewriter.getI32IntegerAttr(0)));
  Value zeroF32 = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(f32x4, rewriter.getF32FloatAttr(0.0f)));

  auto load8 = [&](Value source, Value row, Value column) {
    return rewriter.create<vector::TransferReadOp>(
        loc, i8x8, source, ValueRange{row, column}, rowMap, inBounds);
  };
  auto concat = [&](Value low, Value high) {
    return rewriter.create<vector::ShuffleOp>(loc, low, high, concatMask);
  };
  auto extract = [&](Value vector, int64_t lane) {
    return rewriter.create<vector::ExtractElementOp>(
        loc, vector, rewriter.create<arith::ConstantIndexOp>(loc, lane));
  };
  auto weightScale = [&](Value column, Value qblockIndex) -> Value {
    Value scaleOffset = rewriter.create<arith::MulIOp>(
        loc, qblockIndex, rewriter.create<arith::ConstantIndexOp>(loc, 34));
    Value lowByte = rewriter.create<memref::LoadOp>(
        loc, weight, ValueRange{column, scaleOffset});
    Value highByte = rewriter.create<memref::LoadOp>(
        loc, weight,
        ValueRange{column, rewriter.create<arith::AddIOp>(
                               loc, scaleOffset, one)});
    Value low = rewriter.create<arith::ExtUIOp>(loc, i16, lowByte);
    Value high = rewriter.create<arith::ExtUIOp>(loc, i16, highByte);
    Value shift =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(8));
    Value bits = rewriter.create<arith::OrIOp>(
        loc, low, rewriter.create<arith::ShLIOp>(loc, high, shift));
    Value half =
        rewriter.create<arith::BitcastOp>(loc, rewriter.getF16Type(), bits);
    return rewriter.create<arith::ExtFOp>(loc, f32, half);
  };

  auto bBlockValue = rewriter.create<arith::ConstantIndexOp>(loc, kBBlock);
  auto aBlockValue = rewriter.create<arith::ConstantIndexOp>(loc, kABlock);
  auto bTileValue = rewriter.create<arith::ConstantIndexOp>(loc, kBTile);
  auto aTileValue = rewriter.create<arith::ConstantIndexOp>(loc, kATile);
  auto kBlockValue = rewriter.create<arith::ConstantIndexOp>(loc, kKBlock);
  Value mainRows = rewriter.create<arith::SubIOp>(
      loc, rows,
      rewriter.create<arith::RemUIOp>(
          loc, rows, rewriter.create<arith::ConstantIndexOp>(loc, kATile)));

  auto bBlockLoop = rewriter.create<scf::ForOp>(
      loc, zero, channelsValue, bBlockValue, ValueRange{});
  bBlockLoop->setAttr("llmcp.parallel",
                      UnitAttr::get(rewriter.getContext()));
  rewriter.setInsertionPointToStart(bBlockLoop.getBody());
  Value bBlockStart = bBlockLoop.getInductionVar();
  Value bBlockEnd = rewriter.create<arith::MinUIOp>(
      loc, rewriter.create<arith::AddIOp>(loc, bBlockStart, bBlockValue),
      channelsValue);
  auto aBlockLoop = rewriter.create<scf::ForOp>(
      loc, zero, mainRows, aBlockValue, ValueRange{});
  rewriter.setInsertionPointToStart(aBlockLoop.getBody());
  Value aBlockStart = aBlockLoop.getInductionVar();
  Value aBlockEnd = rewriter.create<arith::MinUIOp>(
      loc, rewriter.create<arith::AddIOp>(loc, aBlockStart, aBlockValue),
      mainRows);
  auto bTileLoop = rewriter.create<scf::ForOp>(
      loc, bBlockStart, bBlockEnd, bTileValue, ValueRange{});
  rewriter.setInsertionPointToStart(bTileLoop.getBody());
  Value bTileStart = bTileLoop.getInductionVar();
  Value bTileEnd = rewriter.create<arith::MinUIOp>(
      loc, rewriter.create<arith::AddIOp>(loc, bTileStart, bTileValue),
      bBlockEnd);
  auto aTileLoop = rewriter.create<scf::ForOp>(
      loc, aBlockStart, aBlockEnd, aTileValue, ValueRange{});
  rewriter.setInsertionPointToStart(aTileLoop.getBody());
  Value aTileStart = aTileLoop.getInductionVar();
  Value aTileEnd = rewriter.create<arith::MinUIOp>(
      loc, rewriter.create<arith::AddIOp>(loc, aTileStart, aTileValue),
      aBlockEnd);
  (void)aTileEnd;

  auto kBlockLoop = rewriter.create<scf::ForOp>(
      loc, zero, reductionValue, kBlockValue,
      ValueRange{zeroF32, zeroF32, zeroF32, zeroF32});
  rewriter.setInsertionPointToStart(kBlockLoop.getBody());
  Value kBlockStart = kBlockLoop.getInductionVar();
  Value kBlockEnd = rewriter.create<arith::MinUIOp>(
      loc, rewriter.create<arith::AddIOp>(loc, kBlockStart, kBlockValue),
      reductionValue);
  auto qblockLoop = rewriter.create<scf::ForOp>(
      loc, kBlockStart, kBlockEnd, blockSize,
      ValueRange{kBlockLoop.getRegionIterArgs()});
  rewriter.setInsertionPointToStart(qblockLoop.getBody());
  Value qblock = qblockLoop.getInductionVar();
  Value qblockEnd = rewriter.create<arith::MinUIOp>(
      loc, rewriter.create<arith::AddIOp>(loc, qblock, blockSize), kBlockEnd);

  Value qblockIndex =
      rewriter.create<arith::DivUIOp>(loc, qblock, blockSize);
  SmallVector<Value, 4> intAcc(4, zeroI32);
  auto k8Loop = rewriter.create<scf::ForOp>(
      loc, qblock, qblockEnd, rewriter.create<arith::ConstantIndexOp>(loc, 8),
      ValueRange{intAcc});
  rewriter.setInsertionPointToStart(k8Loop.getBody());
  Value k8 = k8Loop.getInductionVar();
  Value weightBase = rewriter.create<arith::AddIOp>(
      loc, rewriter.create<arith::MulIOp>(
               loc, qblockIndex, rewriter.create<arith::ConstantIndexOp>(loc, 34)),
      rewriter.create<arith::ConstantIndexOp>(loc, 2));
  Value k8Offset =
      rewriter.create<arith::SubIOp>(loc, k8, qblock);
  Value weightK = rewriter.create<arith::AddIOp>(loc, weightBase, k8Offset);
  SmallVector<Value, 4> nextInt;
  for (int64_t rp = 0; rp < 2; ++rp) {
    Value aRow0 = rewriter.create<arith::AddIOp>(
        loc, aTileStart, rewriter.create<arith::ConstantIndexOp>(loc, 2 * rp));
    Value aRow1 = rewriter.create<arith::AddIOp>(
        loc, aRow0, one);
    Value aVec = concat(load8(quantized, aRow0, k8),
                        load8(quantized, aRow1, k8));
    for (int64_t cp = 0; cp < 2; ++cp) {
      Value bCol0 = rewriter.create<arith::AddIOp>(
          loc, bTileStart,
          rewriter.create<arith::ConstantIndexOp>(loc, 2 * cp));
      Value bCol1 = rewriter.create<arith::AddIOp>(loc, bCol0, one);
      Value bVec = concat(load8(weight, bCol0, weightK),
                          load8(weight, bCol1, weightK));
      Value acc = k8Loop.getRegionIterArg(rp * 2 + cp);
      auto mma = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, i32x4,
          StringAttr::get(rewriter.getContext(), "llvm.aarch64.neon.smmla"),
          ValueRange{acc, aVec, bVec});
      nextInt.push_back(mma.getResult(0));
    }
  }
  rewriter.create<scf::YieldOp>(loc, ValueRange(nextInt));
  rewriter.setInsertionPointAfter(k8Loop);

  SmallVector<Value, 4> nextF32;
  for (int64_t rp = 0; rp < 2; ++rp) {
    Value aRow0 = rewriter.create<arith::AddIOp>(
        loc, aTileStart, rewriter.create<arith::ConstantIndexOp>(loc, 2 * rp));
    Value aRow1 = rewriter.create<arith::AddIOp>(loc, aRow0, one);
    Value aScale0 = rewriter.create<memref::LoadOp>(
        loc, scales, ValueRange{aRow0, qblockIndex});
    Value aScale1 = rewriter.create<memref::LoadOp>(
        loc, scales, ValueRange{aRow1, qblockIndex});
    for (int64_t cp = 0; cp < 2; ++cp) {
      Value bCol0 = rewriter.create<arith::AddIOp>(
          loc, bTileStart,
          rewriter.create<arith::ConstantIndexOp>(loc, 2 * cp));
      Value bCol1 = rewriter.create<arith::AddIOp>(loc, bCol0, one);
      Value w0 = weightScale(bCol0, qblockIndex);
      Value w1 = weightScale(bCol1, qblockIndex);
      Value s0 = rewriter.create<arith::MulFOp>(loc, aScale0, w0);
      Value s1 = rewriter.create<arith::MulFOp>(loc, aScale0, w1);
      Value s2 = rewriter.create<arith::MulFOp>(loc, aScale1, w0);
      Value s3 = rewriter.create<arith::MulFOp>(loc, aScale1, w1);
      Value scaleVec = rewriter.create<vector::FromElementsOp>(
          loc, f32x4, ValueRange{s0, s1, s2, s3});
      Value dotVec = rewriter.create<arith::SIToFPOp>(
          loc, f32x4, k8Loop.getResult(rp * 2 + cp));
      Value scaled = rewriter.create<arith::MulFOp>(loc, dotVec, scaleVec);
      Value accum = rewriter.create<arith::AddFOp>(
          loc, qblockLoop.getRegionIterArg(rp * 2 + cp), scaled);
      nextF32.push_back(accum);
    }
  }
  rewriter.create<scf::YieldOp>(loc, ValueRange(nextF32));
  rewriter.setInsertionPointAfter(qblockLoop);
  rewriter.create<scf::YieldOp>(loc, ValueRange{qblockLoop.getResults()});
  rewriter.setInsertionPointAfter(kBlockLoop);

  auto storeValue = [&](Value value, Value row, Value column) {
    value = applyActivation(rewriter, loc, value, activation);
    if (resultType.getElementType().isBF16())
      value =
          rewriter.create<arith::TruncFOp>(loc, rewriter.getBF16Type(), value);
    rewriter.create<memref::StoreOp>(loc, value, output,
                                     ValueRange{row, column});
  };
  for (int64_t rp = 0; rp < 2; ++rp) {
    Value row0 = rewriter.create<arith::AddIOp>(
        loc, aTileStart, rewriter.create<arith::ConstantIndexOp>(loc, 2 * rp));
    Value row1 = rewriter.create<arith::AddIOp>(loc, row0, one);
    for (int64_t cp = 0; cp < 2; ++cp) {
      Value col0 = rewriter.create<arith::AddIOp>(
          loc, bTileStart,
          rewriter.create<arith::ConstantIndexOp>(loc, 2 * cp));
      Value col1 = rewriter.create<arith::AddIOp>(loc, col0, one);
      Value vec = kBlockLoop.getResult(rp * 2 + cp);
      storeValue(extract(vec, 0), row0, col0);
      storeValue(extract(vec, 1), row0, col1);
      storeValue(extract(vec, 2), row1, col0);
      storeValue(extract(vec, 3), row1, col1);
    }
  }
  rewriter.setInsertionPointAfter(aTileLoop);
  rewriter.setInsertionPointAfter(bTileLoop);
  rewriter.setInsertionPointAfter(aBlockLoop);
  rewriter.setInsertionPointAfter(bBlockLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    /*restrict=*/true);
}

Value lowerMatmulInternal(PatternRewriter &rewriter, Location loc, Value input,
                          Value weight, RankedTensorType resultType,
                          StringRef weightFormat, bool transposeB, bool decode,
                          StringRef activation) {
  if (weightFormat == "q8_0") {
    if (transposeB)
      return Value();
    if (!decode) {
      auto inputType = cast<RankedTensorType>(input.getType());
      const int64_t channels = resultType.getDimSize(1);
      if (!ShapedType::isDynamic(channels) && channels % 4 == 0 &&
          inputType.getElementType().isBF16()) {
        Value rows = rewriter.create<tensor::DimOp>(loc, input, 0);
        Value remainder = rewriter.create<arith::RemUIOp>(
            loc, rows, rewriter.create<arith::ConstantIndexOp>(loc, 4));
        Value aligned = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, remainder,
            rewriter.create<arith::ConstantIndexOp>(loc, 0));
        auto ifOp =
            rewriter.create<scf::IfOp>(loc, resultType, aligned, true);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          Value mma =
              lowerQ8MmaPrefill(rewriter, loc, input, weight, resultType,
                                activation);
          if (!mma)
            return Value();
          rewriter.create<scf::YieldOp>(loc, mma);
        }
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
          Value fallback =
              emitPrefillQ8Blocked(rewriter, loc, input, weight, resultType,
                                   activation);
          if (!fallback)
            return Value();
          rewriter.create<scf::YieldOp>(loc, fallback);
        }
        return ifOp.getResult(0);
      }
      if (Value blocked = emitPrefillQ8Blocked(rewriter, loc, input, weight,
                                                resultType, activation))
        return blocked;
    }
    return emitQ8Matmul(rewriter, loc, input, weight, resultType, decode,
                        activation);
  }

  Value tensorWeight = toTensor(rewriter, loc, weight);
  if (!transposeB) {
#if defined(__aarch64__)
    if (!decode && weightFormat != "q8_0" &&
        resultType.getElementType().isBF16()) {
      if (Value blocked =
              emitPrefillBfDotBlocked(rewriter, loc, input, tensorWeight,
                                      resultType, activation))
        return blocked;
    }
    if (Value value = emitBfDotMatmul(rewriter, loc, input, tensorWeight,
                                      resultType, decode, activation))
      return value;
#endif
    if (Value value = emitVectorDotMatmul(rewriter, loc, input, tensorWeight,
                                          resultType, decode, activation))
      return value;
  }
  return emitVectorMatmul(rewriter, loc, input, tensorWeight, resultType,
                          transposeB, decode, activation);
}

} // namespace llmcp
} // namespace mlir
