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
#include "llvm/ADT/STLExtras.h"
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
                              Value columns, Value step, bool decode,
                              Value columnStart = {}) {
  MatmulLoops loops;
  loops.columnLoop = rewriter.create<scf::ForOp>(
      loc, columnStart ? columnStart : zero, columns, step, ValueRange{});
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

} // namespace

Value applyMatmulActivation(PatternRewriter &rewriter, Location loc, Value value,
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

namespace {

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
  result = applyMatmulActivation(rewriter, loc, result, activation);
  if (resultType.getElementType().isBF16())
    result =
        rewriter.create<arith::TruncFOp>(loc, rewriter.getBF16Type(), result);
  rewriter.create<memref::StoreOp>(loc, result, output,
                                   ValueRange{row, column});
  rewriter.setInsertionPointAfter(loops.columnLoop);
  return rewriter.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                    /*restrict=*/true);
}

#if defined(__aarch64__) && defined(LLMCP_ENABLE_BFDOT)
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
  Value result = applyMatmulActivation(rewriter, loc, sum, activation);
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
    value = applyMatmulActivation(rewriter, loc, value, activation);
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

Value lowerMatmulInternal(PatternRewriter &rewriter, Location loc, Value input,
                          Value weight, RankedTensorType resultType,
                          StringRef weightFormat, bool transposeB, bool decode,
                          StringRef activation) {
  if (weightFormat == "q8_0_p4") {
    if (transposeB)
      return Value();
    Value output =
        decode ? emitQ8PackedDecodeMatmul(rewriter, loc, input, weight,
                                          resultType, activation)
               : emitQ8PackedMatmul(rewriter, loc, input, weight, resultType,
                                    activation);
    if (!output)
      return Value();
    return rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, output, /*restrict=*/true);
  }
  Value tensorWeight = toTensor(rewriter, loc, weight);
  if (!transposeB) {
#if defined(__aarch64__) && defined(LLMCP_ENABLE_BFDOT)
    if (!decode && resultType.getElementType().isBF16()) {
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
