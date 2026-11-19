//===- Q8Decode.cpp - Vectorized Q8 SDOT decode GEMV ----------------------===//
//
// Computes four output columns per SDOT from pre-packed weight lanes.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/MatmulLoweringInternal.h"
#include "llmcp/Support/Q8PackedLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::llmcp;

#if defined(__aarch64__) && !defined(__AARCH64EB__) &&                         \
    defined(LLMCP_ENABLE_SDOT)
namespace {

constexpr int64_t kQuantBlock = 32;
constexpr int64_t kParallelColumns = 64;
constexpr int64_t kParallelElements = 256 * 1024;

class Q8Decode {
public:
  Q8Decode(PatternRewriter &builder, Location loc)
      : builder(builder), loc(loc), f32(builder.getF32Type()),
        i8(builder.getI8Type()), i32(builder.getI32Type()),
        i8x16(VectorType::get({16}, i8)), i32x4(VectorType::get({4}, i32)),
        f32x4(VectorType::get({4}, f32)), zero(index(0)), one(index(1)),
        zeroF(floating(0)) {}

  Value emitPacked(Value input, Value weight, RankedTensorType resultType,
                   StringRef activation);

private:
  Value index(int64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc, value);
  }

  Value floating(float value) {
    return builder.create<arith::ConstantOp>(loc,
                                             builder.getF32FloatAttr(value));
  }

  Value add(Value base, int64_t offset) {
    return builder.create<arith::AddIOp>(loc, base, index(offset));
  }

  scf::ForOp loop(Value begin, Value end, int64_t step = 1,
                  ValueRange init = {}) {
    auto op = builder.create<scf::ForOp>(loc, begin, end, index(step), init);
    builder.setInsertionPointToStart(op.getBody());
    return op;
  }

  Value splat(VectorType type, Value scalar) {
    return builder.create<vector::BroadcastOp>(loc, type, scalar);
  }

  void quantize(Value input, int64_t reduction, Value data, Value scales);
  void store(Value value, Value output, Value column, StringRef activation);

  PatternRewriter &builder;
  Location loc;
  Type f32;
  Type i8;
  Type i32;
  VectorType i8x16;
  VectorType i32x4;
  VectorType f32x4;
  Value zero;
  Value one;
  Value zeroF;
};

void Q8Decode::quantize(Value input, int64_t reduction, Value data,
                        Value scales) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputVector = VectorType::get({kQuantBlock}, inputType.getElementType());
  auto floatVector = VectorType::get({kQuantBlock}, f32);
  auto integerVector = VectorType::get({kQuantBlock}, i32);
  auto byteVector = VectorType::get({kQuantBlock}, i8);
  auto rowMap = AffineMap::getMinorIdentityMap(2, 1, builder.getContext());
  auto blockLoop = loop(zero, index(reduction / kQuantBlock));
  Value block = blockLoop.getInductionVar();
  Value base = builder.create<arith::MulIOp>(loc, block, index(kQuantBlock));
  Value values = builder.create<vector::TransferReadOp>(
      loc, inputVector, input, ValueRange{zero, base}, rowMap,
      ArrayRef<bool>{true});
  if (inputType.getElementType().isBF16())
    values = builder.create<arith::ExtFOp>(loc, floatVector, values);
  Value absolute = builder.create<math::AbsFOp>(loc, values);
  Value maximum = builder.create<vector::ReductionOp>(
      loc, vector::CombiningKind::MAXNUMF, absolute);
  Value scale = builder.create<arith::DivFOp>(loc, maximum, floating(127));
  builder.create<memref::StoreOp>(loc, scale, scales, ValueRange{block});

  // Preserve the existing reciprocal and rounding order for finite inputs.
  Value isZero = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                               maximum, zeroF);
  Value oneF = floating(1);
  Value denominator = builder.create<arith::SelectOp>(loc, isZero, oneF, scale);
  Value reciprocal = builder.create<arith::DivFOp>(loc, oneF, denominator);
  Value inverse =
      builder.create<arith::SelectOp>(loc, isZero, zeroF, reciprocal);
  Value scaled =
      builder.create<arith::MulFOp>(loc, values, splat(floatVector, inverse));
  Value rounded = builder.create<math::RoundEvenOp>(loc, scaled);
  Value integers = builder.create<arith::FPToSIOp>(loc, integerVector, rounded);
  Value bytes = builder.create<arith::TruncIOp>(loc, byteVector, integers);
  builder.create<vector::StoreOp>(loc, bytes, data, ValueRange{base});
  builder.setInsertionPointAfter(blockLoop);
}

void Q8Decode::store(Value value, Value output, Value column,
                     StringRef activation) {
  auto elementType = cast<MemRefType>(output.getType()).getElementType();
  value = applyMatmulActivation(builder, loc, value, activation);
  if (elementType.isBF16())
    value = builder.create<arith::TruncFOp>(loc, elementType, value);
  builder.create<memref::StoreOp>(loc, value, output, ValueRange{zero, column});
}

Value Q8Decode::emitPacked(Value input, Value weight,
                           RankedTensorType resultType, StringRef activation) {
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t reduction = inputType.getDimSize(1);
  int64_t channels = resultType.getDimSize(1);
  int64_t blockCount = reduction / kQuantBlock;
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output = builder.create<memref::AllocOp>(
      loc, outputType, ValueRange{one}, builder.getI64IntegerAttr(128));
  Value data = builder.create<memref::AllocOp>(
      loc, MemRefType::get({reduction}, i8), ValueRange{},
      builder.getI64IntegerAttr(64));
  Value scales = builder.create<memref::AllocOp>(
      loc, MemRefType::get({reduction / kQuantBlock}, f32), ValueRange{},
      builder.getI64IntegerAttr(64));
  quantize(input, reduction, data, scales);

  auto byteVector = VectorType::get({2 * kQ8PackedLanes}, i8);
  auto halfVector = VectorType::get({kQ8PackedLanes}, builder.getF16Type());
  Value groups = index(channels / kQ8PackedLanes);
  auto groupLoop = loop(zero, groups, 1, ValueRange{});
  if (channels >= kParallelColumns &&
      channels * blockCount >= kParallelElements / kQuantBlock)
    groupLoop->setAttr("llmcp.parallel", builder.getUnitAttr());
  Value group = groupLoop.getInductionVar();
  auto blockLoop =
      loop(zero, index(blockCount), 1, ValueRange{splat(f32x4, zeroF)});
  Value block = blockLoop.getInductionVar();
  Value recordBase = builder.create<arith::MulIOp>(
      loc, group, index(blockCount * kQ8PackedBlockBytes));
  Value record = builder.create<arith::AddIOp>(
      loc, recordBase,
      builder.create<arith::MulIOp>(loc, block, index(kQ8PackedBlockBytes)));

  // One SDOT produces one four-column output tile: lane r accumulates
  // column 4*group+r over four K elements.
  Value dot =
      builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(i32x4));
  Value blockBase =
      builder.create<arith::MulIOp>(loc, block, index(kQuantBlock));
  for (int64_t half = 0; half < kQuantBlock / 16; ++half) {
    Value a = builder.create<vector::LoadOp>(
        loc, i8x16, data, ValueRange{add(blockBase, half * 16)});
    Value words = builder.create<vector::BitCastOp>(loc, i32x4, a);
    for (int64_t word = 0; word < 4; ++word) {
      SmallVector<int64_t, 4> mask(4, word);
      Value broadcast =
          builder.create<vector::ShuffleOp>(loc, words, words, mask);
      broadcast = builder.create<vector::BitCastOp>(loc, i8x16, broadcast);
      int64_t chunkOffset =
          kQ8PackedScaleBytes + (half * 4 + word) * kQ8PackedChunkBytes;
      Value chunk = builder.create<vector::LoadOp>(
          loc, i8x16, weight, ValueRange{add(record, chunkOffset)});
      dot = builder
                .create<LLVM::CallIntrinsicOp>(
                    loc, i32x4, builder.getStringAttr("llvm.aarch64.neon.sdot"),
                    ValueRange{dot, chunk, broadcast})
                .getResult(0);
    }
  }
  Value scaleBytes = builder.create<vector::LoadOp>(loc, byteVector, weight,
                                                    ValueRange{record});
  Value weightScales =
      builder.create<vector::BitCastOp>(loc, halfVector, scaleBytes);
  weightScales = builder.create<arith::ExtFOp>(loc, f32x4, weightScales);
  Value activationScale =
      builder.create<memref::LoadOp>(loc, scales, ValueRange{block});
  Value factor = builder.create<arith::MulFOp>(
      loc, splat(f32x4, activationScale), weightScales);
  Value dotF = builder.create<arith::SIToFPOp>(loc, f32x4, dot);
  Value accumulator = builder.create<math::FmaOp>(
      loc, dotF, factor, blockLoop.getRegionIterArg(0));
  builder.create<scf::YieldOp>(loc, accumulator);
  builder.setInsertionPointAfter(blockLoop);

  Value result = blockLoop.getResult(0);
  Value column =
      builder.create<arith::MulIOp>(loc, group, index(kQ8PackedLanes));
  auto elementType = cast<MemRefType>(output.getType()).getElementType();
  if (activation == "none") {
    if (elementType.isBF16())
      result = builder.create<arith::TruncFOp>(
          loc, VectorType::get({kQ8PackedLanes}, elementType), result);
    builder.create<vector::StoreOp>(loc, result, output,
                                    ValueRange{zero, column});
  } else {
    for (int64_t lane = 0; lane < kQ8PackedLanes; ++lane) {
      Value value = builder.create<vector::ExtractOp>(loc, result, lane);
      store(value, output, add(column, lane), activation);
    }
  }
  builder.setInsertionPointAfter(groupLoop);
  return output;
}

} // namespace
#endif

Value mlir::llmcp::emitQ8PackedDecodeMatmul(PatternRewriter &rewriter,
                                            Location loc, Value input,
                                            Value weight,
                                            RankedTensorType resultType,
                                            StringRef activation) {
#if defined(__aarch64__) && !defined(__AARCH64EB__) &&                         \
    defined(LLMCP_ENABLE_SDOT)
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto weightType = dyn_cast<MemRefType>(weight.getType());
  if (!inputType || !weightType || inputType.getRank() != 2 ||
      weightType.getRank() != 1 || resultType.getRank() != 2 ||
      !weightType.getElementType().isInteger(8) ||
      (!inputType.getElementType().isBF16() &&
       !inputType.getElementType().isF32()) ||
      (!resultType.getElementType().isBF16() &&
       !resultType.getElementType().isF32()))
    return Value();
  int64_t reduction = inputType.getDimSize(1);
  int64_t channels = resultType.getDimSize(1);
  if (ShapedType::isDynamic(reduction) || reduction <= 0 ||
      reduction % kQuantBlock != 0 || ShapedType::isDynamic(channels) ||
      channels <= 0 || channels % kQ8PackedLanes != 0)
    return Value();
  return Q8Decode(rewriter, loc)
      .emitPacked(input, weight, resultType, activation);
#else
  return emitQ8PackedScalarMatmul(rewriter, loc, input, weight, resultType,
                                  activation);
#endif
}
