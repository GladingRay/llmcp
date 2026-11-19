//===- Q8Decode.cpp - Vectorized Q8 SDOT decode GEMV ----------------------===//
//
// Reduces four independent column dots into one vector for scaled accumulation.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/MatmulLoweringInternal.h"

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
constexpr int64_t kColumnsPerTile = 4;
constexpr int64_t kParallelColumns = 64;
constexpr int64_t kParallelElements = 256 * 1024;
constexpr int64_t kVectorLanes = 4;

static_assert(kColumnsPerTile % kVectorLanes == 0);

class Q8Decode {
public:
  Q8Decode(PatternRewriter &builder, Location loc)
      : builder(builder), loc(loc), f32(builder.getF32Type()),
        i8(builder.getI8Type()), i32(builder.getI32Type()),
        i8x16(VectorType::get({16}, i8)), i32x4(VectorType::get({4}, i32)),
        f32x4(VectorType::get({4}, f32)), zero(index(0)), one(index(1)),
        zeroF(floating(0)) {}

  Value emit(Value input, Value weight, RankedTensorType resultType,
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

  Value pairAdd(Value lhs, Value rhs) {
    return builder
        .create<LLVM::CallIntrinsicOp>(
            loc, i32x4, builder.getStringAttr("llvm.aarch64.neon.addp"),
            ValueRange{lhs, rhs})
        .getResult(0);
  }

  void quantize(Value input, int64_t reduction, Value data, Value scales);
  Value loadScale(Value column, Value block);
  Value emitDot(Value weight, Value column, Value block, Value a0, Value a1,
                int64_t blockCount);
  void emitColumns(Value weight, Value output, Value data, Value scales,
                   int64_t channels, int64_t blockCount, Value begin, Value end,
                   int64_t width, StringRef activation);
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
  Value weightAddress;
  Value weightOffset;
  Value weightStride;
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

Value Q8Decode::loadScale(Value column, Value block) {
  Value rowOffset = builder.create<arith::MulIOp>(loc, column, weightStride);
  Value scaleOffset = builder.create<arith::MulIOp>(loc, block, index(2));
  Value offset = builder.create<arith::AddIOp>(loc, weightOffset, rowOffset);
  offset = builder.create<arith::AddIOp>(loc, offset, scaleOffset);
  Value address = builder.create<arith::AddIOp>(loc, weightAddress, offset);
  Value integer =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), address);
  auto pointerType = LLVM::LLVMPointerType::get(builder.getContext());
  Value pointer = builder.create<LLVM::IntToPtrOp>(loc, pointerType, integer);
  return builder.create<LLVM::LoadOp>(loc, builder.getF16Type(), pointer, 1);
}

Value Q8Decode::emitDot(Value weight, Value column, Value block, Value a0,
                        Value a1, int64_t blockCount) {
  Value base = builder.create<arith::MulIOp>(loc, block, index(kQuantBlock));
  base = add(base, blockCount * 2);
  Value w0 = builder.create<vector::LoadOp>(loc, i8x16, weight,
                                            ValueRange{column, base});
  Value w1 = builder.create<vector::LoadOp>(loc, i8x16, weight,
                                            ValueRange{column, add(base, 16)});
  Value initial =
      builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(i32x4));
  Value dot =
      builder
          .create<LLVM::CallIntrinsicOp>(
              loc, i32x4, builder.getStringAttr("llvm.aarch64.neon.sdot"),
              ValueRange{initial, a0, w0})
          .getResult(0);
  return builder
      .create<LLVM::CallIntrinsicOp>(
          loc, i32x4, builder.getStringAttr("llvm.aarch64.neon.sdot"),
          ValueRange{dot, a1, w1})
      .getResult(0);
}

void Q8Decode::store(Value value, Value output, Value column,
                     StringRef activation) {
  auto elementType = cast<MemRefType>(output.getType()).getElementType();
  value = applyMatmulActivation(builder, loc, value, activation);
  if (elementType.isBF16())
    value = builder.create<arith::TruncFOp>(loc, elementType, value);
  builder.create<memref::StoreOp>(loc, value, output, ValueRange{zero, column});
}

void Q8Decode::emitColumns(Value weight, Value output, Value data, Value scales,
                           int64_t channels, int64_t blockCount, Value begin,
                           Value end, int64_t width, StringRef activation) {
  auto columnLoop = loop(begin, end, width);
  if (width >= kVectorLanes && channels >= kParallelColumns &&
      channels * blockCount >= kParallelElements / kQuantBlock)
    columnLoop->setAttr("llmcp.parallel", builder.getUnitAttr());
  Value firstColumn = columnLoop.getInductionVar();
  bool vectorized = width >= kVectorLanes;
  int64_t accumulators = vectorized ? width / kVectorLanes : 1;
  Value initial = vectorized ? splat(f32x4, zeroF) : zeroF;
  SmallVector<Value> init(accumulators, initial);
  auto blockLoop = loop(zero, index(blockCount), 1, init);
  Value block = blockLoop.getInductionVar();
  Value base = builder.create<arith::MulIOp>(loc, block, index(kQuantBlock));
  Value a0 = builder.create<vector::LoadOp>(loc, i8x16, data, ValueRange{base});
  Value a1 = builder.create<vector::LoadOp>(loc, i8x16, data,
                                            ValueRange{add(base, 16)});
  Value aScale = builder.create<memref::LoadOp>(loc, scales, ValueRange{block});
  SmallVector<Value> next;
  if (vectorized) {
    auto halfVector = VectorType::get({kVectorLanes}, builder.getF16Type());
    for (int64_t group = 0; group < accumulators; ++group) {
      SmallVector<Value, kVectorLanes> dots;
      SmallVector<Value, kVectorLanes> weightScales;
      for (int64_t lane = 0; lane < kVectorLanes; ++lane) {
        Value column = add(firstColumn, group * kVectorLanes + lane);
        dots.push_back(emitDot(weight, column, block, a0, a1, blockCount));
        weightScales.push_back(loadScale(column, block));
      }
      // Three pairwise integer adds place four exact K=32 dots in four lanes.
      Value dot = pairAdd(pairAdd(dots[0], dots[1]), pairAdd(dots[2], dots[3]));
      Value dotF = builder.create<arith::SIToFPOp>(loc, f32x4, dot);
      Value halfScales =
          builder.create<vector::FromElementsOp>(loc, halfVector, weightScales);
      Value wScales = builder.create<arith::ExtFOp>(loc, f32x4, halfScales);
      Value factor =
          builder.create<arith::MulFOp>(loc, splat(f32x4, aScale), wScales);
      Value contribution = builder.create<arith::MulFOp>(loc, dotF, factor);
      next.push_back(builder.create<arith::AddFOp>(
          loc, blockLoop.getRegionIterArg(group), contribution));
    }
  } else {
    Value dot = emitDot(weight, firstColumn, block, a0, a1, blockCount);
    dot = builder.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD,
                                              dot);
    Value dotF = builder.create<arith::SIToFPOp>(loc, f32, dot);
    Value weightScale = loadScale(firstColumn, block);
    weightScale = builder.create<arith::ExtFOp>(loc, f32, weightScale);
    Value factor = builder.create<arith::MulFOp>(loc, aScale, weightScale);
    Value contribution = builder.create<arith::MulFOp>(loc, dotF, factor);
    next.push_back(builder.create<arith::AddFOp>(
        loc, blockLoop.getRegionIterArg(0), contribution));
  }
  builder.create<scf::YieldOp>(loc, next);
  builder.setInsertionPointAfter(blockLoop);
  auto elementType = cast<MemRefType>(output.getType()).getElementType();
  for (int64_t group = 0; group < accumulators; ++group) {
    Value result = blockLoop.getResult(group);
    Value column = add(firstColumn, group * kVectorLanes);
    if (!vectorized) {
      store(result, output, column, activation);
    } else if (activation == "none") {
      if (elementType.isBF16())
        result = builder.create<arith::TruncFOp>(
            loc, VectorType::get({kVectorLanes}, elementType), result);
      builder.create<vector::StoreOp>(loc, result, output,
                                      ValueRange{zero, column});
    } else {
      for (int64_t lane = 0; lane < kVectorLanes; ++lane) {
        Value value = builder.create<vector::ExtractOp>(loc, result, lane);
        store(value, output, add(column, lane), activation);
      }
    }
  }
  builder.setInsertionPointAfter(columnLoop);
}

Value Q8Decode::emit(Value input, Value weight, RankedTensorType resultType,
                     StringRef activation) {
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t reduction = inputType.getDimSize(1);
  int64_t channels = resultType.getDimSize(1);
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
  // The raw half load uses byte offsets and permits unaligned Q8_0 views.
  auto metadata = builder.create<memref::ExtractStridedMetadataOp>(loc, weight);
  weightOffset = metadata.getOffset();
  weightStride = metadata.getStrides()[0];
  weightAddress =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, weight);
  int64_t tiledColumns = channels - channels % kColumnsPerTile;
  if (tiledColumns)
    emitColumns(weight, output, data, scales, channels, reduction / kQuantBlock,
                zero, index(tiledColumns), kColumnsPerTile, activation);
  if (tiledColumns != channels)
    emitColumns(weight, output, data, scales, channels, reduction / kQuantBlock,
                index(tiledColumns), index(channels), 1, activation);
  return output;
}

} // namespace
#endif

Value mlir::llmcp::emitQ8DecodeMatmul(PatternRewriter &rewriter, Location loc,
                                      Value input, Value weight,
                                      RankedTensorType resultType,
                                      StringRef activation) {
#if defined(__aarch64__) && !defined(__AARCH64EB__) &&                         \
    defined(LLMCP_ENABLE_SDOT)
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto weightType = dyn_cast<MemRefType>(weight.getType());
  if (!inputType || !weightType || inputType.getRank() != 2 ||
      weightType.getRank() != 2 || resultType.getRank() != 2 ||
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
      channels <= 0)
    return Value();
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(weightType.getStridesAndOffset(strides, offset)) ||
      strides[1] != 1)
    return Value();
  return Q8Decode(rewriter, loc).emit(input, weight, resultType, activation);
#else
  return Value();
#endif
}
