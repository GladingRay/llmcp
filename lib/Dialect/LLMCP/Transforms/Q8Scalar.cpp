//===- Q8Scalar.cpp - Portable packed Q8 matmul ---------------------------===//
//
// Scalar fallback for hosts without Arm int8 dot-product support.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/MatmulLoweringInternal.h"
#include "llmcp/Support/Q8PackedLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::llmcp;

namespace {

class ScalarQ8 {
public:
  ScalarQ8(PatternRewriter &builder, Location loc)
      : builder(builder), loc(loc), f32(builder.getF32Type()),
        i8(builder.getI8Type()), i32(builder.getI32Type()), zero(index(0)),
        one(index(1)), zeroF(floating(0)), oneF(floating(1)) {}

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
                  ValueRange init = {}, bool parallel = false) {
    auto op = builder.create<scf::ForOp>(loc, begin, end, index(step), init);
    if (parallel)
      op->setAttr("llmcp.parallel", builder.getUnitAttr());
    builder.setInsertionPointToStart(op.getBody());
    return op;
  }

  PatternRewriter &builder;
  Location loc;
  Type f32;
  Type i8;
  Type i32;
  Value zero;
  Value one;
  Value zeroF;
  Value oneF;
};

Value ScalarQ8::emit(Value input, Value weight, RankedTensorType resultType,
                     StringRef activation) {
  auto inputType = cast<RankedTensorType>(input.getType());
  const int64_t reduction = inputType.getDimSize(1);
  const int64_t channels = resultType.getDimSize(1);
  const int64_t blocks = reduction / 32;

  Value rows = builder.create<tensor::DimOp>(loc, input, 0);
  constexpr int64_t dyn = ShapedType::kDynamic;
  auto outputType =
      MemRefType::get({dyn, channels}, resultType.getElementType());
  Value output = builder.create<memref::AllocOp>(
      loc, outputType, ValueRange{rows}, builder.getI64IntegerAttr(128));
  Value aData = builder.create<memref::AllocOp>(
      loc, MemRefType::get({dyn, reduction}, i8), ValueRange{rows});
  Value aScales = builder.create<memref::AllocOp>(
      loc, MemRefType::get({dyn, blocks}, f32), ValueRange{rows});

  // Quantize each activation row exactly like the SDOT kernel.
  auto rowQuant = loop(zero, rows, 1, {}, /*parallel=*/true);
  Value row = rowQuant.getInductionVar();
  auto blockQuant = loop(zero, index(blocks));
  Value block = blockQuant.getInductionVar();
  Value base =
      builder.create<arith::MulIOp>(loc, block, index(kQ8PackedBlockSize));
  auto maxLoop = loop(zero, index(kQ8PackedBlockSize), 1, ValueRange{zeroF});
  Value element = maxLoop.getInductionVar();
  Value source = builder.create<arith::AddIOp>(loc, base, element);
  Value value =
      builder.create<tensor::ExtractOp>(loc, input, ValueRange{row, source});
  if (value.getType().isBF16())
    value = builder.create<arith::ExtFOp>(loc, f32, value);
  Value magnitude = builder.create<math::AbsFOp>(loc, value);
  Value maximum = builder.create<arith::MaxNumFOp>(
      loc, maxLoop.getRegionIterArg(0), magnitude);
  builder.create<scf::YieldOp>(loc, maximum);
  builder.setInsertionPointAfter(maxLoop);
  Value maxAbs = maxLoop.getResult(0);
  Value scale = builder.create<arith::DivFOp>(loc, maxAbs, floating(127));
  builder.create<memref::StoreOp>(loc, scale, aScales, ValueRange{row, block});
  Value isZero = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                               maxAbs, zeroF);
  Value reciprocal = builder.create<arith::DivFOp>(loc, oneF, scale);
  Value inverse =
      builder.create<arith::SelectOp>(loc, isZero, zeroF, reciprocal);
  auto quantLoop = loop(zero, index(kQ8PackedBlockSize));
  element = quantLoop.getInductionVar();
  source = builder.create<arith::AddIOp>(loc, base, element);
  value =
      builder.create<tensor::ExtractOp>(loc, input, ValueRange{row, source});
  if (value.getType().isBF16())
    value = builder.create<arith::ExtFOp>(loc, f32, value);
  Value scaled = builder.create<arith::MulFOp>(loc, value, inverse);
  Value rounded = builder.create<math::RoundEvenOp>(loc, scaled);
  Value integer = builder.create<arith::FPToSIOp>(loc, i32, rounded);
  Value byte = builder.create<arith::TruncIOp>(loc, i8, integer);
  builder.create<memref::StoreOp>(loc, byte, aData, ValueRange{row, source});
  builder.setInsertionPointAfter(quantLoop);
  builder.setInsertionPointAfter(blockQuant);
  builder.setInsertionPointAfter(rowQuant);

  // One scalar dot per output element. Each packed lane stores four K bytes.
  auto rowOutput = loop(zero, rows, 1, {}, /*parallel=*/true);
  row = rowOutput.getInductionVar();
  auto columnLoop = loop(zero, index(channels));
  Value column = columnLoop.getInductionVar();
  Value group =
      builder.create<arith::DivUIOp>(loc, column, index(kQ8PackedLanes));
  Value lane =
      builder.create<arith::RemUIOp>(loc, column, index(kQ8PackedLanes));
  auto blockLoop = loop(zero, index(blocks), 1, ValueRange{zeroF});
  block = blockLoop.getInductionVar();
  Value groupBase = builder.create<arith::MulIOp>(
      loc, group, index(blocks * kQ8PackedBlockBytes));
  Value record = builder.create<arith::AddIOp>(
      loc, groupBase,
      builder.create<arith::MulIOp>(loc, block, index(kQ8PackedBlockBytes)));
  Value scaleBase = builder.create<arith::AddIOp>(
      loc, record, builder.create<arith::MulIOp>(loc, lane, index(2)));
  Value lowByte =
      builder.create<memref::LoadOp>(loc, weight, ValueRange{scaleBase});
  Value highByte = builder.create<memref::LoadOp>(
      loc, weight, ValueRange{add(scaleBase, 1)});
  Value low =
      builder.create<arith::ExtUIOp>(loc, builder.getI16Type(), lowByte);
  Value high =
      builder.create<arith::ExtUIOp>(loc, builder.getI16Type(), highByte);
  Value shift =
      builder.create<arith::ConstantOp>(loc, builder.getI16IntegerAttr(8));
  Value bits = builder.create<arith::OrIOp>(
      loc, low, builder.create<arith::ShLIOp>(loc, high, shift));
  Value half =
      builder.create<arith::BitcastOp>(loc, builder.getF16Type(), bits);
  Value weightScale = builder.create<arith::ExtFOp>(loc, f32, half);

  Value dotInit =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(0));
  auto dotLoop = loop(zero, index(kQ8PackedBlockSize), 1, ValueRange{dotInit});
  element = dotLoop.getInductionVar();
  Value activationIndex = builder.create<arith::AddIOp>(
      loc, builder.create<arith::MulIOp>(loc, block, index(kQ8PackedBlockSize)),
      element);
  Value activationValue = builder.create<memref::LoadOp>(
      loc, aData, ValueRange{row, activationIndex});
  Value activationInt =
      builder.create<arith::ExtSIOp>(loc, i32, activationValue);
  Value chunk = builder.create<arith::DivUIOp>(loc, element, index(4));
  Value byteInWord = builder.create<arith::RemUIOp>(loc, element, index(4));
  Value weightIndex =
      builder.create<arith::AddIOp>(loc, record, index(kQ8PackedScaleBytes));
  weightIndex = builder.create<arith::AddIOp>(
      loc, weightIndex,
      builder.create<arith::MulIOp>(
          loc, chunk,
          builder.create<arith::ConstantIndexOp>(loc, kQ8PackedChunkBytes)));
  weightIndex = builder.create<arith::AddIOp>(
      loc, weightIndex, builder.create<arith::MulIOp>(loc, lane, index(4)));
  weightIndex = builder.create<arith::AddIOp>(loc, weightIndex, byteInWord);
  Value quantized =
      builder.create<memref::LoadOp>(loc, weight, ValueRange{weightIndex});
  Value quantizedInt = builder.create<arith::ExtSIOp>(loc, i32, quantized);
  Value product =
      builder.create<arith::MulIOp>(loc, activationInt, quantizedInt);
  Value sum =
      builder.create<arith::AddIOp>(loc, dotLoop.getRegionIterArg(0), product);
  builder.create<scf::YieldOp>(loc, sum);
  builder.setInsertionPointAfter(dotLoop);
  Value activationScale =
      builder.create<memref::LoadOp>(loc, aScales, ValueRange{row, block});
  Value factor =
      builder.create<arith::MulFOp>(loc, activationScale, weightScale);
  Value dot = builder.create<arith::SIToFPOp>(loc, f32, dotLoop.getResult(0));
  Value contribution = builder.create<arith::MulFOp>(loc, dot, factor);
  Value accumulator = builder.create<arith::AddFOp>(
      loc, blockLoop.getRegionIterArg(0), contribution);
  builder.create<scf::YieldOp>(loc, accumulator);
  builder.setInsertionPointAfter(blockLoop);
  Value result = blockLoop.getResult(0);
  result = applyMatmulActivation(builder, loc, result, activation);
  if (resultType.getElementType().isBF16())
    result =
        builder.create<arith::TruncFOp>(loc, builder.getBF16Type(), result);
  builder.create<memref::StoreOp>(loc, result, output, ValueRange{row, column});
  builder.setInsertionPointAfter(columnLoop);
  builder.setInsertionPointAfter(rowOutput);

  return output;
}

} // namespace

Value mlir::llmcp::emitQ8PackedScalarMatmul(PatternRewriter &rewriter,
                                            Location loc, Value input,
                                            Value weight,
                                            RankedTensorType resultType,
                                            StringRef activation) {
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
      reduction % kQ8PackedBlockSize != 0 || ShapedType::isDynamic(channels) ||
      channels <= 0 || channels % kQ8PackedLanes != 0)
    return Value();
  return ScalarQ8(rewriter, loc).emit(input, weight, resultType, activation);
}
