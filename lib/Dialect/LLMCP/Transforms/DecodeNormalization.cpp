//===- DecodeNormalization.cpp - Decode normalization lowering ------------===//
//
// Keeps reductions and scaling in one serial loop nest for decode rows.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/DecodeNormalization.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCP.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::llmcp;

namespace {

class DecodeNormalization {
public:
  DecodeNormalization(PatternRewriter &builder, Location loc)
      : builder(builder), loc(loc), f32(builder.getF32Type()), zero(index(0)),
        one(index(1)), zeroF(floating(0)), oneF(floating(1)) {}

  Value emit(Value input, Value gate, Value weight, RankedTensorType resultType,
             int64_t groupSize, double epsilon, bool zeroCentered,
             bool gateActivated);

private:
  Value index(int64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc, value);
  }

  Value floating(float value) {
    return builder.create<arith::ConstantOp>(loc,
                                             builder.getF32FloatAttr(value));
  }

  Value extend(Value value) {
    if (value.getType().isBF16())
      return builder.create<arith::ExtFOp>(loc, f32, value);
    return value;
  }

  Value load(Value tensor, Value row, Value column) {
    return extend(builder.create<tensor::ExtractOp>(loc, tensor,
                                                    ValueRange{row, column}));
  }

  scf::ForOp loop(Value end, ValueRange init = {}) {
    auto op = builder.create<scf::ForOp>(loc, zero, end, one, init);
    builder.setInsertionPointToStart(op.getBody());
    return op;
  }

  PatternRewriter &builder;
  Location loc;
  Type f32;
  Value zero;
  Value one;
  Value zeroF;
  Value oneF;
};

Value DecodeNormalization::emit(Value input, Value gate, Value weight,
                                RankedTensorType resultType, int64_t groupSize,
                                double epsilon, bool zeroCentered,
                                bool gateActivated) {
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t hidden = inputType.getDimSize(1);
  Value rows = builder.create<tensor::DimOp>(loc, input, 0);
  auto outputType =
      MemRefType::get(resultType.getShape(), resultType.getElementType());
  SmallVector<Value> dynamicSizes;
  if (resultType.isDynamicDim(0))
    dynamicSizes.push_back(rows);
  if (resultType.isDynamicDim(1))
    dynamicSizes.push_back(index(hidden));
  Value output = builder.create<memref::AllocOp>(loc, outputType, dynamicSizes);
  Value groupWidth = index(groupSize);
  Value divisor = floating(static_cast<float>(groupSize));
  Value eps = floating(static_cast<float>(epsilon));

  auto rowLoop = loop(rows);
  Value row = rowLoop.getInductionVar();
  auto groupLoop = loop(index(hidden / groupSize));
  Value group = groupLoop.getInductionVar();
  Value base = builder.create<arith::MulIOp>(loc, group, groupWidth);

  // Preserve the scalar reduction order from the existing lowering.
  auto sumLoop = loop(groupWidth, ValueRange{zeroF});
  Value d = sumLoop.getInductionVar();
  Value column = builder.create<arith::AddIOp>(loc, base, d);
  Value x = load(input, row, column);
  Value square = builder.create<arith::MulFOp>(loc, x, x);
  Value sum =
      builder.create<arith::AddFOp>(loc, sumLoop.getRegionIterArg(0), square);
  builder.create<scf::YieldOp>(loc, sum);
  builder.setInsertionPointAfter(sumLoop);
  Value mean =
      builder.create<arith::DivFOp>(loc, sumLoop.getResult(0), divisor);
  Value variance = builder.create<arith::AddFOp>(loc, mean, eps);
  Value rstd = builder.create<math::RsqrtOp>(loc, variance);

  auto scaleLoop = loop(groupWidth);
  d = scaleLoop.getInductionVar();
  column = builder.create<arith::AddIOp>(loc, base, d);
  x = load(input, row, column);
  Value weightIndex = gate ? d : column;
  Value w = extend(
      builder.create<memref::LoadOp>(loc, weight, ValueRange{weightIndex}));
  Value value;
  if (gate) {
    Value g = load(gate, row, column);
    Value silu = g;
    if (!gateActivated) {
      Value negative = builder.create<arith::NegFOp>(loc, g);
      Value exponent = builder.create<math::ExpOp>(loc, negative);
      Value denominator = builder.create<arith::AddFOp>(loc, oneF, exponent);
      Value sigmoid = builder.create<arith::DivFOp>(loc, oneF, denominator);
      silu = builder.create<arith::MulFOp>(loc, g, sigmoid);
    }
    // Match the original gated normalization's multiplication order.
    value = builder.create<arith::MulFOp>(loc, x, silu);
    value = builder.create<arith::MulFOp>(loc, value, w);
    value = builder.create<arith::MulFOp>(loc, value, rstd);
  } else {
    if (zeroCentered)
      w = builder.create<arith::AddFOp>(loc, w, oneF);
    value = builder.create<arith::MulFOp>(loc, x, rstd);
    value = builder.create<arith::MulFOp>(loc, value, w);
  }
  if (resultType.getElementType().isBF16())
    value = builder.create<arith::TruncFOp>(loc, resultType.getElementType(),
                                            value);
  builder.create<memref::StoreOp>(loc, value, output, ValueRange{row, column});
  builder.setInsertionPointAfter(rowLoop);
  return builder.create<bufferization::ToTensorOp>(loc, resultType, output,
                                                   /*restrict=*/true);
}

bool isSupported(Value input, RankedTensorType resultType, int64_t groupSize) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() != 2 || resultType.getRank() != 2 ||
      (!inputType.getElementType().isBF16() &&
       !inputType.getElementType().isF32()) ||
      (!resultType.getElementType().isBF16() &&
       !resultType.getElementType().isF32()))
    return false;
  int64_t hidden = inputType.getDimSize(1);
  return !ShapedType::isDynamic(hidden) && hidden > 0 && groupSize > 0 &&
         hidden % groupSize == 0;
}

} // namespace

Value mlir::llmcp::emitDecodeRmsNorm(PatternRewriter &rewriter, RmsNormOp op) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!inputType || !resultType || inputType.getRank() != 2)
    return Value();
  int64_t hidden = inputType.getDimSize(1);
  if (!isSupported(op.getInput(), resultType, hidden))
    return Value();
  return DecodeNormalization(rewriter, op.getLoc())
      .emit(op.getInput(), {}, op.getWeight(), resultType, hidden,
            op.getEpsAttr().getValueAsDouble(), op.getZeroCentered(), false);
}

Value mlir::llmcp::emitDecodeGatedRmsNorm(PatternRewriter &rewriter,
                                          GatedRmsNormOp op) {
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultType || !isSupported(op.getInput(), resultType, op.getGroupSize()))
    return Value();
  return DecodeNormalization(rewriter, op.getLoc())
      .emit(op.getInput(), op.getGate(), op.getWeight(), resultType,
            op.getGroupSize(), op.getEpsAttr().getValueAsDouble(), false,
            op.getGateActivated());
}
