//===- LowerGatedDeltaNet.cpp - Lower llmcp.gated_delta_net ---------------===//
//
// Naive token-by-token recurrence. The same lowering handles prefill and
// decode; chunked prefill is a later optimization.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/Passes.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCP.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include <cmath>

using namespace mlir;
using namespace mlir::llmcp;

namespace {

scf::ForOp
createLoop(PatternRewriter &rewriter, Location loc, Value lower, Value upper,
           Value step, ValueRange iterArgs,
           function_ref<void(OpBuilder &, Location, Value, ValueRange)> body) {
  auto loop = rewriter.create<scf::ForOp>(loc, lower, upper, step, iterArgs);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    body(rewriter, loc, loop.getInductionVar(), loop.getRegionIterArgs());
  }
  return loop;
}

class GatedDeltaNetOpLowering : public OpRewritePattern<GatedDeltaNetOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GatedDeltaNetOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto convType = dyn_cast<RankedTensorType>(op.getConv().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    auto stateType = dyn_cast<MemRefType>(op.getState().getType());
    if (!convType || !resultType || !stateType)
      return failure();
    if (convType.getRank() != 2 || resultType.getRank() != 2 ||
        stateType.getRank() != 4)
      return failure();

    const int64_t numK = op.getNumKHeads();
    const int64_t numV = op.getNumVHeads();
    const int64_t headK = op.getHeadKDim();
    const int64_t headV = op.getHeadVDim();
    const int64_t ratio = numV / numK;
    if (numK <= 0 || numV <= 0 || headK <= 0 || headV <= 0 || numV % numK != 0)
      return failure();

    Type f32 = rewriter.getF32Type();
    Type bf16 = rewriter.getBF16Type();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value numKValue = rewriter.create<arith::ConstantIndexOp>(loc, numK);
    Value numVValue = rewriter.create<arith::ConstantIndexOp>(loc, numV);
    Value headKValue = rewriter.create<arith::ConstantIndexOp>(loc, headK);
    Value headVValue = rewriter.create<arith::ConstantIndexOp>(loc, headV);
    Value ratioValue = rewriter.create<arith::ConstantIndexOp>(loc, ratio);
    Value qDim = rewriter.create<arith::MulIOp>(loc, numKValue, headKValue);
    Value vDim = rewriter.create<arith::MulIOp>(loc, numVValue, headVValue);
    Value layerValue =
        rewriter.create<arith::ConstantIndexOp>(loc, op.getLayerIndex());
    Value sequence;
    if (convType.isDynamicDim(0))
      sequence = rewriter.create<tensor::DimOp>(loc, op.getConv(), 0);
    else
      sequence =
          rewriter.create<arith::ConstantIndexOp>(loc, convType.getDimSize(0));

    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i)
      if (resultType.isDynamicDim(i))
        dynamicSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, op.getConv(), i));
    auto outputType =
        MemRefType::get(resultType.getShape(), resultType.getElementType());
    Value output =
        rewriter.create<memref::AllocOp>(loc, outputType, dynamicSizes);

    Value zeroF =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value oneF =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
    Value eps = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(1.0e-6f));
    Value qScale = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(static_cast<float>(
                 1.0 / std::sqrt(static_cast<double>(headK)))));

    auto qnType = MemRefType::get({headK}, f32);
    auto knType = MemRefType::get({headV}, f32);
    auto deltaType = MemRefType::get({headV}, f32);
    Value qn = rewriter.create<memref::AllocOp>(loc, qnType);
    Value kn = rewriter.create<memref::AllocOp>(loc, knType);
    Value delta = rewriter.create<memref::AllocOp>(loc, deltaType);

    auto loadBf16 = [&](Value tensor, Value index0, Value index1) -> Value {
      Value value = rewriter.create<tensor::ExtractOp>(
          loc, tensor, ValueRange{index0, index1});
      return rewriter.create<arith::ExtFOp>(loc, f32, value);
    };

    auto timeLoop = createLoop(
        rewriter, loc, zero, sequence, one, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value time, ValueRange) {
          auto headLoop = createLoop(
              rewriter, loc, zero, numVValue, one, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value head, ValueRange) {
                Value kHead =
                    rewriter.create<arith::DivUIOp>(loc, head, ratioValue);
                Value qOffset =
                    rewriter.create<arith::MulIOp>(loc, kHead, headKValue);
                Value kOffsetBase =
                    rewriter.create<arith::MulIOp>(loc, qDim, one);
                Value kOffset = rewriter.create<arith::AddIOp>(
                    loc, kOffsetBase,
                    rewriter.create<arith::MulIOp>(loc, kHead, headKValue));
                Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);
                Value vOffset = rewriter.create<arith::AddIOp>(
                    loc, rewriter.create<arith::MulIOp>(loc, qDim, two),
                    rewriter.create<arith::MulIOp>(loc, head, headVValue));

                // q and k L2 norms, then scaled vectors.
                auto qSumLoop = createLoop(
                    rewriter, loc, zero, headKValue, one, ValueRange{zeroF},
                    [&](OpBuilder &builder, Location loc, Value index,
                        ValueRange args) {
                      Value offset =
                          rewriter.create<arith::AddIOp>(loc, qOffset, index);
                      Value value = loadBf16(op.getConv(), time, offset);
                      Value square =
                          rewriter.create<arith::MulFOp>(loc, value, value);
                      Value sum =
                          rewriter.create<arith::AddFOp>(loc, args[0], square);
                      rewriter.create<scf::YieldOp>(loc, sum);
                    });
                rewriter.setInsertionPointAfter(qSumLoop);
                Value qSum = qSumLoop.getResult(0);
                Value qSumEps = rewriter.create<arith::AddFOp>(loc, qSum, eps);
                Value qInv = rewriter.create<math::RsqrtOp>(loc, qSumEps);
                Value qNormScale =
                    rewriter.create<arith::MulFOp>(loc, qInv, qScale);

                auto kSumLoop = createLoop(
                    rewriter, loc, zero, headKValue, one, ValueRange{zeroF},
                    [&](OpBuilder &builder, Location loc, Value index,
                        ValueRange args) {
                      Value offset =
                          rewriter.create<arith::AddIOp>(loc, kOffset, index);
                      Value value = loadBf16(op.getConv(), time, offset);
                      Value square =
                          rewriter.create<arith::MulFOp>(loc, value, value);
                      Value sum =
                          rewriter.create<arith::AddFOp>(loc, args[0], square);
                      rewriter.create<scf::YieldOp>(loc, sum);
                    });
                rewriter.setInsertionPointAfter(kSumLoop);
                Value kSum = kSumLoop.getResult(0);
                Value kSumEps = rewriter.create<arith::AddFOp>(loc, kSum, eps);
                Value kInv = rewriter.create<math::RsqrtOp>(loc, kSumEps);

                auto normalizeLoop = createLoop(
                    rewriter, loc, zero, headKValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value index,
                        ValueRange) {
                      Value qValue = loadBf16(
                          op.getConv(), time,
                          rewriter.create<arith::AddIOp>(loc, qOffset, index));
                      Value kValue = loadBf16(
                          op.getConv(), time,
                          rewriter.create<arith::AddIOp>(loc, kOffset, index));
                      Value qNormalized = rewriter.create<arith::MulFOp>(
                          loc, qValue, qNormScale);
                      Value kNormalized =
                          rewriter.create<arith::MulFOp>(loc, kValue, kInv);
                      rewriter.create<memref::StoreOp>(loc, qNormalized, qn,
                                                       ValueRange{index});
                      rewriter.create<memref::StoreOp>(loc, kNormalized, kn,
                                                       ValueRange{index});
                    });
                rewriter.setInsertionPointAfter(normalizeLoop);

                // beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias).
                Value bValue = loadBf16(op.getB(), time, head);
                Value beta = rewriter.create<arith::DivFOp>(
                    loc, oneF,
                    rewriter.create<arith::AddFOp>(
                        loc, oneF,
                        rewriter.create<math::ExpOp>(
                            loc, rewriter.create<arith::NegFOp>(loc, bValue))));
                Value aValue = loadBf16(op.getA(), time, head);
                Value dtValue = rewriter.create<memref::LoadOp>(
                    loc, op.getDtBias(), ValueRange{head});
                Value dtF32 = rewriter.create<arith::ExtFOp>(loc, f32, dtValue);
                Value aPlusDt =
                    rewriter.create<arith::AddFOp>(loc, aValue, dtF32);
                Value softplus = rewriter.create<math::Log1pOp>(
                    loc, rewriter.create<math::ExpOp>(loc, aPlusDt));
                Value aLog = rewriter.create<memref::LoadOp>(loc, op.getALog(),
                                                             ValueRange{head});
                Value minusExpA = rewriter.create<arith::NegFOp>(
                    loc, rewriter.create<math::ExpOp>(loc, aLog));
                Value g =
                    rewriter.create<arith::MulFOp>(loc, minusExpA, softplus);
                Value decay = rewriter.create<math::ExpOp>(loc, g);

                // delta[v] = (v - S^T k) * beta, using the old state.
                auto deltaLoop = createLoop(
                    rewriter, loc, zero, headVValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value vIndex,
                        ValueRange) {
                      auto kvLoop = createLoop(
                          rewriter, loc, zero, headKValue, one,
                          ValueRange{zeroF},
                          [&](OpBuilder &builder, Location loc, Value kIndex,
                              ValueRange args) {
                            Value stateValue = rewriter.create<memref::LoadOp>(
                                loc, op.getState(),
                                ValueRange{layerValue, head, kIndex, vIndex});
                            Value decayedState = rewriter.create<arith::MulFOp>(
                                loc, stateValue, decay);
                            Value kValue = rewriter.create<memref::LoadOp>(
                                loc, kn, ValueRange{kIndex});
                            Value product = rewriter.create<arith::MulFOp>(
                                loc, decayedState, kValue);
                            Value sum = rewriter.create<arith::AddFOp>(
                                loc, args[0], product);
                            rewriter.create<scf::YieldOp>(loc, sum);
                          });
                      rewriter.setInsertionPointAfter(kvLoop);
                      Value vValue = loadBf16(
                          op.getConv(), time,
                          rewriter.create<arith::AddIOp>(loc, vOffset, vIndex));
                      Value difference = rewriter.create<arith::SubFOp>(
                          loc, vValue, kvLoop.getResult(0));
                      Value deltaValue =
                          rewriter.create<arith::MulFOp>(loc, difference, beta);
                      rewriter.create<memref::StoreOp>(loc, deltaValue, delta,
                                                       ValueRange{vIndex});
                    });
                rewriter.setInsertionPointAfter(deltaLoop);

                // S = S * decay + k^T delta.
                auto updateLoop = createLoop(
                    rewriter, loc, zero, headKValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value kIndex,
                        ValueRange) {
                      Value kValue = rewriter.create<memref::LoadOp>(
                          loc, kn, ValueRange{kIndex});
                      auto valueLoop = createLoop(
                          rewriter, loc, zero, headVValue, one, ValueRange{},
                          [&](OpBuilder &builder, Location loc, Value vIndex,
                              ValueRange) {
                            Value stateValue = rewriter.create<memref::LoadOp>(
                                loc, op.getState(),
                                ValueRange{layerValue, head, kIndex, vIndex});
                            Value decayed = rewriter.create<arith::MulFOp>(
                                loc, stateValue, decay);
                            Value deltaValue = rewriter.create<memref::LoadOp>(
                                loc, delta, ValueRange{vIndex});
                            Value update = rewriter.create<arith::MulFOp>(
                                loc, kValue, deltaValue);
                            Value updated = rewriter.create<arith::AddFOp>(
                                loc, decayed, update);
                            rewriter.create<memref::StoreOp>(
                                loc, updated, op.getState(),
                                ValueRange{layerValue, head, kIndex, vIndex});
                          });
                      rewriter.setInsertionPointAfter(valueLoop);
                    });
                rewriter.setInsertionPointAfter(updateLoop);

                // o[v] = S_new^T q.
                auto outputLoop = createLoop(
                    rewriter, loc, zero, headVValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value vIndex,
                        ValueRange) {
                      auto oLoop = createLoop(
                          rewriter, loc, zero, headKValue, one,
                          ValueRange{zeroF},
                          [&](OpBuilder &builder, Location loc, Value kIndex,
                              ValueRange args) {
                            Value stateValue = rewriter.create<memref::LoadOp>(
                                loc, op.getState(),
                                ValueRange{layerValue, head, kIndex, vIndex});
                            Value qValue = rewriter.create<memref::LoadOp>(
                                loc, qn, ValueRange{kIndex});
                            Value product = rewriter.create<arith::MulFOp>(
                                loc, stateValue, qValue);
                            Value sum = rewriter.create<arith::AddFOp>(
                                loc, args[0], product);
                            rewriter.create<scf::YieldOp>(loc, sum);
                          });
                      rewriter.setInsertionPointAfter(oLoop);
                      Value value = oLoop.getResult(0);
                      Value resultBf16 =
                          rewriter.create<arith::TruncFOp>(loc, bf16, value);
                      Value column = rewriter.create<arith::AddIOp>(
                          loc,
                          rewriter.create<arith::MulIOp>(loc, head, headVValue),
                          vIndex);
                      rewriter.create<memref::StoreOp>(
                          loc, resultBf16, output, ValueRange{time, column});
                    });
                rewriter.setInsertionPointAfter(outputLoop);
              });
          rewriter.setInsertionPointAfter(headLoop);
        });
    rewriter.setInsertionPointAfter(timeLoop);

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, output, /*restrict=*/true);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

} // namespace

void mlir::llmcp::populateGatedDeltaNetLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<GatedDeltaNetOpLowering>(patterns.getContext());
}
