//===- LowerCausalConv1d.cpp - Lower llmcp.causal_conv1d ------------------===//
//
// Causal depthwise convolution lowering and state updates.
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

using namespace mlir;
using namespace mlir::llmcp;

namespace {

class CausalConv1dOpLowering : public OpRewritePattern<CausalConv1dOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CausalConv1dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto weightType = dyn_cast<MemRefType>(op.getWeight().getType());
    auto stateType = dyn_cast<MemRefType>(op.getState().getType());
    if (!inputType || !resultType || !weightType || !stateType)
      return failure();
    if (inputType.getRank() != 2 || resultType.getRank() != 2 ||
        weightType.getRank() != 3 || stateType.getRank() != 3)
      return failure();

    const int64_t channels = inputType.getDimSize(1);
    const int64_t kernel = op.getKernelSize();
    const int64_t stateLen = kernel - 1;
    if (ShapedType::isDynamic(channels) || kernel <= 1)
      return failure();

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value kernelValue = rewriter.create<arith::ConstantIndexOp>(loc, kernel);
    Value stateLenValue =
        rewriter.create<arith::ConstantIndexOp>(loc, stateLen);
    Value channelsValue =
        rewriter.create<arith::ConstantIndexOp>(loc, channels);
    Value layerValue =
        rewriter.create<arith::ConstantIndexOp>(loc, op.getLayerIndex());
    Value sequence;
    if (inputType.isDynamicDim(0))
      sequence = rewriter.create<tensor::DimOp>(loc, op.getInput(), 0);
    else
      sequence =
          rewriter.create<arith::ConstantIndexOp>(loc, inputType.getDimSize(0));

    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i)
      if (resultType.isDynamicDim(i))
        dynamicSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, op.getInput(), i));
    auto outputType =
        MemRefType::get(resultType.getShape(), resultType.getElementType());
    Value output =
        rewriter.create<memref::AllocOp>(loc, outputType, dynamicSizes);

    // out[t, c] = silu(sum_k w[c, 0, k] * padded[t + k]).
    auto timeLoop =
        rewriter.create<scf::ForOp>(loc, zero, sequence, one, ValueRange{});
    timeLoop->setAttr("llmcp.parallel", UnitAttr::get(rewriter.getContext()));
    rewriter.setInsertionPointToStart(timeLoop.getBody());
    auto channelLoop = rewriter.create<scf::ForOp>(loc, zero, channelsValue,
                                                   one, ValueRange{});
    rewriter.setInsertionPointToStart(channelLoop.getBody());

    Value time = timeLoop.getInductionVar();
    Value channel = channelLoop.getInductionVar();
    Value accInit =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    auto kernelLoop = rewriter.create<scf::ForOp>(loc, zero, kernelValue, one,
                                                  ValueRange{accInit});
    rewriter.setInsertionPointToStart(kernelLoop.getBody());

    Value k = kernelLoop.getInductionVar();
    Value acc = kernelLoop.getRegionIterArg(0);
    Value offset = rewriter.create<arith::SubIOp>(loc, k, stateLenValue);
    Value inputIndex = rewriter.create<arith::AddIOp>(loc, time, offset);
    Value isNegative = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, inputIndex, zero);

    auto select = rewriter.create<scf::IfOp>(
        loc, TypeRange{rewriter.getBF16Type()}, isNegative,
        /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&select.getThenRegion().front());
      Value stateIndex =
          rewriter.create<arith::AddIOp>(loc, inputIndex, stateLenValue);
      Value value = rewriter.create<memref::LoadOp>(
          loc, op.getState(), ValueRange{layerValue, stateIndex, channel});
      rewriter.create<scf::YieldOp>(loc, value);
    }
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&select.getElseRegion().front());
      Value value = rewriter.create<tensor::ExtractOp>(
          loc, op.getInput(), ValueRange{inputIndex, channel});
      rewriter.create<scf::YieldOp>(loc, value);
    }

    Value weight = rewriter.create<memref::LoadOp>(
        loc, op.getWeight(), ValueRange{channel, zero, k});
    Value value = select.getResult(0);
    Value valueF32 =
        rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), value);
    Value weightF32 =
        rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), weight);
    Value product = rewriter.create<arith::MulFOp>(loc, valueF32, weightF32);
    Value sum = rewriter.create<arith::AddFOp>(loc, acc, product);
    rewriter.create<scf::YieldOp>(loc, sum);

    rewriter.setInsertionPointAfter(kernelLoop);
    Value resultSum = kernelLoop.getResult(0);
    Value negated = rewriter.create<arith::NegFOp>(loc, resultSum);
    Value exponent = rewriter.create<math::ExpOp>(loc, negated);
    Value oneF32 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
    Value denominator = rewriter.create<arith::AddFOp>(loc, oneF32, exponent);
    Value sigmoid = rewriter.create<arith::DivFOp>(loc, oneF32, denominator);
    Value silu = rewriter.create<arith::MulFOp>(loc, resultSum, sigmoid);
    Value siluBf16 =
        rewriter.create<arith::TruncFOp>(loc, rewriter.getBF16Type(), silu);
    rewriter.create<memref::StoreOp>(loc, siluBf16, output,
                                     ValueRange{time, channel});

    // Keep the last `stateLen` input tokens in the layer's state slot.
    rewriter.setInsertionPointAfter(channelLoop);
    rewriter.setInsertionPointToStart(timeLoop.getBody());
    rewriter.setInsertionPointAfter(timeLoop);
    auto shiftLoop = rewriter.create<scf::ForOp>(loc, zero, stateLenValue, one,
                                                 ValueRange{});
    rewriter.setInsertionPointToStart(shiftLoop.getBody());
    auto shiftChannelLoop = rewriter.create<scf::ForOp>(
        loc, zero, channelsValue, one, ValueRange{});
    rewriter.setInsertionPointToStart(shiftChannelLoop.getBody());

    Value shiftTime = shiftLoop.getInductionVar();
    Value shiftChannel = shiftChannelLoop.getInductionVar();
    Value sourceIndex =
        rewriter.create<arith::SubIOp>(loc, sequence, stateLenValue);
    sourceIndex = rewriter.create<arith::AddIOp>(loc, sourceIndex, shiftTime);
    Value sourceNegative = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, sourceIndex, zero);
    auto sourceSelect = rewriter.create<scf::IfOp>(
        loc, TypeRange{rewriter.getBF16Type()}, sourceNegative,
        /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&sourceSelect.getThenRegion().front());
      Value oldIndex =
          rewriter.create<arith::AddIOp>(loc, sourceIndex, stateLenValue);
      Value value = rewriter.create<memref::LoadOp>(
          loc, op.getState(), ValueRange{layerValue, oldIndex, shiftChannel});
      rewriter.create<scf::YieldOp>(loc, value);
    }
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&sourceSelect.getElseRegion().front());
      Value value = rewriter.create<tensor::ExtractOp>(
          loc, op.getInput(), ValueRange{sourceIndex, shiftChannel});
      rewriter.create<scf::YieldOp>(loc, value);
    }
    rewriter.create<memref::StoreOp>(
        loc, sourceSelect.getResult(0), op.getState(),
        ValueRange{layerValue, shiftTime, shiftChannel});

    rewriter.setInsertionPointAfter(shiftLoop);
    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, output, /*restrict=*/true);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

} // namespace

void mlir::llmcp::populateCausalConv1dLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CausalConv1dOpLowering>(patterns.getContext());
}
