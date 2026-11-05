//===- LowerAttention.cpp - Lower llmcp.attention -------------------------===//
//
// FlashAttention-2 style online softmax and KV-cache lowering.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/Passes.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCP.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

/// Returns true when the enclosing function was built for prefill.
bool isPrefillMode(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func)
    return false;
  auto mode = func->getAttrOfType<StringAttr>("llmcp.execution_mode");
  return mode && mode.getValue() == "prefill";
}

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

class AttentionOpLowering : public OpRewritePattern<AttentionOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AttentionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    const bool prefill = isPrefillMode(op);
    auto qgType = dyn_cast<RankedTensorType>(op.getQg().getType());
    auto kType = dyn_cast<RankedTensorType>(op.getK().getType());
    auto vType = dyn_cast<RankedTensorType>(op.getV().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    auto cacheType = dyn_cast<MemRefType>(op.getKvCache().getType());
    if (!qgType || !kType || !vType || !resultType || !cacheType)
      return failure();
    if (qgType.getRank() != 2 || kType.getRank() != 2 || vType.getRank() != 2 ||
        resultType.getRank() != 2 || cacheType.getRank() != 5)
      return failure();

    const int64_t numHeads = op.getNumHeads();
    const int64_t numKvHeads = op.getNumKvHeads();
    const int64_t headDim = op.getHeadDim();
    const int64_t rotaryDim = op.getRotaryDim();
    if (numHeads <= 0 || numKvHeads <= 0 || numHeads % numKvHeads != 0 ||
        headDim <= 0 || rotaryDim < 0 || rotaryDim > headDim ||
        (rotaryDim % 2) != 0)
      return failure();
    const int64_t group = numHeads / numKvHeads;
    const int64_t halfRotary = rotaryDim / 2;

    auto f32 = rewriter.getF32Type();
    auto bf16 = rewriter.getBF16Type();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value headDimValue = rewriter.create<arith::ConstantIndexOp>(loc, headDim);
    Value numHeadsValue =
        rewriter.create<arith::ConstantIndexOp>(loc, numHeads);
    Value numKvHeadsValue =
        rewriter.create<arith::ConstantIndexOp>(loc, numKvHeads);
    Value rotaryValue = rewriter.create<arith::ConstantIndexOp>(loc, rotaryDim);
    Value halfRotaryValue =
        rewriter.create<arith::ConstantIndexOp>(loc, halfRotary);
    Value groupValue = rewriter.create<arith::ConstantIndexOp>(loc, group);
    Value layerValue =
        rewriter.create<arith::ConstantIndexOp>(loc, op.getLayerIndex());
    Value writePosition = op.getPosition();
    Value cacheLength = op.getCacheLength();
    Value sequence;
    if (qgType.isDynamicDim(0))
      sequence = rewriter.create<tensor::DimOp>(loc, op.getQg(), 0);
    else
      sequence =
          rewriter.create<arith::ConstantIndexOp>(loc, qgType.getDimSize(0));

    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    Value scaleValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(scale));
    Value zeroF =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value oneF =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0f));
    Value eps = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(1.0e-6f));
    Value invHeadDim = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(1.0f / static_cast<float>(headDim)));
    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(-1.0e30f));

    SmallVector<Value> sequenceSizes;
    if (qgType.isDynamicDim(0))
      sequenceSizes.push_back(sequence);
    Value qNormalized = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get({qgType.getDimSize(0), numHeads, headDim}, bf16),
        sequenceSizes);
    constexpr int64_t kBlock = 64;
    Value blockValue = rewriter.create<arith::ConstantIndexOp>(loc, kBlock);
    Value scratchRows =
        rewriter.create<arith::MulIOp>(loc, sequence, numHeadsValue);
    Value acc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get({ShapedType::kDynamic, headDim}, f32),
        ValueRange{scratchRows});
    Value scoreTile = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get({ShapedType::kDynamic, kBlock}, f32),
        ValueRange{scratchRows});
    Value normTemp = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get({numHeads, headDim}, f32));
    SmallVector<Value> outputSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i)
      if (resultType.isDynamicDim(i))
        outputSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, op.getQg(), i));
    Value output = rewriter.create<memref::AllocOp>(
        loc,
        MemRefType::get(resultType.getShape(), resultType.getElementType()),
        outputSizes);

    auto loadBf16 = [&](Value tensor, Value index0, Value index1) -> Value {
      Value value = rewriter.create<tensor::ExtractOp>(
          loc, tensor, ValueRange{index0, index1});
      return rewriter.create<arith::ExtFOp>(loc, f32, value);
    };
    auto siluLikeSigmoid = [&](Value value) -> Value {
      return rewriter.create<arith::DivFOp>(
          loc, oneF,
          rewriter.create<arith::AddFOp>(
              loc, oneF,
              rewriter.create<math::ExpOp>(
                  loc, rewriter.create<arith::NegFOp>(loc, value))));
    };

    // Normalize q and apply partial RoPE. qg layout is [head][q | gate].
    auto qNormLoop = createLoop(
        rewriter, loc, zero, sequence, one, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value position, ValueRange) {
          auto headLoop = createLoop(
              rewriter, loc, zero, numHeadsValue, one, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value head, ValueRange) {
                Value qBase = rewriter.create<arith::MulIOp>(
                    loc, head,
                    rewriter.create<arith::ConstantIndexOp>(loc, 2 * headDim));
                auto sumLoop = createLoop(
                    rewriter, loc, zero, headDimValue, one, ValueRange{zeroF},
                    [&](OpBuilder &builder, Location loc, Value d,
                        ValueRange args) {
                      Value value = loadBf16(
                          op.getQg(), position,
                          rewriter.create<arith::AddIOp>(loc, qBase, d));
                      Value square =
                          rewriter.create<arith::MulFOp>(loc, value, value);
                      Value sum =
                          rewriter.create<arith::AddFOp>(loc, args[0], square);
                      rewriter.create<scf::YieldOp>(loc, sum);
                    });
                rewriter.setInsertionPointAfter(sumLoop);
                Value mean = rewriter.create<arith::MulFOp>(
                    loc, sumLoop.getResult(0), invHeadDim);
                Value inv = rewriter.create<math::RsqrtOp>(
                    loc, rewriter.create<arith::AddFOp>(loc, mean, eps));
                auto normalize = createLoop(
                    rewriter, loc, zero, headDimValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value d, ValueRange) {
                      Value value = loadBf16(
                          op.getQg(), position,
                          rewriter.create<arith::AddIOp>(loc, qBase, d));
                      Value weightValue = rewriter.create<memref::LoadOp>(
                          loc, op.getQNorm(), ValueRange{d});
                      Value weightF32 =
                          rewriter.create<arith::ExtFOp>(loc, f32, weightValue);
                      Value centered =
                          rewriter.create<arith::AddFOp>(loc, oneF, weightF32);
                      Value normalized = rewriter.create<arith::MulFOp>(
                          loc, rewriter.create<arith::MulFOp>(loc, value, inv),
                          centered);
                      rewriter.create<memref::StoreOp>(
                          loc, normalized, normTemp, ValueRange{head, d});
                    });
                rewriter.setInsertionPointAfter(normalize);

                auto rope = createLoop(
                    rewriter, loc, zero, rotaryValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value d, ValueRange) {
                      Value isSecond = rewriter.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::uge, d, halfRotaryValue);
                      Value partner = rewriter.create<arith::SelectOp>(
                          loc, isSecond,
                          rewriter.create<arith::SubIOp>(loc, d,
                                                         halfRotaryValue),
                          rewriter.create<arith::AddIOp>(loc, d,
                                                         halfRotaryValue));
                      Value sign = rewriter.create<arith::SelectOp>(
                          loc, isSecond,
                          rewriter.create<arith::ConstantOp>(
                              loc, builder.getF32FloatAttr(1.0f)),
                          rewriter.create<arith::ConstantOp>(
                              loc, builder.getF32FloatAttr(-1.0f)));
                      Value value = rewriter.create<memref::LoadOp>(
                          loc, normTemp, ValueRange{head, d});
                      Value partnerValue = rewriter.create<memref::LoadOp>(
                          loc, normTemp, ValueRange{head, partner});
                      Value cos = rewriter.create<tensor::ExtractOp>(
                          loc, op.getCos(), ValueRange{position, d});
                      Value sin = rewriter.create<tensor::ExtractOp>(
                          loc, op.getSin(), ValueRange{position, d});
                      Value rotated = rewriter.create<arith::MulFOp>(
                          loc, sign, partnerValue);
                      Value result = rewriter.create<arith::AddFOp>(
                          loc, rewriter.create<arith::MulFOp>(loc, value, cos),
                          rewriter.create<arith::MulFOp>(loc, rotated, sin));
                      Value resultBf16 =
                          rewriter.create<arith::TruncFOp>(loc, bf16, result);
                      rewriter.create<memref::StoreOp>(
                          loc, resultBf16, qNormalized,
                          ValueRange{position, head, d});
                    });
                rewriter.setInsertionPointAfter(rope);

                auto tail = createLoop(
                    rewriter, loc, rotaryValue, headDimValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value d, ValueRange) {
                      Value value = rewriter.create<memref::LoadOp>(
                          loc, normTemp, ValueRange{head, d});
                      Value resultBf16 =
                          rewriter.create<arith::TruncFOp>(loc, bf16, value);
                      rewriter.create<memref::StoreOp>(
                          loc, resultBf16, qNormalized,
                          ValueRange{position, head, d});
                    });
                rewriter.setInsertionPointAfter(tail);
              });
          if (!prefill)
            headLoop->setAttr("llmcp.parallel",
                              UnitAttr::get(rewriter.getContext()));
          rewriter.setInsertionPointAfter(headLoop);
        });
    rewriter.setInsertionPointAfter(qNormLoop);

    // Normalize k, apply RoPE, and write k/v into the KV cache.
    auto kvNormLoop = createLoop(
        rewriter, loc, zero, sequence, one, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value position, ValueRange) {
          auto kvHeadLoop = createLoop(
              rewriter, loc, zero, numKvHeadsValue, one, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value kvHead, ValueRange) {
                Value base = rewriter.create<arith::MulIOp>(
                    loc, kvHead,
                    rewriter.create<arith::ConstantIndexOp>(loc, headDim));
                auto sumLoop = createLoop(
                    rewriter, loc, zero, headDimValue, one, ValueRange{zeroF},
                    [&](OpBuilder &builder, Location loc, Value d,
                        ValueRange args) {
                      Value value = loadBf16(
                          op.getK(), position,
                          rewriter.create<arith::AddIOp>(loc, base, d));
                      Value square =
                          rewriter.create<arith::MulFOp>(loc, value, value);
                      Value sum =
                          rewriter.create<arith::AddFOp>(loc, args[0], square);
                      rewriter.create<scf::YieldOp>(loc, sum);
                    });
                rewriter.setInsertionPointAfter(sumLoop);
                Value mean = rewriter.create<arith::MulFOp>(
                    loc, sumLoop.getResult(0), invHeadDim);
                Value inv = rewriter.create<math::RsqrtOp>(
                    loc, rewriter.create<arith::AddFOp>(loc, mean, eps));
                auto normalize = createLoop(
                    rewriter, loc, zero, headDimValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value d, ValueRange) {
                      Value value = loadBf16(
                          op.getK(), position,
                          rewriter.create<arith::AddIOp>(loc, base, d));
                      Value weightValue = rewriter.create<memref::LoadOp>(
                          loc, op.getKNorm(), ValueRange{d});
                      Value weightF32 =
                          rewriter.create<arith::ExtFOp>(loc, f32, weightValue);
                      Value centered =
                          rewriter.create<arith::AddFOp>(loc, oneF, weightF32);
                      Value normalized = rewriter.create<arith::MulFOp>(
                          loc, rewriter.create<arith::MulFOp>(loc, value, inv),
                          centered);
                      rewriter.create<memref::StoreOp>(
                          loc, normalized, normTemp, ValueRange{kvHead, d});
                    });
                rewriter.setInsertionPointAfter(normalize);

                auto writeK = createLoop(
                    rewriter, loc, zero, rotaryValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value d, ValueRange) {
                      Value second = rewriter.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::uge, d, halfRotaryValue);
                      Value partner = rewriter.create<arith::SelectOp>(
                          loc, second,
                          rewriter.create<arith::SubIOp>(loc, d,
                                                         halfRotaryValue),
                          rewriter.create<arith::AddIOp>(loc, d,
                                                         halfRotaryValue));
                      Value sign = rewriter.create<arith::SelectOp>(
                          loc, second,
                          rewriter.create<arith::ConstantOp>(
                              loc, builder.getF32FloatAttr(1.0f)),
                          rewriter.create<arith::ConstantOp>(
                              loc, builder.getF32FloatAttr(-1.0f)));
                      Value value = rewriter.create<memref::LoadOp>(
                          loc, normTemp, ValueRange{kvHead, d});
                      Value partnerValue = rewriter.create<memref::LoadOp>(
                          loc, normTemp, ValueRange{kvHead, partner});
                      Value cos = rewriter.create<tensor::ExtractOp>(
                          loc, op.getCos(), ValueRange{position, d});
                      Value sin = rewriter.create<tensor::ExtractOp>(
                          loc, op.getSin(), ValueRange{position, d});
                      Value rotated = rewriter.create<arith::MulFOp>(
                          loc, sign, partnerValue);
                      Value result = rewriter.create<arith::AddFOp>(
                          loc, rewriter.create<arith::MulFOp>(loc, value, cos),
                          rewriter.create<arith::MulFOp>(loc, rotated, sin));
                      Value resultBf16 =
                          rewriter.create<arith::TruncFOp>(loc, bf16, result);
                      rewriter.create<memref::StoreOp>(
                          loc, resultBf16, op.getKvCache(),
                          ValueRange{layerValue, zero, kvHead,
                                     rewriter.create<arith::AddIOp>(
                                         loc, writePosition, position),
                                     d});
                    });
                rewriter.setInsertionPointAfter(writeK);

                auto writeKTail = createLoop(
                    rewriter, loc, rotaryValue, headDimValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value d, ValueRange) {
                      Value value = rewriter.create<memref::LoadOp>(
                          loc, normTemp, ValueRange{kvHead, d});
                      Value resultBf16 =
                          rewriter.create<arith::TruncFOp>(loc, bf16, value);
                      rewriter.create<memref::StoreOp>(
                          loc, resultBf16, op.getKvCache(),
                          ValueRange{layerValue, zero, kvHead,
                                     rewriter.create<arith::AddIOp>(
                                         loc, writePosition, position),
                                     d});
                    });
                rewriter.setInsertionPointAfter(writeKTail);

                auto writeV = createLoop(
                    rewriter, loc, zero, headDimValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value d, ValueRange) {
                      Value value = rewriter.create<tensor::ExtractOp>(
                          loc, op.getV(),
                          ValueRange{position, rewriter.create<arith::AddIOp>(
                                                   loc, base, d)});
                      rewriter.create<memref::StoreOp>(
                          loc, value, op.getKvCache(),
                          ValueRange{layerValue, one, kvHead,
                                     rewriter.create<arith::AddIOp>(
                                         loc, writePosition, position),
                                     d});
                    });
                rewriter.setInsertionPointAfter(writeV);
              });
          if (!prefill)
            kvHeadLoop->setAttr("llmcp.parallel",
                                UnitAttr::get(rewriter.getContext()));
          rewriter.setInsertionPointAfter(kvHeadLoop);
        });
    rewriter.setInsertionPointAfter(kvNormLoop);

    // Causal attention with a materialized score buffer.
    auto queryLoop = createLoop(
        rewriter, loc, zero, sequence, one, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value queryPos, ValueRange) {
          auto queryHeadLoop = createLoop(
              rewriter, loc, zero, numHeadsValue, one, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value head, ValueRange) {
                Value scratchHead = rewriter.create<arith::AddIOp>(
                    loc,
                    rewriter.create<arith::MulIOp>(loc, queryPos,
                                                   numHeadsValue),
                    head);
                Value kvHead =
                    rewriter.create<arith::DivUIOp>(loc, head, groupValue);
                Value queryEnd = rewriter.create<arith::AddIOp>(
                    loc,
                    rewriter.create<arith::AddIOp>(loc, writePosition,
                                                   queryPos),
                    one);

                // Initialize the output accumulator for this head.
                auto initAcc = createLoop(
                    rewriter, loc, zero, headDimValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value d, ValueRange) {
                      rewriter.create<memref::StoreOp>(
                          loc, zeroF, acc, ValueRange{scratchHead, d});
                    });
                rewriter.setInsertionPointAfter(initAcc);

                // FlashAttention-2 style online softmax over key blocks.
                auto keyBlockLoop = createLoop(
                    rewriter, loc, zero, queryEnd, blockValue,
                    ValueRange{negInf, zeroF},
                    [&](OpBuilder &builder, Location loc, Value blockStart,
                        ValueRange blockArgs) {
                      Value runningMax = blockArgs[0];
                      Value runningSum = blockArgs[1];
                      Value blockEnd = rewriter.create<arith::MinUIOp>(
                          loc,
                          rewriter.create<arith::AddIOp>(loc, blockStart,
                                                         blockValue),
                          queryEnd);

                      auto tileMaxLoop = createLoop(
                          rewriter, loc, blockStart, blockEnd, one,
                          ValueRange{negInf},
                          [&](OpBuilder &builder, Location loc, Value keyPos,
                              ValueRange args) {
                            auto dotLoop = createLoop(
                                rewriter, loc, zero, headDimValue, one,
                                ValueRange{zeroF},
                                [&](OpBuilder &builder, Location loc, Value d,
                                    ValueRange dotArgs) {
                                  Value qValue =
                                      rewriter.create<memref::LoadOp>(
                                          loc, qNormalized,
                                          ValueRange{queryPos, head, d});
                                  Value kValue =
                                      rewriter.create<memref::LoadOp>(
                                          loc, op.getKvCache(),
                                          ValueRange{layerValue, zero, kvHead,
                                                     keyPos, d});
                                  Value qF32 = rewriter.create<arith::ExtFOp>(
                                      loc, f32, qValue);
                                  Value kF32 = rewriter.create<arith::ExtFOp>(
                                      loc, f32, kValue);
                                  Value product =
                                      rewriter.create<arith::MulFOp>(loc, qF32,
                                                                     kF32);
                                  Value sum = rewriter.create<arith::AddFOp>(
                                      loc, dotArgs[0], product);
                                  rewriter.create<scf::YieldOp>(loc, sum);
                                });
                            rewriter.setInsertionPointAfter(dotLoop);
                            Value score = rewriter.create<arith::MulFOp>(
                                loc, dotLoop.getResult(0), scaleValue);
                            rewriter.create<memref::StoreOp>(
                                loc, score, scoreTile,
                                ValueRange{scratchHead,
                                           rewriter.create<arith::SubIOp>(
                                               loc, keyPos, blockStart)});
                            Value max = rewriter.create<arith::MaxNumFOp>(
                                loc, args[0], score);
                            rewriter.create<scf::YieldOp>(loc, max);
                          });
                      rewriter.setInsertionPointAfter(tileMaxLoop);

                      Value newMax = rewriter.create<arith::MaxNumFOp>(
                          loc, runningMax, tileMaxLoop.getResult(0));
                      Value correction = rewriter.create<math::ExpOp>(
                          loc, rewriter.create<arith::SubFOp>(loc, runningMax,
                                                              newMax));

                      auto rescaleAcc = createLoop(
                          rewriter, loc, zero, headDimValue, one, ValueRange{},
                          [&](OpBuilder &builder, Location loc, Value d,
                              ValueRange) {
                            Value value = rewriter.create<memref::LoadOp>(
                                loc, acc, ValueRange{scratchHead, d});
                            Value scaled = rewriter.create<arith::MulFOp>(
                                loc, value, correction);
                            rewriter.create<memref::StoreOp>(
                                loc, scaled, acc, ValueRange{scratchHead, d});
                          });
                      rewriter.setInsertionPointAfter(rescaleAcc);
                      Value scaledSum = rewriter.create<arith::MulFOp>(
                          loc, runningSum, correction);

                      auto weightLoop = createLoop(
                          rewriter, loc, blockStart, blockEnd, one,
                          ValueRange{scaledSum},
                          [&](OpBuilder &builder, Location loc, Value keyPos,
                              ValueRange weightArgs) {
                            Value score = rewriter.create<memref::LoadOp>(
                                loc, scoreTile,
                                ValueRange{scratchHead,
                                           rewriter.create<arith::SubIOp>(
                                               loc, keyPos, blockStart)});
                            Value probability = rewriter.create<math::ExpOp>(
                                loc, rewriter.create<arith::SubFOp>(loc, score,
                                                                    newMax));
                            Value newSum = rewriter.create<arith::AddFOp>(
                                loc, weightArgs[0], probability);
                            auto valueLoop = createLoop(
                                rewriter, loc, zero, headDimValue, one,
                                ValueRange{},
                                [&](OpBuilder &builder, Location loc, Value d,
                                    ValueRange) {
                                  Value vValue =
                                      rewriter.create<memref::LoadOp>(
                                          loc, op.getKvCache(),
                                          ValueRange{layerValue, one, kvHead,
                                                     keyPos, d});
                                  Value vF32 = rewriter.create<arith::ExtFOp>(
                                      loc, f32, vValue);
                                  Value product =
                                      rewriter.create<arith::MulFOp>(
                                          loc, probability, vF32);
                                  Value current =
                                      rewriter.create<memref::LoadOp>(
                                          loc, acc, ValueRange{scratchHead, d});
                                  Value updated =
                                      rewriter.create<arith::AddFOp>(
                                          loc, current, product);
                                  rewriter.create<memref::StoreOp>(
                                      loc, updated, acc,
                                      ValueRange{scratchHead, d});
                                });
                            rewriter.setInsertionPointAfter(valueLoop);
                            rewriter.create<scf::YieldOp>(loc, newSum);
                          });
                      rewriter.setInsertionPointAfter(weightLoop);
                      rewriter.create<scf::YieldOp>(
                          loc, ValueRange{newMax, weightLoop.getResult(0)});
                    });
                rewriter.setInsertionPointAfter(keyBlockLoop);
                Value finalSum = keyBlockLoop.getResult(1);

                auto outputDimLoop = createLoop(
                    rewriter, loc, zero, headDimValue, one, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value d, ValueRange) {
                      Value accumulated = rewriter.create<memref::LoadOp>(
                          loc, acc, ValueRange{scratchHead, d});
                      Value normalized = rewriter.create<arith::DivFOp>(
                          loc, accumulated, finalSum);
                      Value gateRaw = rewriter.create<tensor::ExtractOp>(
                          loc, op.getQg(),
                          ValueRange{
                              queryPos,
                              rewriter.create<arith::AddIOp>(
                                  loc,
                                  rewriter.create<arith::AddIOp>(
                                      loc,
                                      rewriter.create<arith::MulIOp>(
                                          loc, head,
                                          rewriter
                                              .create<arith::ConstantIndexOp>(
                                                  loc, 2 * headDim)),
                                      rewriter.create<arith::ConstantIndexOp>(
                                          loc, headDim)),
                                  d)});
                      Value gateF32 =
                          rewriter.create<arith::ExtFOp>(loc, f32, gateRaw);
                      Value gate = siluLikeSigmoid(gateF32);
                      Value gated =
                          rewriter.create<arith::MulFOp>(loc, normalized, gate);
                      Value resultBf16 =
                          rewriter.create<arith::TruncFOp>(loc, bf16, gated);
                      Value column = rewriter.create<arith::AddIOp>(
                          loc,
                          rewriter.create<arith::MulIOp>(loc, head,
                                                         headDimValue),
                          d);
                      rewriter.create<memref::StoreOp>(
                          loc, resultBf16, output,
                          ValueRange{queryPos, column});
                    });
                rewriter.setInsertionPointAfter(outputDimLoop);
              });
          if (!prefill)
            queryHeadLoop->setAttr("llmcp.parallel",
                                   UnitAttr::get(rewriter.getContext()));
          rewriter.setInsertionPointAfter(queryHeadLoop);
        });
    if (prefill)
      queryLoop->setAttr("llmcp.parallel",
                         UnitAttr::get(rewriter.getContext()));
    rewriter.setInsertionPointAfter(queryLoop);

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, output, /*restrict=*/true);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

} // namespace

void mlir::llmcp::populateAttentionLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AttentionOpLowering>(patterns.getContext());
}
