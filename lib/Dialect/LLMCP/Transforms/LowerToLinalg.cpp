//===- LowerToLinalg.cpp - Lower LLMCP ops to Linalg ----------------------===//
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/Passes.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCP.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
    Value weight = toTensor(rewriter, loc, op.getWeight());

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
    auto input = op.getInput();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !resultType || inputType.getRank() != 2)
      return failure();

    const int64_t hidden = inputType.getDimSize(1);
    if (ShapedType::isDynamic(hidden))
      return failure();

    auto f32Type = rewriter.getF32Type();
    Value xf = castToF32(rewriter, loc, input);
    Value weight = toTensor(rewriter, loc, op.getWeight());

    // sum(x^2) over the last dimension.
    auto rowType = RankedTensorType::get({inputType.getDimSize(0)}, f32Type);
    Value rowEmpty = createEmpty(rewriter, loc, rowType, input);
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value sumInit =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{rowEmpty})
            .getResult(0);

    SmallVector<AffineMap> sumMaps = {
        rewriter.getMultiDimIdentityMap(2),
        AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0)},
                       rewriter.getContext())};
    SmallVector<utils::IteratorType> sumIters = {
        utils::IteratorType::parallel, utils::IteratorType::reduction};
    auto sum = rewriter.create<linalg::GenericOp>(
        loc, rowType, ValueRange{xf}, ValueRange{sumInit}, sumMaps, sumIters,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value square = builder.create<arith::MulFOp>(loc, args[0], args[0]);
          Value sum = builder.create<arith::AddFOp>(loc, args[1], square);
          builder.create<linalg::YieldOp>(loc, sum);
        });

    // rstd = rsqrt(mean(x^2) + eps).
    Value rstdEmpty = createEmpty(rewriter, loc, rowType, input);
    SmallVector<AffineMap> statMaps = {rewriter.getMultiDimIdentityMap(1),
                                       rewriter.getMultiDimIdentityMap(1)};
    SmallVector<utils::IteratorType> statIters(1,
                                               utils::IteratorType::parallel);
    auto rstd = rewriter.create<linalg::GenericOp>(
        loc, rowType, ValueRange{sum.getResult(0)}, ValueRange{rstdEmpty},
        statMaps, statIters,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value dim = builder.create<arith::ConstantOp>(
              loc, builder.getF32FloatAttr(static_cast<float>(hidden)));
          Value mean = builder.create<arith::DivFOp>(loc, args[0], dim);
          Value eps = builder.create<arith::ConstantOp>(
              loc, builder.getF32FloatAttr(op.getEpsAttr().getValueAsDouble()));
          Value variance = builder.create<arith::AddFOp>(loc, mean, eps);
          Value value = builder.create<math::RsqrtOp>(loc, variance);
          builder.create<linalg::YieldOp>(loc, value);
        });

    Value output = createEmpty(rewriter, loc, resultType, input);
    SmallVector<AffineMap> maps = {
        rewriter.getMultiDimIdentityMap(2),
        AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                       rewriter.getContext()),
        AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0)},
                       rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(2)};
    SmallVector<utils::IteratorType> iterators(2,
                                               utils::IteratorType::parallel);
    auto scaled = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{xf, weight, rstd.getResult(0)},
        ValueRange{output}, maps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value w = args[1];
          if (w.getType().isBF16())
            w = builder.create<arith::ExtFOp>(loc, f32Type, w);
          if (op.getZeroCentered()) {
            Value one = builder.create<arith::ConstantOp>(
                loc, builder.getF32FloatAttr(1.0f));
            w = builder.create<arith::AddFOp>(loc, w, one);
          }
          Value normalized = builder.create<arith::MulFOp>(loc, args[0], w);
          Value value = builder.create<arith::MulFOp>(loc, normalized, args[2]);
          Value truncated = builder.create<arith::TruncFOp>(
              loc, resultType.getElementType(), value);
          builder.create<linalg::YieldOp>(loc, truncated);
        });

    rewriter.replaceOp(op, scaled.getResult(0));
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
    Value table = toTensor(rewriter, loc, op.getTable());
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
