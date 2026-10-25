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

    Value matmul;
    if (op.getTransposeB()) {
      auto mm = rewriter.create<linalg::MatmulOp>(
          loc, TypeRange{accType}, ValueRange{input, weight}, ValueRange{acc});
      matmul = mm.getResult(0);
    } else {
      auto mm = rewriter.create<linalg::MatmulTransposeBOp>(
          loc, TypeRange{accType}, ValueRange{input, weight}, ValueRange{acc});
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
    patterns.add<LinearOpLowering, RmsNormOpLowering>(context);
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
