//===- FuseLinearActivations.cpp - Fuse activations into linears ----------===//
//
// Rewrites direct linear -> activation patterns into a linear with an
// activation epilogue. SwiGLU becomes linear(activation=silu) followed by mul.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/IR/LLMCP.h"
#include "llmcp/Dialect/LLMCP/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::llmcp;

namespace {

static bool hasOtherUse(Value value, Operation *except) {
  for (Operation *user : value.getUsers())
    if (user != except)
      return true;
  return false;
}

class FuseSiluMul : public OpRewritePattern<SiluMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SiluMulOp op,
                                PatternRewriter &rewriter) const override {
    auto linear = op.getGate().getDefiningOp<LinearOp>();
    if (!linear || hasOtherUse(linear.getOutput(), op) ||
        linear.getActivation() != "none")
      return failure();

    rewriter.modifyOpInPlace(linear, [&] { linear.setActivation("silu"); });
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getResult().getType(),
                                       linear.getOutput(), op.getUp());
    return success();
  }
};

class FuseGatedRmsNorm : public OpRewritePattern<GatedRmsNormOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GatedRmsNormOp op,
                                PatternRewriter &rewriter) const override {
    auto linear = op.getGate().getDefiningOp<LinearOp>();
    if (!linear || hasOtherUse(linear.getOutput(), op) ||
        linear.getActivation() != "none" || op.getGateActivated())
      return failure();

    rewriter.modifyOpInPlace(linear, [&] { linear.setActivation("silu"); });
    rewriter.modifyOpInPlace(op, [&] { op.setGateActivated(true); });
    return success();
  }
};

class FuseDeltaNetBeta : public OpRewritePattern<GatedDeltaNetOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GatedDeltaNetOp op,
                                PatternRewriter &rewriter) const override {
    auto linear = op.getB().getDefiningOp<LinearOp>();
    if (!linear || hasOtherUse(linear.getOutput(), op) ||
        linear.getActivation() != "none" || op.getBActivated())
      return failure();

    rewriter.modifyOpInPlace(linear, [&] { linear.setActivation("sigmoid"); });
    rewriter.modifyOpInPlace(op, [&] { op.setBActivated(true); });
    return success();
  }
};

struct FuseLinearActivationsPass
    : public PassWrapper<FuseLinearActivationsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseLinearActivationsPass)

  StringRef getArgument() const final {
    return "llmcp-fuse-linear-activations";
  }
  StringRef getDescription() const final {
    return "Fuse direct linear -> activation patterns into linear epilogues";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseSiluMul, FuseGatedRmsNorm, FuseDeltaNetBeta>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::llmcp::createFuseLinearActivationsPass() {
  return std::make_unique<FuseLinearActivationsPass>();
}

void mlir::llmcp::registerFuseLinearActivationsPass() {
  PassRegistration<FuseLinearActivationsPass>();
}
