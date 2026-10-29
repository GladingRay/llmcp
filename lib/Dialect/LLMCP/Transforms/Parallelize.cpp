//===- Parallelize.cpp - Lower marked loops to scf.parallel ---------------===//
//
// Loops marked with `llmcp.parallel` are independent outer loops.  Converting
// them to scf.parallel lets the OpenMP backend distribute them.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct ParallelizeLoopPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("llmcp.parallel"))
      return failure();
    if (!op.getResults().empty())
      return failure();

    auto parallel = rewriter.create<scf::ParallelOp>(
        op.getLoc(), ValueRange{op.getLowerBound()},
        ValueRange{op.getUpperBound()}, ValueRange{op.getStep()});
    Block *body = parallel.getBody();
    Operation *terminator = body->getTerminator();
    op.getInductionVar().replaceAllUsesWith(
        parallel.getInductionVars().front());
    for (Operation &nested :
         llvm::make_early_inc_range(op.getBody()->without_terminator()))
      nested.moveBefore(terminator);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ParallelizePass
    : public PassWrapper<ParallelizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelizePass)

  StringRef getArgument() const final { return "llmcp-parallelize"; }
  StringRef getDescription() const final {
    return "Convert llmcp.parallel scf.for loops to scf.parallel";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    context->getOrLoadDialect<arith::ArithDialect>();
    context->getOrLoadDialect<scf::SCFDialect>();
    context->getOrLoadDialect<LLVM::LLVMDialect>();

    RewritePatternSet patterns(context);
    patterns.add<ParallelizeLoopPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::llmcp::createParallelizePass() {
  return std::make_unique<ParallelizePass>();
}

void mlir::llmcp::registerParallelizePass() {
  PassRegistration<ParallelizePass>();
}
