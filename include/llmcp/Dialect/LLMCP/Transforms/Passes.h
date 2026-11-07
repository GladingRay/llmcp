//===- Passes.h - LLMCP transformation passes -------------------*- C++ -*-===//
//
// Pass factories and registration for LLMCP transformations.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_DIALECT_LLMCP_TRANSFORMS_PASSES_H
#define LLMCP_DIALECT_LLMCP_TRANSFORMS_PASSES_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace llmcp {

/// Lowers `llmcp.linear` and `llmcp.rms_norm` to linalg/tensor operations.
std::unique_ptr<Pass> createLowerToLinalgPass();
std::unique_ptr<Pass> createParallelizePass();
std::unique_ptr<Pass> createFuseLinearActivationsPass();

void populateCausalConv1dLoweringPatterns(RewritePatternSet &patterns);
void populateGatedDeltaNetLoweringPatterns(RewritePatternSet &patterns);
void populateAttentionLoweringPatterns(RewritePatternSet &patterns);

void registerLowerToLinalgPass();
void registerParallelizePass();
void registerFuseLinearActivationsPass();

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_DIALECT_LLMCP_TRANSFORMS_PASSES_H
