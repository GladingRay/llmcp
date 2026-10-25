//===- Passes.h - LLMCP transformation passes -------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_DIALECT_LLMCP_TRANSFORMS_PASSES_H
#define LLMCP_DIALECT_LLMCP_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace llmcp {

/// Lowers `llmcp.linear` and `llmcp.rms_norm` to linalg/tensor operations.
std::unique_ptr<Pass> createLowerToLinalgPass();

void registerLowerToLinalgPass();

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_DIALECT_LLMCP_TRANSFORMS_PASSES_H
