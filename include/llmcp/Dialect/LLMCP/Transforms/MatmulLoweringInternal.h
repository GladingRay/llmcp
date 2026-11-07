//===- MatmulLoweringInternal.h - Shared matmul lowering --------*- C++ -*-===//
//
// Internal entry point shared by the prefill GEMM and decode GEMV wrappers.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_DIALECT_LLMCP_TRANSFORMS_MATMULLOWERINGINTERNAL_H
#define LLMCP_DIALECT_LLMCP_TRANSFORMS_MATMULLOWERINGINTERNAL_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace llmcp {

Value lowerMatmulInternal(PatternRewriter &rewriter, Location loc, Value input,
                          Value weight, RankedTensorType resultType,
                          llvm::StringRef weightFormat, bool transposeB,
                          bool decode, llvm::StringRef activation);

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_DIALECT_LLMCP_TRANSFORMS_MATMULLOWERINGINTERNAL_H
