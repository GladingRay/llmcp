//===- MatmulLowering.h - Prefill/decode matmul lowering --------*- C++ -*-===//
//
// Declarations for the independent prefill GEMM and decode GEMV lowering paths.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_DIALECT_LLMCP_TRANSFORMS_MATMULLOWERING_H
#define LLMCP_DIALECT_LLMCP_TRANSFORMS_MATMULLOWERING_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace llmcp {

/// Lowers a prefill matmul as a GEMM.
Value lowerPrefillGemm(PatternRewriter &rewriter, Location loc, Value input,
                       Value weight, RankedTensorType resultType,
                       llvm::StringRef weightFormat, bool transposeB);

/// Lowers a decode matmul as a GEMV (M is one).
Value lowerDecodeGemm(PatternRewriter &rewriter, Location loc, Value input,
                      Value weight, RankedTensorType resultType,
                      llvm::StringRef weightFormat, bool transposeB);

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_DIALECT_LLMCP_TRANSFORMS_MATMULLOWERING_H
