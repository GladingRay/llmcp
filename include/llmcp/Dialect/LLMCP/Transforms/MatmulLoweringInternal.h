//===- MatmulLoweringInternal.h - Shared matmul lowering --------*- C++ -*-===//
//
// Internal entry point shared by the prefill GEMM and decode GEMV wrappers.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_DIALECT_LLMCP_TRANSFORMS_MATMULLOWERINGINTERNAL_H
#define LLMCP_DIALECT_LLMCP_TRANSFORMS_MATMULLOWERINGINTERNAL_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace llmcp {

Value applyMatmulActivation(PatternRewriter &rewriter, Location loc,
                            Value value, llvm::StringRef activation);

/// Returns the output buffer when the packed-layout decode GEMV is available.
/// The weight is a flat byte payload in the four-column lane layout.
Value emitQ8PackedDecodeMatmul(PatternRewriter &rewriter, Location loc,
                               Value input, Value weight,
                               RankedTensorType resultType,
                               llvm::StringRef activation);

/// Returns the output buffer when the packed-layout prefill GEMM is available.
Value emitQ8PackedMatmul(PatternRewriter &rewriter, Location loc, Value input,
                         Value weight, RankedTensorType resultType,
                         llvm::StringRef activation);

/// Portable scalar matmul over the packed layout, used without DotProd.
Value emitQ8PackedScalarMatmul(PatternRewriter &rewriter, Location loc,
                               Value input, Value weight,
                               RankedTensorType resultType,
                               llvm::StringRef activation);

Value lowerMatmulInternal(PatternRewriter &rewriter, Location loc, Value input,
                          Value weight, RankedTensorType resultType,
                          llvm::StringRef weightFormat, bool transposeB,
                          bool decode, llvm::StringRef activation);

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_DIALECT_LLMCP_TRANSFORMS_MATMULLOWERINGINTERNAL_H
