//===- DecodeNormalization.h - Decode normalization lowering ---*- C++ -*-===//
//
// Serial normalization kernels for single-token decoding.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_DIALECT_LLMCP_TRANSFORMS_DECODENORMALIZATION_H
#define LLMCP_DIALECT_LLMCP_TRANSFORMS_DECODENORMALIZATION_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace llmcp {

class RmsNormOp;
class GatedRmsNormOp;

Value emitDecodeRmsNorm(PatternRewriter &rewriter, RmsNormOp op);
Value emitDecodeGatedRmsNorm(PatternRewriter &rewriter, GatedRmsNormOp op);

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_DIALECT_LLMCP_TRANSFORMS_DECODENORMALIZATION_H
