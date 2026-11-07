//===- LowerDecodeGemm.cpp - Decode GEMV lowering -------------------------===//
//
// Entry point for decode GEMV lowering.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/MatmulLowering.h"
#include "llmcp/Dialect/LLMCP/Transforms/MatmulLoweringInternal.h"

using namespace mlir;
using namespace mlir::llmcp;
using llvm::StringRef;

Value mlir::llmcp::lowerDecodeGemm(PatternRewriter &rewriter, Location loc,
                                  Value input, Value weight,
                                  RankedTensorType resultType,
                                  StringRef weightFormat, bool transposeB,
                                  StringRef activation) {
  return lowerMatmulInternal(rewriter, loc, input, weight, resultType,
                             weightFormat, transposeB, /*decode=*/true,
                             activation);
}
