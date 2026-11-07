//===- LowerPrefillGemm.cpp - Prefill GEMM lowering -----------------------===//
//
// Entry point for prefill GEMM lowering.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/MatmulLowering.h"
#include "llmcp/Dialect/LLMCP/Transforms/MatmulLoweringInternal.h"

using namespace mlir;
using namespace mlir::llmcp;
using llvm::StringRef;

Value mlir::llmcp::lowerPrefillGemm(PatternRewriter &rewriter, Location loc,
                                   Value input, Value weight,
                                   RankedTensorType resultType,
                                   StringRef weightFormat, bool transposeB,
                                   StringRef activation) {
  return lowerMatmulInternal(rewriter, loc, input, weight, resultType,
                             weightFormat, transposeB, /*decode=*/false,
                             activation);
}
