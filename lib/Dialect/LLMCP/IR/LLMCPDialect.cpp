//===- LLMCPDialect.cpp - LLMCP dialect implementation --------------------===//
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/IR/LLMCPDialect.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCPOps.h"
#include "llmcp/Dialect/LLMCP/IR/LLMCPTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCPOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::llmcp;

void LLMCPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "llmcp/Dialect/LLMCP/IR/LLMCPOps.cpp.inc"
      >();
}
