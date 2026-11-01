//===- LLMCPTypes.cpp - LLMCP type implementation -------------------------===//
//
// Implementation of LLMCP custom types.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/IR/LLMCPTypes.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCPDialect.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "llmcp/Dialect/LLMCP/IR/LLMCPOpsTypes.cpp.inc"
