//===- LLMCPTypes.cpp - LLMCP type implementation
//--------------------------===//
//
// The initial dialect has no custom types. This translation unit keeps the
// generated type definitions linked and gives future types a natural home.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/IR/LLMCPTypes.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCPDialect.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "llmcp/Dialect/LLMCP/IR/LLMCPOpsTypes.cpp.inc"
