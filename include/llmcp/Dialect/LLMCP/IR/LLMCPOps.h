//===- LLMCPOps.h - LLMCP operation declarations ----------------*- C++ -*-===//
//
// Declarations of LLMCP operation classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_DIALECT_LLMCP_IR_LLMCPOPS_H
#define LLMCP_DIALECT_LLMCP_IR_LLMCPOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCPDialect.h"

#define GET_OP_CLASSES
#include "llmcp/Dialect/LLMCP/IR/LLMCPOps.h.inc"

#endif // LLMCP_DIALECT_LLMCP_IR_LLMCPOPS_H
