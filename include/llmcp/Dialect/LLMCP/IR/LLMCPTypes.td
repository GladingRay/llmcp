//===- LLMCPTypes.td - LLMCP type definitions ----------------*- tablegen -*-===//
//
// The initial dialect intentionally uses builtin tensor and memref types.
// Custom state types (for example a typed KV cache) can be added here later
// without changing the op surface.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_TYPES_TD
#define LLMCP_TYPES_TD

include "mlir/IR/OpBase.td"

#endif // LLMCP_TYPES_TD
