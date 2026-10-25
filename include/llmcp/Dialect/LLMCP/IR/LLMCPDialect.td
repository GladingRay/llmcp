//===- LLMCPDialect.td - LLMCP dialect definition ----------*- tablegen -*-===//
//
// Dialect definition for the high-level llmcp inference IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_DIALECT_TD
#define LLMCP_DIALECT_TD

include "mlir/IR/OpBase.td"

def LLMCP_Dialect : Dialect {
  let name = "llmcp";
  let cppNamespace = "::mlir::llmcp";

  let summary = "High-level dialect for hybrid LLM inference graphs";

  let description = [{
    `llmcp` models the parts of a hybrid LLM that cannot be expressed cleanly
    with pure tensor dataflow alone: stateful linear attention, KV-cache
    attention, and quantization-aware matrix products.

    Activations use tensor semantics. Weights and persistent state use memref
    semantics so that in-place state updates remain explicit.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "math::MathDialect",
    "tensor::TensorDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
  ];
}

#endif // LLMCP_DIALECT_TD
