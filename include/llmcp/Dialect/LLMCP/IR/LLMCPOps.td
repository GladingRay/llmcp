//===- LLMCPOps.td - LLMCP operation definitions -------------*- tablegen -*-===//
//
// High-level operations for the initial llmcp dialect. The op set is kept
// small on purpose: lowering and scheduling decisions live in passes, not in
// the IR surface.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_OPS_TD
#define LLMCP_OPS_TD

include "llmcp/Dialect/LLMCP/IR/LLMCPDialect.td"
include "llmcp/Dialect/LLMCP/IR/LLMCPTypes.td"
include "llmcp/Dialect/LLMCP/IR/LLMCPAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class LLMCP_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLMCP_Dialect, mnemonic, traits>;

def LLMCP_LinearOp : LLMCP_Op<"linear", [Pure]> {
  let summary = "Matrix product with f32 accumulation";
  let description = [{
    Computes `output = input * weight^T` when `transpose_b` is false, or
    `output = input * weight` when `transpose_b` is true.

    `input` is a rank-2 tensor, `weight` is a rank-2 memref, and `output` is a
    rank-2 tensor. The multiplication is accumulated in f32 and the result is
    converted back to the element type of `output`.
  }];

  let arguments = (ins AnyTensor:$input,
                       AnyMemRef:$weight,
                       DefaultValuedAttr<BoolAttr, "false">:$transpose_b);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $weight attr-dict `:` type($input) `,` type($weight)
      `->` type($output)
  }];

  let hasVerifier = 1;
}

def LLMCP_RmsNormOp : LLMCP_Op<"rms_norm", [Pure]> {
  let summary = "RMS normalization over the last dimension";
  let description = [{
    Normalizes the last dimension of `input` with an RMS norm. When
    `zero_centered` is true, the stored weight is interpreted as an offset:
    `y = x_norm * (1 + weight)`. This matches the Qwen3.5 checkpoint format.

    The reduction and scaling are performed in f32; the result has the same
    element type as `input`.
  }];

  let arguments = (ins AnyTensor:$input,
                       AnyMemRef:$weight,
                       DefaultValuedAttr<F64Attr, "1.0e-6">:$eps,
                       DefaultValuedAttr<BoolAttr, "true">:$zero_centered);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $weight attr-dict `:` type($input) `,` type($weight)
      `->` type($output)
  }];

  let hasVerifier = 1;
}

#endif // LLMCP_OPS_TD
