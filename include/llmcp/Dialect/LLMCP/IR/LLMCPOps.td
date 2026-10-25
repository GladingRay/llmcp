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

def LLMCP_AddOp : LLMCP_Op<"add", [Pure]> {
  let summary = "Elementwise addition of two tensors";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def LLMCP_SiluMulOp : LLMCP_Op<"silu_mul", [Pure]> {
  let summary = "SwiGLU activation: silu(gate) * up";
  let arguments = (ins AnyTensor:$gate, AnyTensor:$up);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def LLMCP_EmbeddingOp : LLMCP_Op<"embedding", [Pure]> {
  let summary = "Token embedding lookup";
  let arguments = (ins AnyTensor:$tokens, AnyMemRef:$table);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def LLMCP_LmHeadOp : LLMCP_Op<"lm_head", [Pure]> {
  let summary = "Tied embedding matrix product producing logits";
  let arguments = (ins AnyTensor:$hidden, AnyMemRef:$table);
  let results = (outs AnyTensor:$logits);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def LLMCP_CausalConv1dOp : LLMCP_Op<"causal_conv1d"> {
  let summary = "Depthwise causal convolution with persistent state";
  let arguments = (ins AnyTensor:$input,
                       AnyMemRef:$weight,
                       AnyMemRef:$state,
                       I64Attr:$layer_index,
                       I64Attr:$kernel_size,
                       StrAttr:$activation);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def LLMCP_GatedDeltaNetOp : LLMCP_Op<"gated_delta_net"> {
  let summary = "Gated DeltaNet recurrence with persistent state";
  let arguments = (ins AnyTensor:$conv,
                       AnyTensor:$a,
                       AnyTensor:$b,
                       AnyMemRef:$a_log,
                       AnyMemRef:$dt_bias,
                       AnyMemRef:$state,
                       I64Attr:$layer_index,
                       I64Attr:$num_k_heads,
                       I64Attr:$num_v_heads,
                       I64Attr:$head_k_dim,
                       I64Attr:$head_v_dim,
                       StrAttr:$mode);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def LLMCP_AttentionOp : LLMCP_Op<"attention"> {
  let summary = "GQA attention with KV cache, q/k norm, RoPE, and output gate";
  let arguments = (ins AnyTensor:$qg,
                       AnyTensor:$k,
                       AnyTensor:$v,
                       AnyMemRef:$q_norm,
                       AnyMemRef:$k_norm,
                       AnyTensor:$cos,
                       AnyTensor:$sin,
                       AnyMemRef:$kv_cache,
                       Index:$position,
                       Index:$cache_length,
                       I64Attr:$layer_index,
                       I64Attr:$num_heads,
                       I64Attr:$num_kv_heads,
                       I64Attr:$head_dim,
                       I64Attr:$rotary_dim,
                       StrAttr:$mode);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def LLMCP_GatedRmsNormOp : LLMCP_Op<"gated_rms_norm", [Pure]> {
  let summary = "Per-head RMSNorm followed by a SiLU gate";
  let arguments = (ins AnyTensor:$input,
                       AnyTensor:$gate,
                       AnyMemRef:$weight,
                       I64Attr:$group_size,
                       F64Attr:$eps);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif // LLMCP_OPS_TD
