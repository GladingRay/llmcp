// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @gated_delta_net
func.func @gated_delta_net(%conv: tensor<2x16xbf16>, %a: tensor<2x4xbf16>,
                           %b: tensor<2x4xbf16>, %alog: memref<4xf32>,
                           %dt: memref<4xbf16>,
                           %state: memref<1x4x2x2xf32>)
    -> tensor<2x8xbf16> {
  // CHECK: memref.alloc
  // CHECK: scf.for
  // CHECK: math.rsqrt
  // CHECK: math.log1p
  // CHECK: math.exp
  // CHECK: memref.store
  // CHECK: bufferization.to_tensor
  // CHECK-NOT: llmcp.gated_delta_net
  %0 = llmcp.gated_delta_net %conv, %a, %b, %alog, %dt, %state
      {layer_index = 0 : i64, num_k_heads = 2 : i64, num_v_heads = 4 : i64,
       head_k_dim = 2 : i64, head_v_dim = 2 : i64, mode = "prefill"} :
      (tensor<2x16xbf16>, tensor<2x4xbf16>, tensor<2x4xbf16>,
       memref<4xf32>, memref<4xbf16>, memref<1x4x2x2xf32>)
      -> tensor<2x8xbf16>
  return %0 : tensor<2x8xbf16>
}
