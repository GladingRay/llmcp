// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @attention
func.func @attention(%qg: tensor<2x16xbf16>, %k: tensor<2x4xbf16>,
                     %v: tensor<2x4xbf16>, %qn: memref<4xbf16>,
                     %kn: memref<4xbf16>, %cos: tensor<2x2xf32>,
                     %sin: tensor<2x2xf32>,
                     %cache: memref<1x2x1x2x4xbf16>,
                     %pos: index, %cache_length: index)
    -> tensor<2x8xbf16> {
  // CHECK: memref.alloc
  // CHECK: scf.for
  // CHECK: math.rsqrt
  // CHECK: math.exp
  // CHECK: memref.store
  // CHECK: bufferization.to_tensor
  // CHECK-NOT: llmcp.attention
  %0 = llmcp.attention %qg, %k, %v, %qn, %kn, %cos, %sin, %cache,
      %pos, %cache_length
      {layer_index = 0 : i64, num_heads = 2 : i64, num_kv_heads = 1 : i64,
       head_dim = 4 : i64, rotary_dim = 2 : i64, mode = "prefill"} :
      (tensor<2x16xbf16>, tensor<2x4xbf16>, tensor<2x4xbf16>,
       memref<4xbf16>, memref<4xbf16>, tensor<2x2xf32>, tensor<2x2xf32>,
       memref<1x2x1x2x4xbf16>, index, index) -> tensor<2x8xbf16>
  return %0 : tensor<2x8xbf16>
}
