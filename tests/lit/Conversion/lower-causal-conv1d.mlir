// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @causal_conv1d
func.func @causal_conv1d(%x: tensor<3x4xbf16>, %w: memref<4x1x3xbf16>,
                         %s: memref<1x2x4xbf16>) -> tensor<3x4xbf16> {
  // CHECK: memref.alloc
  // CHECK: scf.for
  // CHECK: scf.if
  // CHECK: memref.load
  // CHECK: math.exp
  // CHECK: memref.store
  // CHECK: bufferization.to_tensor
  // CHECK-NOT: llmcp.causal_conv1d
  %0 = llmcp.causal_conv1d %x, %w, %s
      {layer_index = 0 : i64, kernel_size = 3 : i64, activation = "silu"} :
      (tensor<3x4xbf16>, memref<4x1x3xbf16>, memref<1x2x4xbf16>)
      -> tensor<3x4xbf16>
  return %0 : tensor<3x4xbf16>
}
