// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @lm_head
func.func @lm_head(%h: tensor<2x8xbf16>, %table: memref<32x8xbf16>)
    -> tensor<2x32xf32> {
  // CHECK: bufferization.to_tensor
  // CHECK: linalg.matmul_transpose_b
  // CHECK-NOT: llmcp.lm_head
  %0 = llmcp.lm_head %h, %table :
      (tensor<2x8xbf16>, memref<32x8xbf16>) -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}
