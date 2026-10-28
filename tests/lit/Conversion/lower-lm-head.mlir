// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @lm_head
func.func @lm_head(%h: tensor<2x64xbf16>, %table: memref<64x64xbf16>)
    -> tensor<2x64xf32> {
  // CHECK: bufferization.to_tensor
  // CHECK: vector.transfer_read
  // CHECK: llvm.call_intrinsic "llvm.aarch64.neon.bfdot"
  // CHECK: vector.reduction
  // CHECK: memref.store
  // CHECK-NOT: linalg.matmul
  // CHECK-NOT: llmcp.lm_head
  %0 = llmcp.lm_head %h, %table :
      (tensor<2x64xbf16>, memref<64x64xbf16>) -> tensor<2x64xf32>
  return %0 : tensor<2x64xf32>
}
