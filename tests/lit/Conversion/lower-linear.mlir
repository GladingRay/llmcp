// REQUIRES: bfdot
// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @linear
func.func @linear(%x: tensor<1x2560xbf16>, %w: memref<9216x2560xbf16>)
    -> tensor<1x9216xbf16> {
  // CHECK: bufferization.to_tensor
  // CHECK: scf.for
  // CHECK: vector.transfer_read
  // CHECK: llvm.call_intrinsic "llvm.aarch64.neon.bfdot"
  // CHECK: vector.reduction
  // CHECK: arith.truncf
  // CHECK: memref.store
  // CHECK-NOT: linalg.matmul
  // CHECK-NOT: llmcp.linear
  %0 = llmcp.linear %x, %w :
      tensor<1x2560xbf16>, memref<9216x2560xbf16> -> tensor<1x9216xbf16>
  return %0 : tensor<1x9216xbf16>
}

// CHECK-LABEL: func.func @linear_transposed
func.func @linear_transposed(%x: tensor<1x2560xbf16>,
                             %w: memref<2560x9216xbf16>)
    -> tensor<1x9216xbf16> {
  // CHECK: vector.transfer_read
  // CHECK: vector.fma
  // CHECK-NOT: linalg.matmul
  // CHECK-NOT: llmcp.linear
  %0 = llmcp.linear %x, %w {transpose_b = true} :
      tensor<1x2560xbf16>, memref<2560x9216xbf16> -> tensor<1x9216xbf16>
  return %0 : tensor<1x9216xbf16>
}
