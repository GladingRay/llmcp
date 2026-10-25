// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @linear
func.func @linear(%x: tensor<1x2560xbf16>, %w: memref<9216x2560xbf16>)
    -> tensor<1x9216xbf16> {
  // CHECK: bufferization.to_tensor
  // CHECK: linalg.fill
  // CHECK: linalg.matmul_transpose_b
  // CHECK: linalg.generic
  // CHECK: arith.truncf
  // CHECK-NOT: llmcp.linear
  %0 = llmcp.linear %x, %w :
      tensor<1x2560xbf16>, memref<9216x2560xbf16> -> tensor<1x9216xbf16>
  return %0 : tensor<1x9216xbf16>
}

// CHECK-LABEL: func.func @linear_transposed
func.func @linear_transposed(%x: tensor<1x2560xbf16>,
                             %w: memref<2560x9216xbf16>)
    -> tensor<1x9216xbf16> {
  // CHECK: linalg.fill
  // CHECK: linalg.matmul ins
  // CHECK-NOT: llmcp.linear
  %0 = llmcp.linear %x, %w {transpose_b = true} :
      tensor<1x2560xbf16>, memref<2560x9216xbf16> -> tensor<1x9216xbf16>
  return %0 : tensor<1x9216xbf16>
}
