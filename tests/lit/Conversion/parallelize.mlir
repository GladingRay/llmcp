// RUN: %llmcp-opt %s --llmcp-lower-to-linalg --llmcp-parallelize | %FileCheck %s

// CHECK-LABEL: func.func @linear
func.func @linear(%x: tensor<2x2560xbf16>, %w: memref<9216x2560xbf16>)
    -> tensor<2x9216xbf16> {
  // CHECK: scf.parallel
  // CHECK: scf.reduce
  // CHECK-NOT: llmcp.linear
  %0 = llmcp.linear %x, %w :
      tensor<2x2560xbf16>, memref<9216x2560xbf16> -> tensor<2x9216xbf16>
  return %0 : tensor<2x9216xbf16>
}
