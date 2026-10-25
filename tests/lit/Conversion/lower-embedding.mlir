// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @embedding
func.func @embedding(%tokens: tensor<2xindex>, %table: memref<32x8xbf16>)
    -> tensor<2x8xbf16> {
  // CHECK: bufferization.to_tensor
  // CHECK: linalg.generic
  // CHECK: tensor.extract
  // CHECK-NOT: llmcp.embedding
  %0 = llmcp.embedding %tokens, %table :
      (tensor<2xindex>, memref<32x8xbf16>) -> tensor<2x8xbf16>
  return %0 : tensor<2x8xbf16>
}
