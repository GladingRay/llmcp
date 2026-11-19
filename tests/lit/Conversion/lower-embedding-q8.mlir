// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @embedding_q8
func.func @embedding_q8(%tokens: tensor<2xindex>, %table: memref<32x68xi8>)
    -> tensor<2x64xbf16> {
  // CHECK: memref.alloc
  // CHECK: scf.for
  // CHECK: memref.load %{{.*}}[{{.*}}, {{.*}}] : memref<32x68xi8>
  // CHECK: arith.bitcast %{{.*}} : i16 to f16
  // CHECK: arith.truncf
  // CHECK: memref.store
  // CHECK-NOT: llmcp.embedding
  %0 = llmcp.embedding %tokens, %table {weight_format = "q8_0"} :
      (tensor<2xindex>, memref<32x68xi8>) -> tensor<2x64xbf16>
  return %0 : tensor<2x64xbf16>
}
