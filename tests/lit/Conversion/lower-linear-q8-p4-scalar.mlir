// REQUIRES: no-sdot
// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @linear_q8_p4_scalar
func.func @linear_q8_p4_scalar(%x: tensor<2x64xbf16>,
                               %w: memref<4352xi8>) -> tensor<2x64xbf16> {
  // CHECK: scf.for
  // CHECK: math.roundeven
  // CHECK: arith.fptosi
  // CHECK: memref.load %{{.*}} : memref<4352xi8>
  // CHECK: arith.muli
  // CHECK: memref.store
  // CHECK-NOT: llmcp.linear
  %0 = llmcp.linear %x, %w {weight_format = "q8_0_p4"} :
      tensor<2x64xbf16>, memref<4352xi8> -> tensor<2x64xbf16>
  return %0 : tensor<2x64xbf16>
}
