// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @rms_norm
func.func @rms_norm(%x: tensor<1x2560xbf16>, %w: memref<2560xbf16>)
    -> tensor<1x2560xbf16> {
  // CHECK: scf.parallel
  // CHECK: scf.for
  // CHECK: math.rsqrt
  // CHECK: arith.truncf
  // CHECK: scf.reduce
  // CHECK-NOT: linalg.generic
  // CHECK-NOT: llmcp.rms_norm
  %0 = llmcp.rms_norm %x, %w :
      tensor<1x2560xbf16>, memref<2560xbf16> -> tensor<1x2560xbf16>
  return %0 : tensor<1x2560xbf16>
}
