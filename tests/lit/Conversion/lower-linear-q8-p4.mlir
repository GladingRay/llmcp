// REQUIRES: sdot
// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @linear_q8_p4_prefill
func.func @linear_q8_p4_prefill(%x: tensor<8x64xbf16>,
                                %w: memref<4352xi8>) -> tensor<8x64xbf16> {
  // CHECK: memref.alloc
  // CHECK: arith.fptosi
  // CHECK: scf.for
  // CHECK: llvm.call_intrinsic "llvm.aarch64.neon.sdot"
  // CHECK: math.fma
  // CHECK: vector.store
  // CHECK-NOT: llmcp.linear
  %0 = llmcp.linear %x, %w {weight_format = "q8_0_p4"} :
      tensor<8x64xbf16>, memref<4352xi8> -> tensor<8x64xbf16>
  return %0 : tensor<8x64xbf16>
}

// CHECK-LABEL: func.func @linear_q8_p4_decode
func.func @linear_q8_p4_decode(%x: tensor<1x64xbf16>,
                               %w: memref<4352xi8>) -> tensor<1x64xbf16>
    attributes {llmcp.execution_mode = "decode"} {
  // CHECK: llvm.call_intrinsic "llvm.aarch64.neon.sdot"
  // CHECK: math.fma
  // CHECK: vector.store
  // CHECK-NOT: llmcp.linear
  %0 = llmcp.linear %x, %w {weight_format = "q8_0_p4"} :
      tensor<1x64xbf16>, memref<4352xi8> -> tensor<1x64xbf16>
  return %0 : tensor<1x64xbf16>
}
