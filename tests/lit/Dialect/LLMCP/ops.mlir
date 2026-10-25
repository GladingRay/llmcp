// RUN: %llmcp-opt %s | %FileCheck %s

// CHECK-LABEL: func.func @linear
func.func @linear(%x: tensor<1x2560xbf16>, %w: memref<9216x2560xbf16>)
    -> tensor<1x9216xbf16> {
  // CHECK: llmcp.linear
  %0 = llmcp.linear %x, %w :
      tensor<1x2560xbf16>, memref<9216x2560xbf16> -> tensor<1x9216xbf16>
  return %0 : tensor<1x9216xbf16>
}

// CHECK-LABEL: func.func @rms_norm
func.func @rms_norm(%x: tensor<1x2560xbf16>, %w: memref<2560xbf16>)
    -> tensor<1x2560xbf16> {
  // CHECK: llmcp.rms_norm
  %0 = llmcp.rms_norm %x, %w :
      tensor<1x2560xbf16>, memref<2560xbf16> -> tensor<1x2560xbf16>
  return %0 : tensor<1x2560xbf16>
}
