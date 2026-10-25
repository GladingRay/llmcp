// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @gated_rms_norm
func.func @gated_rms_norm(%x: tensor<2x16xbf16>, %gate: tensor<2x16xbf16>,
                          %w: memref<4xf32>) -> tensor<2x16xbf16> {
  // CHECK: tensor.expand_shape
  // CHECK: linalg.fill
  // CHECK: math.rsqrt
  // CHECK: math.exp
  // CHECK: tensor.collapse_shape
  // CHECK-NOT: llmcp.gated_rms_norm
  %0 = llmcp.gated_rms_norm %x, %gate, %w
      {group_size = 4 : i64, eps = 1.000000e-06 : f64} :
      (tensor<2x16xbf16>, tensor<2x16xbf16>, memref<4xf32>)
      -> tensor<2x16xbf16>
  return %0 : tensor<2x16xbf16>
}
