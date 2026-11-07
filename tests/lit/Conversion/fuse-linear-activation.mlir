// RUN: %llmcp-opt %s --llmcp-fuse-linear-activations | %FileCheck %s

// CHECK-LABEL: func.func @fuse_silu_mul
// CHECK: llmcp.linear {{.*}} {activation = "silu"}
// CHECK: llmcp.mul
// CHECK-NOT: llmcp.silu_mul
func.func @fuse_silu_mul(%x: tensor<2x4xbf16>, %up: tensor<2x4xbf16>,
                         %w: memref<4x4xbf16>) -> tensor<2x4xbf16> {
  %gate = llmcp.linear %x, %w :
      tensor<2x4xbf16>, memref<4x4xbf16> -> tensor<2x4xbf16>
  %out = llmcp.silu_mul %gate, %up :
      (tensor<2x4xbf16>, tensor<2x4xbf16>) -> tensor<2x4xbf16>
  return %out : tensor<2x4xbf16>
}

// CHECK-LABEL: func.func @fuse_gated_rms_norm
// CHECK: llmcp.linear {{.*}} {activation = "silu"}
// CHECK: llmcp.gated_rms_norm {{.*}} gate_activated = true
// CHECK-NOT: llmcp.silu_mul
func.func @fuse_gated_rms_norm(%x: tensor<2x4xbf16>, %gate: tensor<2x4xbf16>,
                               %w: memref<4x4xbf16>,
                               %gw: memref<4xf32>) -> tensor<2x4xbf16> {
  %g = llmcp.linear %x, %w :
      tensor<2x4xbf16>, memref<4x4xbf16> -> tensor<2x4xbf16>
  %out = llmcp.gated_rms_norm %x, %g, %gw
      {group_size = 4 : i64, eps = 1.000000e-06 : f64} :
      (tensor<2x4xbf16>, tensor<2x4xbf16>, memref<4xf32>)
      -> tensor<2x4xbf16>
  return %out : tensor<2x4xbf16>
}
