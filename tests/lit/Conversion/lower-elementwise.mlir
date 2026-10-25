// RUN: %llmcp-opt %s --llmcp-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @add
func.func @add(%a: tensor<2x4xbf16>, %b: tensor<2x4xbf16>)
    -> tensor<2x4xbf16> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK-NOT: llmcp.add
  %0 = llmcp.add %a, %b :
      (tensor<2x4xbf16>, tensor<2x4xbf16>) -> tensor<2x4xbf16>
  return %0 : tensor<2x4xbf16>
}

// CHECK-LABEL: func.func @silu_mul
func.func @silu_mul(%gate: tensor<2x8xbf16>, %up: tensor<2x8xbf16>)
    -> tensor<2x8xbf16> {
  // CHECK: linalg.generic
  // CHECK: math.exp
  // CHECK: arith.divf
  // CHECK: arith.truncf
  // CHECK-NOT: llmcp.silu_mul
  %0 = llmcp.silu_mul %gate, %up :
      (tensor<2x8xbf16>, tensor<2x8xbf16>) -> tensor<2x8xbf16>
  return %0 : tensor<2x8xbf16>
}
