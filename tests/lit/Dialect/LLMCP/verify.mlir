// RUN: %llmcp-opt %s -verify-diagnostics

func.func @bad_linear(%x: tensor<1x2560xbf16>, %w: memref<9216x128xbf16>)
    -> tensor<1x9216xbf16> {
  // expected-error @+1 {{input and weight inner dimensions must match}}
  %0 = llmcp.linear %x, %w :
      tensor<1x2560xbf16>, memref<9216x128xbf16> -> tensor<1x9216xbf16>
  return %0 : tensor<1x9216xbf16>
}
