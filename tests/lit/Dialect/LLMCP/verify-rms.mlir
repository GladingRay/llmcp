// RUN: %llmcp-opt %s -verify-diagnostics

func.func @bad_rms(%x: tensor<1x2560xbf16>, %w: memref<9216x2560xbf16>)
    -> tensor<1x2560xbf16> {
  // expected-error @+1 {{weight must be a rank-1 memref}}
  %0 = llmcp.rms_norm %x, %w :
      tensor<1x2560xbf16>, memref<9216x2560xbf16> -> tensor<1x2560xbf16>
  return %0 : tensor<1x2560xbf16>
}
