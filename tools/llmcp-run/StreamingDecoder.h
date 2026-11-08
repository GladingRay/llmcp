//===- StreamingDecoder.h - ByteLevel streaming output ----------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_TOOLS_STREAMING_DECODER_H
#define LLMCP_TOOLS_STREAMING_DECODER_H

#include <cstdint>
#include <string>
#include <vector>

namespace tokenizers {
class Tokenizer;
}

namespace mlir::llmcp::runtime {

/// Stream a ByteLevel tokenizer without decoding partial UTF-8 bytes alone.
/// The tokenizer must outlive the decoder. Special tokens retain Decode's
/// behavior. Other decoder types may require additional token context.
class ByteLevelStreamingDecoder {
public:
  explicit ByteLevelStreamingDecoder(tokenizers::Tokenizer &tokenizer)
      : tokenizer(tokenizer) {}

  /// Return a complete chunk, or an empty string while bytes are incomplete.
  std::string append(int32_t token);

  /// Emit any remaining text, including Decode's replacement for truncation.
  std::string finish();

private:
  tokenizers::Tokenizer &tokenizer;
  std::vector<int32_t> pendingTokens;
};

} // namespace mlir::llmcp::runtime

#endif // LLMCP_TOOLS_STREAMING_DECODER_H
