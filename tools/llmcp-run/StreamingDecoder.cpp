//===- StreamingDecoder.cpp - ByteLevel streaming output ------------------===//
//
//===----------------------------------------------------------------------===//

#include "StreamingDecoder.h"

#include "tokenizers_cpp.h"

using namespace mlir::llmcp::runtime;

std::string ByteLevelStreamingDecoder::append(int32_t token) {
  pendingTokens.push_back(token);
  std::string text = tokenizer.Decode(pendingTokens);
  // ByteLevel uses lossy UTF-8 decoding. A trailing replacement character may
  // represent bytes completed by the next token, so keep their token IDs.
  // A literal replacement character is delayed as well, never discarded.
  if (text.size() >= 3 && text.compare(text.size() - 3, 3, "\xef\xbf\xbd") == 0)
    return {};
  pendingTokens.clear();
  return text;
}

std::string ByteLevelStreamingDecoder::finish() {
  if (pendingTokens.empty())
    return {};
  std::string text = tokenizer.Decode(pendingTokens);
  pendingTokens.clear();
  return text;
}
