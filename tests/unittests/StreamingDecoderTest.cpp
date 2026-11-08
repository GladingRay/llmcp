//===- StreamingDecoderTest.cpp - ByteLevel streaming tests ----------------===//
//
//===----------------------------------------------------------------------===//

#include "StreamingDecoder.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "tokenizers_cpp.h"
#include "gtest/gtest.h"

#include <random>

using namespace mlir::llmcp::runtime;

namespace {

/// Build the GPT-2 byte alphabet used by the Qwen ByteLevel decoder.
std::vector<std::string> makeByteAlphabet() {
  std::vector<std::string> alphabet;
  unsigned next = 256;
  for (unsigned byte = 0; byte < 256; ++byte) {
    bool direct = (byte >= 33 && byte <= 126) ||
                  (byte >= 161 && byte <= 172) || byte >= 174;
    unsigned codepoint = direct ? byte : next++;
    std::string token;
    if (codepoint < 128) {
      token.push_back(static_cast<char>(codepoint));
    } else {
      token.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
      token.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    alphabet.push_back(token);
  }
  return alphabet;
}

std::unique_ptr<tokenizers::Tokenizer> makeTokenizer() {
  auto alphabet = makeByteAlphabet();
  llvm::json::Object vocab;
  for (unsigned byte = 0; byte < 256; ++byte)
    vocab[alphabet[byte]] = byte;
  // Exercise tokens containing both complete text and a partial next character.
  vocab["A" + alphabet[0xe4]] = 256;
  vocab[alphabet[0xb8] + alphabet[0xad] + "B" + alphabet[0xf0]] = 257;
  vocab[alphabet[0x9f] + alphabet[0x98] + alphabet[0x80]] = 258;
  vocab["<|test|>"] = 259;
  std::string json;
  llvm::raw_string_ostream os(json);
  os << llvm::json::Value(std::move(vocab));
  return tokenizers::Tokenizer::FromBlobByteLevelBPE(json, "");
}

std::string stream(tokenizers::Tokenizer &tokenizer,
                   const std::vector<int32_t> &ids) {
  ByteLevelStreamingDecoder decoder(tokenizer);
  std::string text;
  for (int32_t id : ids) {
    std::string chunk = decoder.append(id);
    EXPECT_TRUE(llvm::json::isUTF8(chunk));
    text += chunk;
  }
  std::string tail = decoder.finish();
  EXPECT_TRUE(llvm::json::isUTF8(tail));
  text += tail;
  EXPECT_TRUE(decoder.finish().empty());
  return text;
}

} // namespace

TEST(StreamingDecoderTest, EmitsAsciiImmediatelyAndBuffersSplitCharacters) {
  auto tokenizer = makeTokenizer();
  ByteLevelStreamingDecoder decoder(*tokenizer);
  EXPECT_EQ(decoder.append('A'), "A");
  EXPECT_TRUE(decoder.append(0xe4).empty());
  EXPECT_TRUE(decoder.append(0xb8).empty());
  EXPECT_EQ(decoder.append(0xad), "中");
  EXPECT_TRUE(decoder.append(0xf0).empty());
  EXPECT_TRUE(decoder.append(0x9f).empty());
  EXPECT_TRUE(decoder.append(0x98).empty());
  EXPECT_EQ(decoder.append(0x80), "😀");
  EXPECT_EQ(decoder.append('B'), "B");
  EXPECT_TRUE(decoder.finish().empty());
}

TEST(StreamingDecoderTest, MatchesFullDecodeAtEveryTokenLimit) {
  auto tokenizer = makeTokenizer();
  for (const std::string text : {"中文你好，世界！", "🙂🚀🧑‍💻", "é\n A 中 😀 B",
                                 "literal � replacement", "ASCII\n\t text"}) {
    auto ids = tokenizer->Encode(text);
    ASSERT_EQ(tokenizer->Decode(ids), text);
    for (size_t count = 0; count <= ids.size(); ++count) {
      std::vector<int32_t> prefix(ids.begin(), ids.begin() + count);
      EXPECT_EQ(stream(*tokenizer, prefix), tokenizer->Decode(prefix));
    }
  }
}

TEST(StreamingDecoderTest, HandlesMixedTokensAndSpecialText) {
  auto tokenizer = makeTokenizer();
  std::vector<int32_t> ids{256, 257, 258, 259};
  EXPECT_EQ(stream(*tokenizer, ids), "A中B😀<|test|>");
  EXPECT_EQ(stream(*tokenizer, ids), tokenizer->Decode(ids));
}

TEST(StreamingDecoderTest, PreservesInvalidBytesAndFinalPartialCharacters) {
  auto tokenizer = makeTokenizer();
  std::mt19937 random(0);
  for (unsigned trial = 0; trial < 128; ++trial) {
    std::vector<int32_t> ids;
    for (unsigned i = 0; i < 32; ++i)
      ids.push_back(random() % 256);
    EXPECT_EQ(stream(*tokenizer, ids), tokenizer->Decode(ids));
  }
}
