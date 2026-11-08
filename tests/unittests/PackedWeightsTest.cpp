//===- PackedWeightsTest.cpp - Weight container tests ---------------------===//
//
//===----------------------------------------------------------------------===//

#include "llmcp/Support/PackedWeights.h"
#include "TestModel.h"
#include "llmcp/Model/WeightPacker.h"
#include "llmcp/Runtime/AotRuntime.h"

#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstring>
#include <limits>

using namespace mlir::llmcp;
using namespace mlir::llmcp::runtime;

namespace {

const char *kManifest = R"({"total_bytes":128,"header_bytes":64,
  "quantization":"none","tensors":[
  {"name":"weight","offset":64,"size":2,"dtype":"BF16","shape":[1]}]})";

std::string makeFile(llvm::StringRef manifest = kManifest) {
  std::string file(128, '\0');
  file[64] = '\x80';
  file[65] = '\x3f';
  file += manifest;
  ::llmcp::PackedWeightsHeader{128, 128, manifest.size()}.write(
      reinterpret_cast<uint8_t *>(file.data()));
  return file;
}

class PackedWeightsTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(
        llvm::sys::fs::createUniqueDirectory("llmcp-weights-test", dir));
  }
  void TearDown() override {
    EXPECT_FALSE(llvm::sys::fs::remove_directories(dir));
  }
  std::string path(llvm::StringRef name) const {
    return (llvm::Twine(dir) + "/" + name).str();
  }
  void write(llvm::StringRef name, llvm::StringRef data) {
    std::error_code ec;
    llvm::raw_fd_ostream out(path(name), ec);
    ASSERT_FALSE(ec);
    out << data;
    out.close();
    ASSERT_FALSE(out.has_error());
  }
  void expectInvalid(llvm::StringRef file) {
    write("invalid.bin", file);
    Arena arena;
    auto error = arena.load(path("invalid.bin"));
    ASSERT_TRUE(!!error);
    llvm::consumeError(std::move(error));
  }
  llvm::SmallString<128> dir;
};

} // namespace

TEST_F(PackedWeightsTest, LoadsEmbeddedManifestAndReloadsTransactionally) {
  write("weights.bin", makeFile());
  Arena arena;
  auto error = arena.load(path("weights.bin"));
  ASSERT_FALSE(!!error) << llvm::toString(std::move(error));
  EXPECT_EQ(arena.size, 128u);
  EXPECT_EQ(arena.quantization, "none");
  ASSERT_EQ(arena.weights.size(), 1u);
  EXPECT_EQ(arena.get("weight").data, arena.base + 64);
  EXPECT_EQ(arena.get("weight").data[1], 0x3f);

  write("bad.bin", "not a packed weight file");
  error = arena.load(path("bad.bin"));
  ASSERT_TRUE(!!error);
  llvm::consumeError(std::move(error));
  EXPECT_EQ(arena.get("weight").data[1], 0x3f);

  std::string next = makeFile();
  next[65] = '\x40';
  write("next.bin", next);
  error = arena.load(path("next.bin"));
  ASSERT_FALSE(!!error) << llvm::toString(std::move(error));
  EXPECT_EQ(arena.get("weight").data[1], 0x40);
}

TEST_F(PackedWeightsTest, RejectsInvalidHeadersAndTruncation) {
  expectInvalid("");
  std::string file = makeFile();
  file[0] = 0;
  expectInvalid(file);
  file = makeFile();
  llvm::support::endian::write32le(file.data() + 8, 2);
  expectInvalid(file);
  file = makeFile();
  llvm::support::endian::write32le(file.data() + 12, 32);
  expectInvalid(file);
  file = makeFile();
  file.pop_back();
  expectInvalid(file);
  file = makeFile();
  llvm::support::endian::write64le(file.data() + 24,
                                   std::numeric_limits<uint64_t>::max());
  expectInvalid(file);
}

TEST_F(PackedWeightsTest, RejectsInvalidMetadataAndTensorBounds) {
  expectInvalid(makeFile("{"));
  expectInvalid(makeFile("[]"));
  for (auto replacement :
       {std::pair<llvm::StringRef, llvm::StringRef>{"\"BF16\"", "\"unknown\""},
        {"\"offset\":64", "\"offset\":128"},
        {"\"offset\":64", "\"offset\":-64"},
        {"\"size\":2", "\"size\":9223372036854775807"},
        {"\"shape\":[1]", "\"shape\":[9223372036854775807,8]"},
        {"\"shape\":[1]", "\"shape\":[0]"},
        {"\"dtype\":\"BF16\"", "\"other\":\"BF16\""},
        {"\"total_bytes\":128", "\"total_bytes\":129"}}) {
    std::string manifest = kManifest;
    manifest.replace(manifest.find(replacement.first.str()),
                     replacement.first.size(), replacement.second.str());
    expectInvalid(makeFile(manifest));
  }
}

TEST_F(PackedWeightsTest, PacksAndLoadsTinyCheckpointWithoutSidecar) {
  auto config = llmcp_test::makeTinyConfig();
  auto tensors = llmcp_test::makeTinyTensors();
  // Use a 32-wide hidden dimension to exercise real Q8 weight packing as well.
  config.text.hiddenSize = 32;
  config.text.intermediateSize = 32;
  for (auto &item : tensors) {
    auto name = item.getKey();
    auto &tensor = item.getValue();
    if (name.ends_with("embed_tokens.weight") || name.contains("mlp.")) {
      tensor.shape = {32, 32};
    } else if (tensor.shape.size() == 2) {
      if (name.ends_with("out_proj.weight") || name.ends_with("o_proj.weight"))
        tensor.shape[0] = 32;
      else
        tensor.shape[1] = 32;
    } else if (name.ends_with("layernorm.weight") ||
               name == "model.language_model.norm.weight") {
      tensor.shape = {32};
    }
    tensor.dataSize = tensor.getNumElements() * (tensor.dtype == "F32" ? 4 : 2);
  }
  llvm::json::Object headers, weightMap;
  std::string data;
  unsigned tensorIndex = 0;
  for (const auto &item : tensors) {
    const auto &tensor = item.getValue();
    llvm::json::Array shape;
    for (int64_t dim : tensor.shape)
      shape.push_back(dim);
    uint64_t begin = data.size();
    data.append(tensor.dataSize, '\0');
    float value = static_cast<float>(++tensorIndex) / 8.0f;
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    unsigned elementBytes = tensor.dtype == "F32" ? 4 : 2;
    for (uint64_t offset = begin; offset < data.size();
         offset += elementBytes) {
      if (elementBytes == 4)
        llvm::support::endian::write32le(data.data() + offset, bits);
      else
        llvm::support::endian::write16le(data.data() + offset, bits >> 16);
    }
    headers[item.getKey()] = llvm::json::Object{
        {"dtype", tensor.dtype},
        {"shape", std::move(shape)},
        {"data_offsets", llvm::json::Array{begin, data.size()}}};
    weightMap[item.getKey()] = "model.safetensors";
  }
  std::string header;
  llvm::raw_string_ostream(header) << llvm::json::Value(std::move(headers));
  std::string shard(8, '\0');
  llvm::support::endian::write64le(shard.data(), header.size());
  shard += header;
  shard += data;
  write("model.safetensors", shard);
  std::string index;
  llvm::raw_string_ostream(index) << llvm::json::Value(
      llvm::json::Object{{"weight_map", std::move(weightMap)}});
  write("model.safetensors.index.json", index);
  auto sources = readModelTensors(dir);
  ASSERT_TRUE(!!sources) << llvm::toString(sources.takeError());
  for (auto quantization : {QuantizationMode::None, QuantizationMode::Q8_0}) {
    auto layout = WeightLayout::build(config, *sources, quantization);
    ASSERT_TRUE(!!layout) << llvm::toString(layout.takeError());
    std::string output =
        quantization == QuantizationMode::None ? "bf16.bin" : "q8.bin";
    auto error = packWeights(dir, *layout, *sources, path(output));
    ASSERT_FALSE(!!error) << llvm::toString(std::move(error));
    EXPECT_FALSE(llvm::sys::fs::exists(path(output + ".json")));
    Arena arena;
    error = arena.load(path(output));
    ASSERT_FALSE(!!error) << llvm::toString(std::move(error));
    EXPECT_EQ(arena.size, layout->getTotalBytes());
    EXPECT_EQ(arena.weights.size(), layout->getEntries().size());
    EXPECT_EQ(arena.quantization, stringifyQuantizationMode(quantization));
    unsigned quantizedTensors = 0;
    for (const auto &entry : layout->getEntries()) {
      const auto &weight = arena.get(entry.name);
      const auto &source = sources->find(entry.name)->second;
      EXPECT_EQ(weight.data, arena.base + entry.offset);
      EXPECT_EQ(weight.dtype, entry.dtype);
      if (entry.quantized) {
        ++quantizedTensors;
        EXPECT_EQ(weight.dtype, "Q8_0");
        std::vector<uint8_t> expected(entry.size);
        error = quantizeTensorQ8_0(
            source,
            reinterpret_cast<const uint8_t *>(shard.data()) + source.dataOffset,
            expected.data());
        ASSERT_FALSE(!!error) << llvm::toString(std::move(error));
        EXPECT_EQ(std::memcmp(weight.data, expected.data(), entry.size), 0);
      } else {
        EXPECT_EQ(std::memcmp(weight.data, shard.data() + source.dataOffset,
                              entry.size),
                  0);
      }
    }
    if (quantization == QuantizationMode::Q8_0)
      EXPECT_GT(quantizedTensors, 0u);
  }
}
