#include "llmcp/Model/Safetensors.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <string>

using namespace mlir::llmcp;

namespace {

const char *kHeader = R"({
  "__metadata__": {"format": "pt"},
  "a": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
  "b": {"dtype": "F32", "shape": [1], "data_offsets": [12, 16]}
})";

void expectError(llvm::Expected<TensorMap> tensors) {
  if (tensors) {
    ADD_FAILURE() << "expected safetensors header parsing to fail";
    return;
  }
  llvm::consumeError(tensors.takeError());
}

} // namespace

TEST(SafetensorsTest, ParsesHeaderAndOffsets) {
  const std::string header = kHeader;
  const uint64_t dataStart = 8 + header.size();
  const uint64_t fileSize = dataStart + 16;

  auto tensors =
      parseSafetensorsHeader(header, dataStart, fileSize);
  ASSERT_TRUE(!!tensors) << llvm::toString(tensors.takeError());
  ASSERT_EQ(tensors->size(), 2u);

  auto a = tensors->find("a");
  ASSERT_NE(a, tensors->end());
  EXPECT_EQ(a->second.dtype, "BF16");
  EXPECT_EQ(a->second.shape.size(), 2u);
  EXPECT_EQ(a->second.getNumElements(), 6u);
  EXPECT_EQ(a->second.dataOffset, dataStart);
  EXPECT_EQ(a->second.dataSize, 12u);

  auto b = tensors->find("b");
  ASSERT_NE(b, tensors->end());
  EXPECT_EQ(b->second.dataOffset, dataStart + 12);
  EXPECT_EQ(b->second.dataSize, 4u);
}

TEST(SafetensorsTest, RejectsSizeMismatch) {
  std::string header = kHeader;
  const std::string offsets = R"("data_offsets": [12, 16])";
  const size_t pos = header.find(offsets);
  ASSERT_NE(pos, std::string::npos);
  header.replace(pos, offsets.size(), R"("data_offsets": [12, 15])");

  const uint64_t dataStart = 8 + header.size();
  expectError(parseSafetensorsHeader(header, dataStart, dataStart + 16));
}

TEST(SafetensorsTest, RejectsUnknownDType) {
  std::string header = kHeader;
  const std::string dtype = R"("dtype": "F32")";
  const size_t pos = header.find(dtype);
  ASSERT_NE(pos, std::string::npos);
  header.replace(pos, dtype.size(), R"("dtype": "F8")");

  const uint64_t dataStart = 8 + header.size();
  expectError(parseSafetensorsHeader(header, dataStart, dataStart + 16));
}

TEST(SafetensorsTest, ReportsDTypeSizes) {
  auto bf16 = getSafetensorsDTypeSize("BF16");
  ASSERT_TRUE(!!bf16);
  EXPECT_EQ(*bf16, 2u);

  auto f32 = getSafetensorsDTypeSize("F32");
  ASSERT_TRUE(!!f32);
  EXPECT_EQ(*f32, 4u);

  auto bad = getSafetensorsDTypeSize("F8");
  EXPECT_FALSE(!!bad);
  llvm::consumeError(bad.takeError());
}
