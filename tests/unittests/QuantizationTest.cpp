//===- QuantizationTest.cpp - Unit tests for Q8_0 quantization ------------===//
//
// Unit tests for Q8_0 block quantization and tensor sizing.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Model/Quantization.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace mlir::llmcp;

namespace {

std::vector<float> dequantize(llvm::ArrayRef<Q8_0Block> blocks,
                              int64_t count) {
  std::vector<float> values(static_cast<size_t>(count));
  dequantizeQ8_0(blocks, values);
  return values;
}

} // namespace

TEST(QuantizationTest, ParsesModes) {
  EXPECT_EQ(*parseQuantizationMode("none"), QuantizationMode::None);
  EXPECT_EQ(*parseQuantizationMode("q8_0"), QuantizationMode::Q8_0);
  EXPECT_EQ(stringifyQuantizationMode(QuantizationMode::Q8_0), "q8_0");
  EXPECT_FALSE(parseQuantizationMode("q4_0"));
}

TEST(QuantizationTest, Q8TensorSize) {
  EXPECT_EQ(*getQ8_0TensorSize({4, 64}), 4 * 2 * sizeof(Q8_0Block));
  EXPECT_FALSE(getQ8_0TensorSize({4, 63}));
  EXPECT_FALSE(getQ8_0TensorSize({4}));
}

TEST(QuantizationTest, RoundTrip) {
  std::vector<float> input(128);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = std::sin(static_cast<float>(i) * 0.1f) * 2.0f;

  std::vector<Q8_0Block> blocks(4);
  quantizeRowQ8_0(input, blocks);
  std::vector<float> output = dequantize(blocks, 128);

  float maxError = 0.0f;
  for (size_t i = 0; i < input.size(); ++i)
    maxError = std::max(maxError, std::fabs(input[i] - output[i]));
  EXPECT_LT(maxError, 0.02f);
}

TEST(QuantizationTest, SplitTensorLayout) {
  TensorInfo tensor;
  tensor.name = "test";
  tensor.dtype = "BF16";
  tensor.shape = {1, 64};
  tensor.dataSize = 64 * sizeof(uint16_t);

  std::vector<uint16_t> source(64, 0x3f80); // bf16 1.0
  std::vector<uint8_t> destination(64 + (64 / 32) * 2);
  ASSERT_FALSE(!!quantizeTensorQ8_0(
      tensor, reinterpret_cast<const uint8_t *>(source.data()),
      destination.data()));

  // Scales occupy the first 2 * (K/32) bytes.
  EXPECT_EQ(destination[0], 0x08);
  EXPECT_EQ(destination[1], 0x20);
  EXPECT_EQ(destination[2], 0x08);
  EXPECT_EQ(destination[3], 0x20);

  // Quantized values start after the scale array.
  constexpr size_t dataOffset = (64 / 32) * 2;
  for (size_t i = 0; i < 64; ++i)
    EXPECT_EQ(static_cast<int8_t>(destination[dataOffset + i]), 127);
}

TEST(QuantizationTest, ZeroBlock) {
  std::vector<float> input(32, 0.0f);
  Q8_0Block block;
  quantizeRowQ8_0(input, llvm::MutableArrayRef<Q8_0Block>(&block, 1));
  EXPECT_EQ(block.d, 0);
  for (int8_t value : block.qs)
    EXPECT_EQ(value, 0);
}
