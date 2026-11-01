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

TEST(QuantizationTest, ZeroBlock) {
  std::vector<float> input(32, 0.0f);
  Q8_0Block block;
  quantizeRowQ8_0(input, llvm::MutableArrayRef<Q8_0Block>(&block, 1));
  EXPECT_EQ(block.d, 0);
  for (int8_t value : block.qs)
    EXPECT_EQ(value, 0);
}
