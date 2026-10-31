//===- Quantization.cpp - Q8_0 quantization helpers ---------------------===//

#include "llmcp/Model/Quantization.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace mlir::llmcp;
using namespace llvm;

namespace {

float bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t floatToHalf(float value) {
  APFloat f(value);
  bool losesInfo = false;
  f.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &losesInfo);
  return static_cast<uint16_t>(f.bitcastToAPInt().getZExtValue());
}

float halfToFloat(uint16_t value) {
  APFloat f(APFloat::IEEEhalf(), APInt(16, value));
  bool losesInfo = false;
  f.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  return f.convertToFloat();
}

float loadFloat(const uint8_t *data, int64_t index, StringRef dtype) {
  if (dtype == "F32") {
    float value;
    std::memcpy(&value, data + index * sizeof(float), sizeof(value));
    return value;
  }
  uint16_t value;
  std::memcpy(&value, data + index * sizeof(uint16_t), sizeof(value));
  return bf16ToFloat(value);
}

} // namespace

Expected<QuantizationMode>
mlir::llmcp::parseQuantizationMode(StringRef name) {
  if (name.empty() || name == "none")
    return QuantizationMode::None;
  if (name == "q8_0")
    return QuantizationMode::Q8_0;
  return createStringError(inconvertibleErrorCode(),
                           "unsupported quantization mode: " + name);
}

StringRef mlir::llmcp::stringifyQuantizationMode(QuantizationMode mode) {
  switch (mode) {
  case QuantizationMode::None:
    return "none";
  case QuantizationMode::Q8_0:
    return "q8_0";
  }
  return "unknown";
}

Expected<uint64_t>
mlir::llmcp::getQ8_0TensorSize(ArrayRef<int64_t> shape) {
  if (shape.size() != 2)
    return createStringError(inconvertibleErrorCode(),
                             "Q8_0 requires a rank-2 tensor");
  if (shape[1] <= 0 || shape[1] % kQ8_0BlockSize != 0)
    return createStringError(inconvertibleErrorCode(),
                             "Q8_0 requires the last dimension to be a "
                             "multiple of 32");
  return static_cast<uint64_t>(shape[0]) * (shape[1] / kQ8_0BlockSize) *
         sizeof(Q8_0Block);
}

void mlir::llmcp::quantizeRowQ8_0(ArrayRef<float> input,
                                  MutableArrayRef<Q8_0Block> output) {
  for (size_t block = 0; block < output.size(); ++block) {
    ArrayRef<float> values = input.slice(block * kQ8_0BlockSize,
                                         kQ8_0BlockSize);
    float amax = 0.0f;
    for (float value : values)
      amax = std::max(amax, std::fabs(value));

    Q8_0Block &out = output[block];
    if (amax == 0.0f) {
      out.d = 0;
      std::memset(out.qs, 0, sizeof(out.qs));
      continue;
    }

    float scale = amax / 127.0f;
    out.d = floatToHalf(scale);
    for (int64_t i = 0; i < kQ8_0BlockSize; ++i) {
      long quantized = std::lround(values[i] / scale);
      quantized = std::clamp(quantized, -127L, 127L);
      out.qs[i] = static_cast<int8_t>(quantized);
    }
  }
}

Error mlir::llmcp::quantizeTensorQ8_0(const TensorInfo &tensor,
                                     const uint8_t *source,
                                     uint8_t *destination) {
  if (tensor.shape.size() != 2)
    return createStringError(inconvertibleErrorCode(),
                             "Q8_0 requires a rank-2 tensor");
  if (tensor.dtype != "BF16" && tensor.dtype != "F32")
    return createStringError(inconvertibleErrorCode(),
                             "Q8_0 source must be BF16 or F32");
  auto sizeOrErr = getQ8_0TensorSize(tensor.shape);
  if (!sizeOrErr)
    return sizeOrErr.takeError();

  const int64_t rows = tensor.shape[0];
  const int64_t cols = tensor.shape[1];
  const int64_t blocks = cols / kQ8_0BlockSize;
  for (int64_t row = 0; row < rows; ++row) {
    std::vector<float> values(static_cast<size_t>(cols));
    for (int64_t col = 0; col < cols; ++col)
      values[col] = loadFloat(source, row * cols + col, tensor.dtype);
    auto *out = reinterpret_cast<Q8_0Block *>(
        destination + static_cast<uint64_t>(row * blocks) * sizeof(Q8_0Block));
    quantizeRowQ8_0(values, MutableArrayRef<Q8_0Block>(out, blocks));
  }
  return Error::success();
}

void mlir::llmcp::dequantizeQ8_0(ArrayRef<Q8_0Block> input,
                                 MutableArrayRef<float> output) {
  for (size_t block = 0; block < input.size(); ++block) {
    float scale = halfToFloat(input[block].d);
    for (int64_t i = 0; i < kQ8_0BlockSize; ++i)
      output[block * kQ8_0BlockSize + i] = scale * input[block].qs[i];
  }
}
