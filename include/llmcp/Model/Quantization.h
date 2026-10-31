//===- Quantization.h - Weight quantization formats ---------------------===//
//
// Q8_0 block format and helpers used by llmcp-pack.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_MODEL_QUANTIZATION_H
#define LLMCP_MODEL_QUANTIZATION_H

#include "llmcp/Model/Safetensors.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace mlir {
namespace llmcp {

enum class QuantizationMode { None, Q8_0 };

llvm::Expected<QuantizationMode>
parseQuantizationMode(llvm::StringRef name);

llvm::StringRef stringifyQuantizationMode(QuantizationMode mode);

/// Number of weights in one Q8_0 block.
inline constexpr int64_t kQ8_0BlockSize = 32;

/// llama.cpp-compatible Q8_0 block: fp16 scale followed by 32 int8 values.
struct Q8_0Block {
  uint16_t d;
  int8_t qs[kQ8_0BlockSize];
};
static_assert(sizeof(Q8_0Block) == 34, "Q8_0 block must be 34 bytes");

/// Returns the packed byte size for a [rows, cols] Q8_0 tensor.
llvm::Expected<uint64_t> getQ8_0TensorSize(llvm::ArrayRef<int64_t> shape);

/// Quantizes a dense F32/BF16 tensor into Q8_0 blocks.
llvm::Error quantizeTensorQ8_0(const TensorInfo &tensor,
                               const uint8_t *source, uint8_t *destination);

/// Quantizes one row of float values into Q8_0 blocks.
void quantizeRowQ8_0(llvm::ArrayRef<float> input,
                     llvm::MutableArrayRef<Q8_0Block> output);

/// Dequantizes Q8_0 blocks into float values.
void dequantizeQ8_0(llvm::ArrayRef<Q8_0Block> input,
                    llvm::MutableArrayRef<float> output);

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_MODEL_QUANTIZATION_H
