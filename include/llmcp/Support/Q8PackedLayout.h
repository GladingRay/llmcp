//===- Q8PackedLayout.h - Packed Q8_0 layout --------------------*- C++ -*-===//
//
// Four output columns share one SDOT lane group.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_SUPPORT_Q8PACKEDLAYOUT_H
#define LLMCP_SUPPORT_Q8PACKEDLAYOUT_H

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace mlir {
namespace llmcp {

/// Number of columns whose dot products share one SDOT lane group.
inline constexpr int64_t kQ8PackedLanes = 4;

/// Number of weights in one Q8_0 block.
inline constexpr int64_t kQ8PackedBlockSize = 32;

/// Bytes of one contiguous four-byte activation word per lane.
inline constexpr int64_t kQ8PackedChunkBytes = 16;

/// Four fp16 scales, one per column of the lane group.
inline constexpr int64_t kQ8PackedScaleBytes = kQ8PackedLanes * 2;

/// Four columns times one 32-element block.
inline constexpr int64_t kQ8PackedDataBytes =
    kQ8PackedLanes * kQ8PackedBlockSize;

/// One lane group of one quantization block.
inline constexpr int64_t kQ8PackedBlockBytes =
    kQ8PackedScaleBytes + kQ8PackedDataBytes;

/// Returns true when rows can be grouped into pairs of four-column lane groups.
inline bool isQ8PackedShape(llvm::ArrayRef<int64_t> shape) {
  return shape.size() == 2 && shape[0] > 0 && shape[1] > 0 &&
         shape[0] % (2 * kQ8PackedLanes) == 0 &&
         shape[1] % kQ8PackedBlockSize == 0;
}

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_SUPPORT_Q8PACKEDLAYOUT_H
