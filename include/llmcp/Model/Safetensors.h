//===- Safetensors.h - Read safetensors headers -----------------*- C++ -*-===//
//
// Reads safetensors headers without loading tensor data.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_MODEL_SAFETENSORS_H
#define LLMCP_MODEL_SAFETENSORS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace llmcp {

/// Metadata for one tensor inside a safetensors shard.
struct TensorInfo {
  std::string name;
  std::string dtype;
  llvm::SmallVector<int64_t, 4> shape;
  uint64_t dataOffset = 0;
  uint64_t dataSize = 0;

  uint64_t getNumElements() const;
};

using TensorMap = llvm::StringMap<TensorInfo>;

/// Mapping from tensor names to shard filenames.
struct SafetensorsIndex {
  uint64_t totalSize = 0;
  llvm::StringMap<std::string> weightMap;
};

/// Returns the element size in bytes for a safetensors dtype.
llvm::Expected<uint64_t> getSafetensorsDTypeSize(llvm::StringRef dtype);

/// Parses a safetensors JSON header.
///
/// `dataStart` is the absolute file offset of the first data byte, and
/// `fileSize` is the total file size in bytes.
llvm::Expected<TensorMap> parseSafetensorsHeader(llvm::StringRef json,
                                                 uint64_t dataStart,
                                                 uint64_t fileSize);

/// Reads the header of one safetensors shard without reading tensor data.
llvm::Expected<TensorMap> readSafetensorsFile(llvm::StringRef path);

/// Reads `model.safetensors.index.json`.
llvm::Expected<SafetensorsIndex> readSafetensorsIndex(llvm::StringRef path);

/// Reads all shard headers of a model directory into one tensor map.
llvm::Expected<TensorMap> readModelTensors(llvm::StringRef modelDir);

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_MODEL_SAFETENSORS_H
