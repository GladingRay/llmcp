//===- WeightLayout.h - Deterministic weight arena layout --------*- C++ -*-===//
//
// Computes byte offsets for the single packed weight arena consumed by the
// compiled kernels. The packer and the compiler share this implementation.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_MODEL_WEIGHTLAYOUT_H
#define LLMCP_MODEL_WEIGHTLAYOUT_H

#include "llmcp/Model/ModelConfig.h"
#include "llmcp/Model/Safetensors.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace llmcp {

/// One tensor placed in the weight arena.
struct WeightEntry {
  std::string name;
  uint64_t offset = 0;
  uint64_t size = 0;
  std::string dtype;
  llvm::SmallVector<int64_t, 4> shape;
};

/// Layout of the packed weight arena.
class WeightLayout {
public:
  /// Builds the layout for a Qwen3.5-style model.
  ///
  /// The current implementation requires a uniform layer-group pattern, for
  /// example `[linear, linear, linear, full]` repeated for every group. This
  /// keeps the group stride constant so the compiler can emit a loop over
  /// groups instead of unrolling all layers.
  static llvm::Expected<WeightLayout> build(const ModelConfig &config,
                                            const TensorMap &tensors);

  llvm::ArrayRef<WeightEntry> getEntries() const { return entries_; }

  std::optional<uint64_t> getOffset(llvm::StringRef name) const;

  uint64_t getTotalBytes() const { return totalBytes_; }
  uint64_t getGroupStride() const { return groupStride_; }
  uint64_t getHeaderBytes() const { return headerBytes_; }
  uint64_t getEmbeddingOffset() const { return embeddingOffset_; }
  uint64_t getFinalNormOffset() const { return finalNormOffset_; }
  int64_t getNumGroups() const { return numGroups_; }

private:
  explicit WeightLayout() = default;

  llvm::Error addTensor(llvm::StringRef name,
                        llvm::ArrayRef<int64_t> expectedShape,
                        const TensorMap &tensors, uint64_t &cursor);

  llvm::Error addLayerTensor(llvm::StringRef prefix, llvm::StringRef suffix,
                             llvm::ArrayRef<int64_t> expectedShape,
                             const TensorMap &tensors, uint64_t &cursor);

  std::vector<WeightEntry> entries_;
  llvm::StringMap<uint64_t> offsets_;
  uint64_t totalBytes_ = 0;
  uint64_t groupStride_ = 0;
  uint64_t headerBytes_ = 0;
  uint64_t embeddingOffset_ = 0;
  uint64_t finalNormOffset_ = 0;
  int64_t numGroups_ = 0;
};

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_MODEL_WEIGHTLAYOUT_H
