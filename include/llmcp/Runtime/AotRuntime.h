//===- AotRuntime.h - Shared AOT arena and runtime --------------*- C++ -*-===//
//
// Shared arena, weight views, and AOT entry points for the runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_RUNTIME_AOT_RUNTIME_H
#define LLMCP_RUNTIME_AOT_RUNTIME_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "llmcp/Support/CacheLineAlignedVector.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir {
namespace llmcp {
namespace runtime {

struct Weight {
  const uint8_t *data = nullptr;
  std::vector<int64_t> shape;
  std::string dtype;
};

/// Memory-mapped packed weight arena and its JSON manifest.
struct Arena {
  int fd = -1;
  const uint8_t *base = nullptr;
  uint64_t size = 0;
  std::string quantization = "none";
  std::unordered_map<std::string, Weight> weights;

  Arena() = default;
  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;
  ~Arena();

  llvm::Error load(llvm::StringRef bin, llvm::StringRef manifest);
  const Weight &get(llvm::StringRef name) const;
};

/// Strided memref descriptor passed to the MLIR C interface.
template <typename T, int N> struct MemRef {
  T *base;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

std::pair<::llmcp::CacheLineAlignedVector<float>,
          ::llmcp::CacheLineAlignedVector<float>>
makeCosSin(int64_t pos, int64_t rotaryDim, float theta);

/// Runtime state for the AOT `llmcp_model_aot` entry point.
class AotRuntime {
public:
  using Fn = void (*)(void *, void *, void *, void *, void *, void *, void *,
                      void *, void *, void *);

  AotRuntime() = default;
  AotRuntime(const AotRuntime &) = delete;
  AotRuntime &operator=(const AotRuntime &) = delete;
  ~AotRuntime();

  bool init(const std::string &path, int64_t maxSequence);
  bool forward(const Arena &arena, const std::vector<int64_t> &tokens,
               int64_t startPos, int64_t cacheLength,
               ::llmcp::CacheLineAlignedVector<float> &logits);

private:
  void *handle = nullptr;
  Fn fn = nullptr;
  int64_t capacity = 0;
  ::llmcp::CacheLineAlignedVector<uint16_t> kv;
  ::llmcp::CacheLineAlignedVector<float> delta;
  ::llmcp::CacheLineAlignedVector<uint16_t> conv;
};

} // namespace runtime
} // namespace llmcp
} // namespace mlir

#endif // LLMCP_RUNTIME_AOT_RUNTIME_H
