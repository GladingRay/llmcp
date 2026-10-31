//===- CacheLineAlignedVector.h - Cache-line aligned vector --------------===//
//
// A std::vector whose heap allocation starts on a cache-line boundary.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_SUPPORT_CACHE_LINE_ALIGNED_VECTOR_H
#define LLMCP_SUPPORT_CACHE_LINE_ALIGNED_VECTOR_H

#include <cstddef>
#include <new>
#include <vector>

namespace llmcp {

/// Apple Silicon exposes a 128-byte cache line through hw.cachelinesize.
/// 128 is also a safe over-alignment on other CPUs.
inline constexpr std::size_t kCacheLineSize = 128;

template <typename T> class CacheLineAlignedAllocator {
public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  CacheLineAlignedAllocator() = default;
  template <typename U>
  CacheLineAlignedAllocator(const CacheLineAlignedAllocator<U> &) noexcept {}

  T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    std::size_t bytes = n * sizeof(T);
    return static_cast<T *>(
        ::operator new(bytes, std::align_val_t(kCacheLineSize)));
  }

  void deallocate(T *ptr, std::size_t) noexcept {
    if (ptr)
      ::operator delete(ptr, std::align_val_t(kCacheLineSize));
  }

  template <typename U> struct rebind {
    using other = CacheLineAlignedAllocator<U>;
  };
};

template <typename T, typename U>
bool operator==(const CacheLineAlignedAllocator<T> &,
                const CacheLineAlignedAllocator<U> &) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(const CacheLineAlignedAllocator<T> &,
                const CacheLineAlignedAllocator<U> &) noexcept {
  return false;
}

/// Plain std::vector storage.
template <typename T>
using CacheLineAlignedVector = std::vector<T>;

} // namespace llmcp

#endif // LLMCP_SUPPORT_CACHE_LINE_ALIGNED_VECTOR_H
