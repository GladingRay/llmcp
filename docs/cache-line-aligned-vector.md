# CacheLineAlignedVector

## 1. 目的

本机 `hw.cachelinesize = 128`，因此 tensor 的堆分配按 128 字节对齐：

- 避免 tensor 起始地址跨 cache line；
- 让 bfdot / NEON 的连续访问更容易落在对齐的 cache line；
- 为后续 cache blocking、prefetch 和 SIMD 提供稳定前提。

## 2. 实现

头文件：

- [include/llmcp/Support/CacheLineAlignedVector.h](include/llmcp/Support/CacheLineAlignedVector.h)

核心：

```cpp
inline constexpr std::size_t kCacheLineSize = 128;

template <typename T> class CacheLineAlignedAllocator {
  T *allocate(std::size_t n) {
    return static_cast<T *>(
        ::operator new(n * sizeof(T),
                       std::align_val_t(kCacheLineSize)));
  }

  void deallocate(T *ptr, std::size_t) noexcept {
    ::operator delete(ptr, std::align_val_t(kCacheLineSize));
  }
};

template <typename T>
using CacheLineAlignedVector =
    std::vector<T, CacheLineAlignedAllocator<T>>;
```

它是 `std::vector` 的 aligned-allocator 版本，API 不变。

## 3. 使用范围

已替换 tools 中存储 tensor 数据的 vector：

- `llmcp-run`
  - `Vec`
  - matmul 的 `flatX` / `flatY` / `block`
  - logits / golden
- `llmcp-benchmark`
  - logits / lastLogits
- `llmcp-matmul-benchmark`
  - A / B / C
- `AotRuntime`
  - KV cache
  - DeltaNet state
  - conv state
  - cos/sin
  - logits

token id、metadata、命令行参数等非 tensor 数据仍使用普通 `std::vector`。

## 4. 测试

[tests/unittests/CacheLineAlignedVectorTest.cpp](tests/unittests/CacheLineAlignedVectorTest.cpp)

覆盖：

- 初始分配 128 字节对齐；
- `push_back` 扩容后仍对齐；
- 空 vector。
