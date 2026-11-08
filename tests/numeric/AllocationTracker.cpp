//===- AllocationTracker.cpp - Check generated buffer lifetimes ------------===//
//
// Track allocations from generated IR without intercepting runtime libraries.
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace {
std::atomic<long> liveAllocations{0};

struct AllocationCheck {
  ~AllocationCheck() {
    long live = liveAllocations.load();
    if (live != 0) {
      std::fprintf(stderr, "generated code leaked %ld allocations\n", live);
      std::abort();
    }
  }
} allocationCheck;
} // namespace

extern "C" void *llmcp_test_malloc(size_t size) {
  void *ptr = std::malloc(size);
  if (ptr)
    ++liveAllocations;
  return ptr;
}

extern "C" void llmcp_test_free(void *ptr) {
  if (ptr)
    --liveAllocations;
  std::free(ptr);
}
