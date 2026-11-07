//===- cpu_microbench.cpp - CPU ISA and bandwidth microbenchmark ----------===//
//
// Measures scalar, NEON, FP16, BF16, and int8 throughput on the local CPU.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/sysctl.h>
#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_bf16.h>
#include <arm_fp16.h>
#include <arm_neon.h>
#endif

using namespace llvm;

namespace {

cl::opt<double> secondsPerBench("seconds",
                                cl::desc("Seconds per microbenchmark"),
                                cl::init(1.0));
cl::opt<int64_t> bufferMB("buffer-mb", cl::desc("Bandwidth buffer in MiB"),
                          cl::init(256));
cl::opt<int64_t> threads("threads", cl::desc("Bandwidth thread count"),
                         cl::init(0));

int sysctlInt(const char *name) {
  int value = 0;
  size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0)
    return 0;
  return value;
}

std::string sysctlString(const char *name) {
  char buffer[256] = {0};
  size_t size = sizeof(buffer);
  if (sysctlbyname(name, buffer, &size, nullptr, 0) != 0)
    return "";
  return buffer;
}

uint64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

template <typename Fn>
double measure(const char *name, Fn fn, uint64_t &operations) {
  fn(1024); // warmup
  auto begin = std::chrono::steady_clock::now();
  operations = 0;
  do {
    operations += fn(100000);
  } while (
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count() < secondsPerBench);
  double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
  (void)name;
  return elapsed;
}

float scalarFma(uint64_t iterations) {
  float a = 1.000001f, b = 1.000002f;
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint64_t i = 0; i < iterations; ++i) {
    c0 = c0 * a + b;
    c1 = c1 * a + b;
    c2 = c2 * a + b;
    c3 = c3 * a + b;
  }
  volatile float sink = c0 + c1 + c2 + c3;
  return sink;
}

#if defined(__aarch64__)
float neonF32(uint64_t iterations) {
  float32x4_t a = vdupq_n_f32(1.000001f);
  float32x4_t b = vdupq_n_f32(1.000002f);
  float32x4_t c0 = vdupq_n_f32(0), c1 = c0, c2 = c0, c3 = c0;
  float32x4_t c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  for (uint64_t i = 0; i < iterations; ++i) {
    c0 = vfmaq_f32(c0, a, b);
    c1 = vfmaq_f32(c1, a, b);
    c2 = vfmaq_f32(c2, a, b);
    c3 = vfmaq_f32(c3, a, b);
    c4 = vfmaq_f32(c4, a, b);
    c5 = vfmaq_f32(c5, a, b);
    c6 = vfmaq_f32(c6, a, b);
    c7 = vfmaq_f32(c7, a, b);
  }
  asm volatile(""
               : "+w"(c0), "+w"(c1), "+w"(c2), "+w"(c3), "+w"(c4), "+w"(c5),
                 "+w"(c6), "+w"(c7));
  return vgetq_lane_f32(vaddq_f32(vaddq_f32(c0, c1), vaddq_f32(c2, c3)), 0);
}

float neonF16(uint64_t iterations) {
  float16x8_t a = vdupq_n_f16(1.0001f);
  float16x8_t b = vdupq_n_f16(1.0002f);
  float16x8_t c0 = vdupq_n_f16(0), c1 = c0, c2 = c0, c3 = c0;
  float16x8_t c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  for (uint64_t i = 0; i < iterations; ++i) {
    c0 = vfmaq_f16(c0, a, b);
    c1 = vfmaq_f16(c1, a, b);
    c2 = vfmaq_f16(c2, a, b);
    c3 = vfmaq_f16(c3, a, b);
    c4 = vfmaq_f16(c4, a, b);
    c5 = vfmaq_f16(c5, a, b);
    c6 = vfmaq_f16(c6, a, b);
    c7 = vfmaq_f16(c7, a, b);
  }
  asm volatile(""
               : "+w"(c0), "+w"(c1), "+w"(c2), "+w"(c3), "+w"(c4), "+w"(c5),
                 "+w"(c6), "+w"(c7));
  return vgetq_lane_f16(vaddq_f16(vaddq_f16(c0, c1), vaddq_f16(c2, c3)), 0);
}

#if defined(__aarch64__) && defined(LLMCP_ENABLE_BFDOT)
float neonBF16(uint64_t iterations) {
  bfloat16x8_t a = vdupq_n_bf16(1.0f);
  bfloat16x8_t b = vdupq_n_bf16(1.0f);
  float32x4_t c0 = vdupq_n_f32(0), c1 = c0, c2 = c0, c3 = c0;
  float32x4_t c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  for (uint64_t i = 0; i < iterations; ++i) {
    c0 = vbfdotq_f32(c0, a, b);
    c1 = vbfdotq_f32(c1, a, b);
    c2 = vbfdotq_f32(c2, a, b);
    c3 = vbfdotq_f32(c3, a, b);
    c4 = vbfdotq_f32(c4, a, b);
    c5 = vbfdotq_f32(c5, a, b);
    c6 = vbfdotq_f32(c6, a, b);
    c7 = vbfdotq_f32(c7, a, b);
  }
  asm volatile(""
               : "+w"(c0), "+w"(c1), "+w"(c2), "+w"(c3), "+w"(c4), "+w"(c5),
                 "+w"(c6), "+w"(c7));
  return vgetq_lane_f32(vaddq_f32(vaddq_f32(c0, c1), vaddq_f32(c2, c3)), 0);
}
#endif

int32_t neonI8MMA(uint64_t iterations) {
  int8x16_t a = vdupq_n_s8(1);
  int8x16_t b = vdupq_n_s8(1);
  int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
  int32x4_t c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  for (uint64_t i = 0; i < iterations; ++i) {
    asm volatile("smmla %0.4s, %1.16b, %2.16b"
                 : "+w"(c0)
                 : "w"(a), "w"(b));
    asm volatile("smmla %0.4s, %1.16b, %2.16b"
                 : "+w"(c1)
                 : "w"(a), "w"(b));
    asm volatile("smmla %0.4s, %1.16b, %2.16b"
                 : "+w"(c2)
                 : "w"(a), "w"(b));
    asm volatile("smmla %0.4s, %1.16b, %2.16b"
                 : "+w"(c3)
                 : "w"(a), "w"(b));
    asm volatile("smmla %0.4s, %1.16b, %2.16b"
                 : "+w"(c4)
                 : "w"(a), "w"(b));
    asm volatile("smmla %0.4s, %1.16b, %2.16b"
                 : "+w"(c5)
                 : "w"(a), "w"(b));
    asm volatile("smmla %0.4s, %1.16b, %2.16b"
                 : "+w"(c6)
                 : "w"(a), "w"(b));
    asm volatile("smmla %0.4s, %1.16b, %2.16b"
                 : "+w"(c7)
                 : "w"(a), "w"(b));
  }
  asm volatile(""
               : "+w"(c0), "+w"(c1), "+w"(c2), "+w"(c3), "+w"(c4), "+w"(c5),
                 "+w"(c6), "+w"(c7));
  return vgetq_lane_s32(vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)), 0);
}

int32_t neonInt8(uint64_t iterations) {
  int8x16_t a = vdupq_n_s8(1);
  int8x16_t b = vdupq_n_s8(1);
  int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
  int32x4_t c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  for (uint64_t i = 0; i < iterations; ++i) {
    c0 = vdotq_s32(c0, a, b);
    c1 = vdotq_s32(c1, a, b);
    c2 = vdotq_s32(c2, a, b);
    c3 = vdotq_s32(c3, a, b);
    c4 = vdotq_s32(c4, a, b);
    c5 = vdotq_s32(c5, a, b);
    c6 = vdotq_s32(c6, a, b);
    c7 = vdotq_s32(c7, a, b);
  }
  asm volatile(""
               : "+w"(c0), "+w"(c1), "+w"(c2), "+w"(c3), "+w"(c4), "+w"(c5),
                 "+w"(c6), "+w"(c7));
  return vgetq_lane_s32(vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)), 0);
}
#endif

double bandwidthRead(int64_t threadCount, int64_t bytes, uint64_t &bytesRead) {
  std::vector<uint64_t> buffer(static_cast<size_t>(bytes / sizeof(uint64_t)),
                               1);
  std::atomic<uint64_t> sink{0};
  auto worker = [&](int64_t id) {
    size_t begin = buffer.size() * id / threadCount;
    size_t end = buffer.size() * (id + 1) / threadCount;
    uint64_t sum = 0;
    for (size_t i = begin; i < end; ++i)
      sum += buffer[i];
    sink.fetch_add(sum, std::memory_order_relaxed);
  };
  std::vector<std::thread> workers;
  auto start = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < threadCount; ++i)
    workers.emplace_back(worker, i);
  for (auto &thread : workers)
    thread.join();
  double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  volatile uint64_t keep = sink.load();
  (void)keep;
  bytesRead = static_cast<uint64_t>(bytes);
  return elapsed;
}

double bandwidthWrite(int64_t threadCount, int64_t bytes,
                      uint64_t &bytesWritten) {
  std::vector<uint64_t> buffer(static_cast<size_t>(bytes / sizeof(uint64_t)),
                               0);
  auto worker = [&](int64_t id) {
    size_t begin = buffer.size() * id / threadCount;
    size_t end = buffer.size() * (id + 1) / threadCount;
    for (size_t i = begin; i < end; ++i)
      buffer[i] = 0x123456789abcdefULL;
  };
  std::vector<std::thread> workers;
  auto start = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < threadCount; ++i)
    workers.emplace_back(worker, i);
  for (auto &thread : workers)
    thread.join();
  volatile uint64_t keep = buffer[0];
  (void)keep;
  double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  bytesWritten = static_cast<uint64_t>(bytes);
  return elapsed;
}

double bandwidthCopy(int64_t threadCount, int64_t bytes, uint64_t &bytesMoved) {
  std::vector<uint8_t> src(static_cast<size_t>(bytes), 1);
  std::vector<uint8_t> dst(static_cast<size_t>(bytes), 0);
  auto worker = [&](int64_t id) {
    size_t begin = src.size() * id / threadCount;
    size_t end = src.size() * (id + 1) / threadCount;
    std::memcpy(dst.data() + begin, src.data() + begin, end - begin);
  };
  std::vector<std::thread> workers;
  auto start = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < threadCount; ++i)
    workers.emplace_back(worker, i);
  for (auto &thread : workers)
    thread.join();
  volatile uint8_t keep = dst[0] ^ src[0];
  (void)keep;
  double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  bytesMoved = static_cast<uint64_t>(bytes) * 2;
  return elapsed;
}

void printRate(const char *name, uint64_t ops, double elapsed,
               double flopsPerOp, const char *unit) {
  double value = static_cast<double>(ops) * flopsPerOp / elapsed / 1.0e9;
  std::cout << std::left << std::setw(24) << name << std::fixed
            << std::setprecision(3) << value << " " << unit << "\n";
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "CPU microbenchmark\n");

  std::string brand = sysctlString("machdep.cpu.brand_string");
  int ncpu = sysctlInt("hw.ncpu");
  int threadCount = threads > 0 ? static_cast<int>(threads) : ncpu;
  if (threadCount <= 0)
    threadCount = 1;

  std::cout << "cpu: " << (brand.empty() ? "unknown" : brand) << "\n";
  std::cout << "logical_cpus: " << ncpu << "\n";
  std::cout << "features:";
  for (const char *feature :
       {"FP16", "DotProd", "BF16", "I8MM", "SVE", "SME", "FHM"})
    std::cout << " " << feature << "="
              << (sysctlInt(
                      (std::string("hw.optional.arm.FEAT_") + feature).c_str())
                      ? "1"
                      : "0");
  std::cout << "\n";
  std::cout << "bandwidth_threads: " << threadCount << "\n";

  std::cout << "\ncompute:\n";
  uint64_t ops = 0;
  double elapsed = measure(
      "scalar_f32",
      [&](uint64_t n) {
        scalarFma(n);
        return n * 4;
      },
      ops);
  printRate("scalar_f32", ops, elapsed, 2.0, "GFLOPS");

#if defined(__aarch64__)
  elapsed = measure(
      "neon_f32",
      [&](uint64_t n) {
        neonF32(n);
        return n * 8;
      },
      ops);
  printRate("neon_f32", ops, elapsed, 2.0 * 4.0, "GFLOPS");

  if (sysctlInt("hw.optional.arm.FEAT_FP16")) {
    elapsed = measure(
        "neon_f16",
        [&](uint64_t n) {
          neonF16(n);
          return n * 8;
        },
        ops);
    printRate("neon_f16", ops, elapsed, 2.0 * 8.0, "GFLOPS");
  }

#if defined(__aarch64__) && defined(LLMCP_ENABLE_BFDOT)
  if (sysctlInt("hw.optional.arm.FEAT_BF16")) {
    elapsed = measure(
        "neon_bf16_dot",
        [&](uint64_t n) {
          neonBF16(n);
          return n * 8;
        },
        ops);
    printRate("neon_bf16_dot", ops, elapsed, 2.0 * 8.0, "GFLOPS");
  }
#endif

  if (sysctlInt("hw.optional.arm.FEAT_DotProd")) {
    elapsed = measure(
        "neon_i8_dot",
        [&](uint64_t n) {
          neonInt8(n);
          return n * 8;
        },
        ops);
    printRate("neon_i8_dot", ops, elapsed, 2.0 * 16.0, "GOPS");
  }

  if (sysctlInt("hw.optional.arm.FEAT_I8MM")) {
    elapsed = measure(
        "neon_i8_mma",
        [&](uint64_t n) {
          neonI8MMA(n);
          return n * 8;
        },
        ops);
    // SMMLA performs a 2x8 by 8x2 integer matrix multiply: 32 MACs.
    printRate("neon_i8_mma", ops, elapsed, 2.0 * 32.0, "GOPS");
  }
#else
  std::cout << "neon: unsupported on this architecture\n";
#endif

  std::cout << "\nbandwidth:\n";
  int64_t bytes = bufferMB * 1024LL * 1024LL;
  uint64_t moved = 0;
  elapsed = bandwidthRead(threadCount, bytes, moved);
  printRate("read", moved, elapsed, 1.0, "GB/s");
  elapsed = bandwidthWrite(threadCount, bytes, moved);
  printRate("write", moved, elapsed, 1.0, "GB/s");
  elapsed = bandwidthCopy(threadCount, bytes, moved);
  printRate("copy", moved, elapsed, 1.0, "GB/s");
  return 0;
}
