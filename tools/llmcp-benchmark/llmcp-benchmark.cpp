//===- llmcp-benchmark.cpp - Benchmark the AOT model ---------------------===//
//
// Measures prefill and decode throughput of llmcp_model_aot.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace llvm;

namespace {

cl::opt<std::string> weightsPath("weights", cl::desc("Packed arena"),
                                 cl::init("build/weights.bin"));
cl::opt<std::string> manifestPath("manifest", cl::desc("Arena manifest"),
                                  cl::init("build/weights.json"));
cl::opt<std::string> aotPath("aot", cl::desc("AOT shared library"),
                             cl::Required);
cl::opt<int64_t> prefillTokens("prefill-tokens",
                               cl::desc("Number of prefill tokens"),
                               cl::init(32));
cl::opt<int64_t> decodeTokens("decode-tokens",
                              cl::desc("Number of decode tokens"), cl::init(8));
cl::opt<int64_t> tokenId("token", cl::desc("Synthetic token id"),
                         cl::init(9419));
cl::opt<bool> verbose("verbose", cl::desc("Print per-phase detail"),
                      cl::init(false));

struct Weight {
  const uint8_t *data = nullptr;
  std::vector<int64_t> shape;
  std::string dtype;
};

struct Arena {
  int fd = -1;
  const uint8_t *base = nullptr;
  uint64_t size = 0;
  std::unordered_map<std::string, Weight> weights;

  ~Arena() {
    if (base)
      munmap(const_cast<uint8_t *>(base), size);
    if (fd >= 0)
      close(fd);
  }

  Error load(StringRef bin, StringRef manifest) {
    fd = open(bin.str().c_str(), O_RDONLY);
    if (fd < 0)
      return errorCodeToError(std::error_code(errno, std::generic_category()));
    struct stat st;
    if (fstat(fd, &st) != 0)
      return errorCodeToError(std::error_code(errno, std::generic_category()));
    size = static_cast<uint64_t>(st.st_size);
    base = static_cast<const uint8_t *>(
        mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (base == MAP_FAILED)
      return errorCodeToError(std::error_code(errno, std::generic_category()));
    auto buffer = MemoryBuffer::getFile(manifest);
    if (!buffer)
      return errorCodeToError(buffer.getError());
    auto json = json::parse((*buffer)->getBuffer());
    if (!json)
      return json.takeError();
    auto *object = json->getAsObject();
    auto *tensors = object ? object->getArray("tensors") : nullptr;
    if (!tensors)
      return createStringError(inconvertibleErrorCode(),
                               "manifest has no tensors");
    for (const json::Value &value : *tensors) {
      auto *entry = value.getAsObject();
      if (!entry)
        continue;
      Weight weight;
      std::string name = entry->getString("name")->str();
      uint64_t offset = entry->getInteger("offset").value_or(0);
      uint64_t bytes = entry->getInteger("size").value_or(0);
      weight.dtype = entry->getString("dtype")->str();
      for (const json::Value &dim : *entry->getArray("shape"))
        weight.shape.push_back(dim.getAsInteger().value_or(0));
      if (offset + bytes > size)
        return createStringError(inconvertibleErrorCode(),
                                 "tensor outside arena: " + name);
      weight.data = base + offset;
      weights[name] = std::move(weight);
    }
    return Error::success();
  }
};

template <typename T, int N> struct MemRef {
  T *base;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

std::pair<std::vector<float>, std::vector<float>>
makeCosSin(int64_t pos, int64_t rotaryDim, float theta) {
  std::vector<float> cos(rotaryDim), sin(rotaryDim);
  for (int64_t d = 0; d < rotaryDim / 2; ++d) {
    float invFreq = 1.0f / std::pow(theta, static_cast<float>(2 * d) /
                                               static_cast<float>(rotaryDim));
    float angle = static_cast<float>(pos) * invFreq;
    cos[d] = cos[d + rotaryDim / 2] = std::cos(angle);
    sin[d] = sin[d + rotaryDim / 2] = std::sin(angle);
  }
  return {cos, sin};
}

struct AotRuntime {
  using Fn = void (*)(void *, void *, void *, void *, void *, void *, void *,
                      void *, void *, void *);
  void *handle = nullptr;
  Fn fn = nullptr;
  int64_t capacity = 0;
  std::vector<uint16_t> kv;
  std::vector<float> delta;
  std::vector<uint16_t> conv;

  ~AotRuntime() {
    if (handle)
      dlclose(handle);
  }

  bool init(const std::string &path, int64_t maxSequence) {
    capacity = maxSequence;
    kv.assign(static_cast<size_t>(8 * 2 * 4 * capacity * 256), 0);
    delta.assign(static_cast<size_t>(24 * 32 * 128 * 128), 0.0f);
    conv.assign(static_cast<size_t>(24 * 3 * 8192), 0);
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      errs() << "llmcp-benchmark: dlopen: " << dlerror() << "\n";
      return false;
    }
    fn = reinterpret_cast<Fn>(dlsym(handle, "_mlir_ciface_llmcp_model_aot"));
    if (!fn) {
      errs() << "llmcp-benchmark: dlsym: " << dlerror() << "\n";
      return false;
    }
    return true;
  }

  bool forward(const Arena &arena, const std::vector<int64_t> &tokens,
               int64_t startPos, int64_t cacheLength,
               std::vector<float> &logits) {
    int64_t sequence = static_cast<int64_t>(tokens.size());
    if (startPos + sequence > capacity)
      return false;
    std::vector<int64_t> tokenData = tokens;
    std::vector<float> cos(static_cast<size_t>(sequence * 64));
    std::vector<float> sin(static_cast<size_t>(sequence * 64));
    for (int64_t t = 0; t < sequence; ++t) {
      auto [c, s] = makeCosSin(startPos + t, 64, 10000000.0f);
      std::copy(c.begin(), c.end(), cos.begin() + t * 64);
      std::copy(s.begin(), s.end(), sin.begin() + t * 64);
    }
    logits.assign(static_cast<size_t>(sequence * 248320), 0.0f);

    MemRef<int64_t, 1> rt{tokenData.data(), tokenData.data(), 0, {0}, {1}};
    rt.sizes[0] = sequence;
    MemRef<float, 2> rc{cos.data(), cos.data(), 0, {0, 64}, {64, 1}};
    rc.sizes[0] = sequence;
    MemRef<float, 2> rs{sin.data(), sin.data(), 0, {0, 64}, {64, 1}};
    rs.sizes[0] = sequence;
    MemRef<int8_t, 1> ra{
        const_cast<int8_t *>(reinterpret_cast<const int8_t *>(arena.base)),
        const_cast<int8_t *>(reinterpret_cast<const int8_t *>(arena.base)),
        0,
        {static_cast<int64_t>(arena.size)},
        {1}};
    MemRef<uint16_t, 5> rk{
        kv.data(),
        kv.data(),
        0,
        {8, 2, 4, 0, 256},
        {2 * 4 * capacity * 256, 4 * capacity * 256, capacity * 256, 256, 1}};
    rk.sizes[3] = capacity;
    MemRef<float, 4> rd{delta.data(),
                        delta.data(),
                        0,
                        {24, 32, 128, 128},
                        {32 * 128 * 128, 128 * 128, 128, 1}};
    MemRef<uint16_t, 3> rv{
        conv.data(), conv.data(), 0, {24, 3, 8192}, {3 * 8192, 8192, 1}};
    MemRef<float, 2> ro{
        logits.data(), logits.data(), 0, {0, 248320}, {248320, 1}};
    ro.sizes[0] = sequence;

    fn(&rt, &rc, &rs, (void *)startPos, (void *)cacheLength, &ra, &rk, &rd, &rv,
       &ro);
    return true;
  }
};

int64_t argmax(const std::vector<float> &values) {
  return static_cast<int64_t>(std::max_element(values.begin(), values.end()) -
                              values.begin());
}

double seconds(std::chrono::steady_clock::time_point begin,
               std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "llmcp-benchmark\n");

  if (prefillTokens <= 0 || decodeTokens < 0) {
    errs() << "llmcp-benchmark: prefill-tokens must be > 0 and "
              "decode-tokens >= 0\n";
    return 1;
  }

  Arena arena;
  if (Error error = arena.load(weightsPath, manifestPath)) {
    errs() << "llmcp-benchmark: " << toString(std::move(error)) << "\n";
    return 1;
  }

  AotRuntime runtime;
  if (!runtime.init(aotPath, prefillTokens + decodeTokens + 8))
    return 1;

  std::vector<int64_t> prompt(static_cast<size_t>(prefillTokens), tokenId);
  std::vector<float> logits;

  auto prefillBegin = std::chrono::steady_clock::now();
  if (!runtime.forward(arena, prompt, 0, prefillTokens, logits))
    return 1;
  auto prefillEnd = std::chrono::steady_clock::now();
  double prefillSeconds = seconds(prefillBegin, prefillEnd);

  std::vector<int64_t> generated;
  generated.reserve(static_cast<size_t>(decodeTokens));
  auto decodeBegin = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < decodeTokens; ++i) {
    int64_t next = argmax(logits);
    generated.push_back(next);
    std::vector<int64_t> one{next};
    int64_t position = prefillTokens + i;
    if (!runtime.forward(arena, one, position, position + 1, logits))
      return 1;
  }
  auto decodeEnd = std::chrono::steady_clock::now();
  double decodeSeconds = seconds(decodeBegin, decodeEnd);
  double totalSeconds = prefillSeconds + decodeSeconds;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "prefill_tokens: " << prefillTokens << "\n";
  std::cout << "decode_tokens: " << decodeTokens << "\n";
  std::cout << "prefill_seconds: " << prefillSeconds << "\n";
  std::cout << "decode_seconds: " << decodeSeconds << "\n";
  std::cout << "total_seconds: " << totalSeconds << "\n";
  std::cout << "prefill_tokens_per_second: " << (prefillTokens / prefillSeconds)
            << "\n";
  std::cout << "decode_tokens_per_second: "
            << (decodeSeconds > 0 ? decodeTokens / decodeSeconds : 0.0) << "\n";
  if (verbose) {
    std::cout << "generated_tokens:";
    for (int64_t id : generated)
      std::cout << " " << id;
    std::cout << "\n";
  }
  return 0;
}
