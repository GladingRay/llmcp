//===- AotRuntime.cpp - Shared AOT arena and runtime ---------------------===//
//
//===----------------------------------------------------------------------===//

#include "llmcp/Runtime/AotRuntime.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace mlir::llmcp::runtime;
using namespace llvm;

Arena::~Arena() {
  if (base)
    munmap(const_cast<uint8_t *>(base), size);
  if (fd >= 0)
    close(fd);
}

Error Arena::load(StringRef bin, StringRef manifest) {
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

const Weight &Arena::get(StringRef name) const {
  auto it = weights.find(name.str());
  assert(it != weights.end() && "missing weight");
  return it->second;
}

std::pair<std::vector<float>, std::vector<float>>
mlir::llmcp::runtime::makeCosSin(int64_t pos, int64_t rotaryDim, float theta) {
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

AotRuntime::~AotRuntime() {
  if (handle)
    dlclose(handle);
}

bool AotRuntime::init(const std::string &path, int64_t maxSequence) {
  capacity = maxSequence;
  kv.assign(static_cast<size_t>(8 * 2 * 4 * capacity * 256), 0);
  delta.assign(static_cast<size_t>(24 * 32 * 128 * 128), 0.0f);
  conv.assign(static_cast<size_t>(24 * 3 * 8192), 0);
  handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    errs() << "AotRuntime: dlopen: " << dlerror() << "\n";
    return false;
  }
  fn = reinterpret_cast<Fn>(dlsym(handle, "_mlir_ciface_llmcp_model_aot"));
  if (!fn) {
    errs() << "AotRuntime: dlsym: " << dlerror() << "\n";
    return false;
  }
  return true;
}

bool AotRuntime::forward(const Arena &arena,
                         const std::vector<int64_t> &tokens, int64_t startPos,
                         int64_t cacheLength, std::vector<float> &logits) {
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
      const_cast<int8_t *>(reinterpret_cast<const int8_t *>(arena.base)), 0,
      {static_cast<int64_t>(arena.size)}, {1}};
  MemRef<uint16_t, 5> rk{kv.data(),
                         kv.data(),
                         0,
                         {8, 2, 4, 0, 256},
                         {2 * 4 * capacity * 256, 4 * capacity * 256,
                          capacity * 256, 256, 1}};
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
