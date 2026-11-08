//===- WeightArena.cpp - Load packed weights ----------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llmcp/Runtime/AotRuntime.h"
#include "llmcp/Support/PackedWeights.h"

#include "llvm/Support/JSON.h"

#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace mlir::llmcp::runtime;
using namespace llvm;

Arena::~Arena() {
  if (base)
    munmap(const_cast<uint8_t *>(base), mappedSize);
  if (fd >= 0)
    close(fd);
}

Error Arena::load(StringRef bin) {
  // Build a new mapping transactionally so a failed reload preserves the arena.
  Arena loaded;
  loaded.fd = open(bin.str().c_str(), O_RDONLY);
  if (loaded.fd < 0)
    return errorCodeToError(std::error_code(errno, std::generic_category()));
  struct stat st;
  if (fstat(loaded.fd, &st) != 0)
    return errorCodeToError(std::error_code(errno, std::generic_category()));
  if (st.st_size < static_cast<off_t>(::llmcp::kPackedWeightsHeaderBytes))
    return createStringError(inconvertibleErrorCode(),
                             "packed weights file is too small");
  loaded.mappedSize = static_cast<uint64_t>(st.st_size);
  void *mapping =
      mmap(nullptr, loaded.mappedSize, PROT_READ, MAP_PRIVATE, loaded.fd, 0);
  if (mapping == MAP_FAILED)
    return errorCodeToError(std::error_code(errno, std::generic_category()));
  loaded.base = static_cast<const uint8_t *>(mapping);
  StringRef file(reinterpret_cast<const char *>(loaded.base),
                 loaded.mappedSize);
  auto header = ::llmcp::PackedWeightsHeader::read(file);
  if (!header)
    return header.takeError();
  loaded.size = header->arenaBytes;
  auto json =
      json::parse(file.substr(header->manifestOffset, header->manifestBytes));
  if (!json)
    return json.takeError();
  auto *object = json->getAsObject();
  auto invalid = []() {
    return createStringError(inconvertibleErrorCode(),
                             "invalid embedded weight manifest");
  };
  if (!object)
    return invalid();
  auto total = object->getInteger("total_bytes");
  auto headerBytes = object->getInteger("header_bytes");
  auto mode = object->getString("quantization");
  auto *tensors = object->getArray("tensors");
  if (!total || *total < 0 || static_cast<uint64_t>(*total) != loaded.size ||
      !headerBytes || *headerBytes != ::llmcp::kPackedWeightsHeaderBytes ||
      !mode || (*mode != "none" && *mode != "q8_0") || !tensors ||
      tensors->empty())
    return invalid();
  loaded.quantization = mode->str();
  if (*mode == "q8_0" && (object->getInteger("block_size") != 32 ||
                          object->getString("quant_layout") != "q8_0_split"))
    return invalid();
  for (const json::Value &value : *tensors) {
    auto *entry = value.getAsObject();
    if (!entry)
      return invalid();
    auto name = entry->getString("name");
    auto offset = entry->getInteger("offset");
    auto bytes = entry->getInteger("size");
    auto dtype = entry->getString("dtype");
    auto *shape = entry->getArray("shape");
    if (!name || name->empty() || !offset || *offset < 64 || *offset % 64 ||
        !bytes || *bytes <= 0 || !dtype || !shape || shape->empty())
      return invalid();
    uint64_t begin = static_cast<uint64_t>(*offset);
    uint64_t length = static_cast<uint64_t>(*bytes);
    if (begin > loaded.size || length > loaded.size - begin)
      return createStringError(inconvertibleErrorCode(),
                               "tensor outside arena: " + name->str());
    Weight weight;
    weight.dtype = dtype->str();
    uint64_t elements = 1;
    for (const json::Value &dim : *shape) {
      auto size = dim.getAsInteger();
      if (!size || *size <= 0 ||
          elements > std::numeric_limits<uint64_t>::max() /
                         static_cast<uint64_t>(*size))
        return invalid();
      elements *= static_cast<uint64_t>(*size);
      weight.shape.push_back(*size);
    }
    uint64_t unitBytes;
    if (*dtype == "Q8_0") {
      if (*mode != "q8_0" || weight.shape.size() != 2 || weight.shape[1] % 32)
        return invalid();
      elements /= 32;
      unitBytes = 34;
    } else if (*dtype == "BF16") {
      unitBytes = 2;
    } else if (*dtype == "F32") {
      unitBytes = 4;
    } else {
      return invalid();
    }
    if (elements > std::numeric_limits<uint64_t>::max() / unitBytes ||
        elements * unitBytes != length)
      return invalid();
    weight.data = loaded.base + begin;
    if (!loaded.weights.emplace(name->str(), std::move(weight)).second)
      return invalid();
  }
  std::swap(fd, loaded.fd);
  std::swap(base, loaded.base);
  std::swap(size, loaded.size);
  std::swap(mappedSize, loaded.mappedSize);
  quantization.swap(loaded.quantization);
  weights.swap(loaded.weights);
  return Error::success();
}

const Weight &Arena::get(StringRef name) const {
  auto it = weights.find(name.str());
  assert(it != weights.end() && "missing weight");
  return it->second;
}
