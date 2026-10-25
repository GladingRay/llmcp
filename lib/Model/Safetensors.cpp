//===- Safetensors.cpp - Read safetensors headers -------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llmcp/Model/Safetensors.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <utility>

using namespace mlir::llmcp;
using llvm::StringRef;
using llvm::json::Object;
using llvm::json::Value;

namespace {

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

constexpr uint64_t kMaxHeaderBytes = 64ull * 1024ull * 1024ull;

} // namespace

uint64_t TensorInfo::getNumElements() const {
  uint64_t numElements = 1;
  for (int64_t dim : shape)
    numElements *= static_cast<uint64_t>(dim);
  return numElements;
}

llvm::Expected<uint64_t>
mlir::llmcp::getSafetensorsDTypeSize(StringRef dtype) {
  if (dtype == "BF16" || dtype == "F16" || dtype == "I16" ||
      dtype == "U16")
    return 2;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32")
    return 4;
  if (dtype == "F64" || dtype == "I64" || dtype == "U64")
    return 8;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" ||
      dtype == "F8_E4M3" || dtype == "F8_E5M2")
    return 1;
  return makeError(llvm::Twine("unsupported safetensors dtype '") + dtype +
                   "'");
}

llvm::Expected<TensorMap>
mlir::llmcp::parseSafetensorsHeader(StringRef json, uint64_t dataStart,
                                   uint64_t fileSize) {
  if (dataStart > fileSize)
    return makeError("safetensors header extends past end of file");

  auto parsed = llvm::json::parse(json);
  if (!parsed)
    return parsed.takeError();

  const Object *root = parsed->getAsObject();
  if (!root)
    return makeError("safetensors header must be a JSON object");

  TensorMap tensors;
  for (const auto &entry : *root) {
    StringRef name = entry.first;
    if (name == "__metadata__")
      continue;

    const Object *info = entry.second.getAsObject();
    if (!info)
      return makeError(llvm::Twine("tensor '") + name +
                       "' metadata must be an object");

    std::optional<StringRef> dtype = info->getString("dtype");
    if (!dtype)
      return makeError(llvm::Twine("tensor '") + name + "' is missing dtype");
    auto elementBytes = getSafetensorsDTypeSize(*dtype);
    if (!elementBytes)
      return elementBytes.takeError();

    const auto *shape = info->getArray("shape");
    if (!shape)
      return makeError(llvm::Twine("tensor '") + name + "' is missing shape");

    TensorInfo tensor;
    tensor.name = name.str();
    tensor.dtype = dtype->str();
    uint64_t numElements = 1;
    for (const Value &dimValue : *shape) {
      auto dim = dimValue.getAsInteger();
      if (!dim || *dim < 0)
        return makeError(llvm::Twine("tensor '") + name +
                         "' has an invalid shape");
      tensor.shape.push_back(*dim);
      numElements *= static_cast<uint64_t>(*dim);
    }

    const auto *offsets = info->getArray("data_offsets");
    if (!offsets || offsets->size() != 2)
      return makeError(llvm::Twine("tensor '") + name +
                       "' is missing data_offsets");
    auto begin = (*offsets)[0].getAsInteger();
    auto end = (*offsets)[1].getAsInteger();
    if (!begin || !end || *begin < 0 || *end < *begin)
      return makeError(llvm::Twine("tensor '") + name +
                       "' has invalid data_offsets");
    if (static_cast<uint64_t>(*end) > fileSize - dataStart)
      return makeError(llvm::Twine("tensor '") + name +
                       "' extends past end of file");

    const uint64_t expectedBytes =
        numElements * static_cast<uint64_t>(*elementBytes);
    if (static_cast<uint64_t>(*end - *begin) != expectedBytes)
      return makeError(llvm::Twine("tensor '") + name +
                       "' data size does not match dtype and shape");

    tensor.dataOffset = dataStart + static_cast<uint64_t>(*begin);
    tensor.dataSize = static_cast<uint64_t>(*end - *begin);

    auto inserted = tensors.try_emplace(name, std::move(tensor));
    if (!inserted.second)
      return makeError(llvm::Twine("duplicate tensor '") + name + "'");
  }

  return tensors;
}

llvm::Expected<TensorMap>
mlir::llmcp::readSafetensorsFile(StringRef path) {
  std::ifstream input(path.str(), std::ios::binary | std::ios::ate);
  if (!input)
    return makeError(llvm::Twine("cannot open safetensors file '") + path +
                     "'");

  const std::streamoff end = input.tellg();
  if (end < 8)
    return makeError(llvm::Twine("safetensors file '") + path +
                     "' is too small");

  input.seekg(0, std::ios::beg);
  std::array<unsigned char, 8> sizeBytes{};
  if (!input.read(reinterpret_cast<char *>(sizeBytes.data()), 8))
    return makeError(llvm::Twine("cannot read header size from '") + path +
                     "'");

  uint64_t headerBytes = 0;
  for (unsigned i = 0; i < 8; ++i)
    headerBytes |= static_cast<uint64_t>(sizeBytes[i]) << (8 * i);

  const uint64_t fileSize = static_cast<uint64_t>(end);
  if (headerBytes > kMaxHeaderBytes || headerBytes > fileSize - 8)
    return makeError(llvm::Twine("invalid safetensors header size in '") +
                     path + "'");

  std::string header(headerBytes, '\0');
  input.seekg(8, std::ios::beg);
  if (!input.read(header.data(), static_cast<std::streamsize>(headerBytes)))
    return makeError(llvm::Twine("cannot read header from '") + path + "'");

  return parseSafetensorsHeader(header, 8 + headerBytes, fileSize);
}

llvm::Expected<SafetensorsIndex>
mlir::llmcp::readSafetensorsIndex(StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return llvm::errorCodeToError(bufferOrErr.getError());

  auto parsed = llvm::json::parse((*bufferOrErr)->getBuffer());
  if (!parsed)
    return parsed.takeError();

  const Object *root = parsed->getAsObject();
  if (!root)
    return makeError("safetensors index must be a JSON object");

  const Object *weightMap = root->getObject("weight_map");
  if (!weightMap)
    return makeError("safetensors index is missing weight_map");

  SafetensorsIndex index;
  index.totalSize =
      static_cast<uint64_t>(root->getInteger("total_size").value_or(0));

  for (const auto &entry : *weightMap) {
    std::optional<StringRef> shard = entry.second.getAsString();
    if (!shard)
      return makeError(llvm::Twine("weight_map entry '") + StringRef(entry.first) +
                       "' must be a string");
    index.weightMap.try_emplace(StringRef(entry.first), shard->str());
  }

  return index;
}
