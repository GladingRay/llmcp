//===- WeightPacker.cpp - Pack safetensors into the runtime arena --------===//

#include "llmcp/Model/WeightPacker.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace mlir::llmcp;

namespace {

class MappedFile {
public:
  MappedFile() = default;
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;
  MappedFile(MappedFile &&other) noexcept { *this = std::move(other); }
  MappedFile &operator=(MappedFile &&other) noexcept {
    std::swap(fd_, other.fd_);
    std::swap(base_, other.base_);
    std::swap(size_, other.size_);
    return *this;
  }
  ~MappedFile() {
    if (base_ && base_ != MAP_FAILED)
      munmap(base_, size_);
    if (fd_ >= 0)
      close(fd_);
  }

  static llvm::Expected<MappedFile> open(llvm::StringRef path, bool writable,
                                         uint64_t size = 0) {
    MappedFile result;
    int flags = writable ? O_RDWR | O_CREAT | O_TRUNC : O_RDONLY;
    result.fd_ = ::open(path.str().c_str(), flags, 0644);
    if (result.fd_ < 0)
      return llvm::errorCodeToError(
          std::error_code(errno, std::generic_category()));

    if (writable) {
      if (::ftruncate(result.fd_, static_cast<off_t>(size)) != 0)
        return llvm::errorCodeToError(
            std::error_code(errno, std::generic_category()));
      result.size_ = size;
    } else {
      struct stat st;
      if (::fstat(result.fd_, &st) != 0)
        return llvm::errorCodeToError(
            std::error_code(errno, std::generic_category()));
      result.size_ = static_cast<uint64_t>(st.st_size);
    }

    int prot = writable ? PROT_READ | PROT_WRITE : PROT_READ;
    result.base_ =
        ::mmap(nullptr, result.size_, prot, MAP_SHARED, result.fd_, 0);
    if (result.base_ == MAP_FAILED)
      return llvm::errorCodeToError(
          std::error_code(errno, std::generic_category()));
    return std::move(result);
  }

  uint8_t *data() const { return static_cast<uint8_t *>(base_); }
  uint64_t size() const { return size_; }

private:
  int fd_ = -1;
  void *base_ = nullptr;
  uint64_t size_ = 0;
};

llvm::Expected<TensorMap> loadTensorHeaders(llvm::StringRef modelDir,
                                            SafetensorsIndex &index,
                                            bool &hasIndex) {
  llvm::SmallString<256> indexPath(modelDir);
  llvm::sys::path::append(indexPath, "model.safetensors.index.json");
  hasIndex = llvm::sys::fs::exists(indexPath);
  if (!hasIndex) {
    llvm::SmallString<256> single(modelDir);
    llvm::sys::path::append(single, "model.safetensors");
    if (!llvm::sys::fs::exists(single))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "no safetensors index or single shard");
    auto tensors = readSafetensorsFile(single);
    if (!tensors)
      return tensors.takeError();
    index.weightMap[""] = "model.safetensors";
    return tensors;
  }

  auto indexOrErr = readSafetensorsIndex(indexPath);
  if (!indexOrErr)
    return indexOrErr.takeError();
  index = std::move(*indexOrErr);

  TensorMap merged;
  for (const auto &item : index.weightMap) {
    llvm::SmallString<256> shard(modelDir);
    llvm::sys::path::append(shard, item.getValue());
    auto shardTensors = readSafetensorsFile(shard);
    if (!shardTensors)
      return shardTensors.takeError();
    for (const auto &tensor : *shardTensors)
      merged[tensor.getKey()] = tensor.getValue();
  }
  return merged;
}

llvm::Error writeManifest(llvm::StringRef path, llvm::StringRef modelDir,
                          const WeightLayout &layout) {
  llvm::json::Object root;
  root["model"] = modelDir.str();
  root["total_bytes"] = layout.getTotalBytes();
  root["header_bytes"] = layout.getHeaderBytes();
  root["group_stride"] = layout.getGroupStride();
  root["num_groups"] = layout.getNumGroups();
  root["alignment"] = 64;
  root["quantization"] = stringifyQuantizationMode(layout.getQuantization());
  if (layout.getQuantization() != QuantizationMode::None)
    root["block_size"] = kQ8_0BlockSize;

  llvm::json::Array entries;
  for (const WeightEntry &entry : layout.getEntries()) {
    llvm::json::Object object;
    object["name"] = entry.name;
    object["offset"] = entry.offset;
    object["size"] = entry.size;
    object["dtype"] = entry.dtype;
    object["source_size"] = entry.sourceSize;
    llvm::json::Array shape;
    for (int64_t dim : entry.shape)
      shape.push_back(dim);
    object["shape"] = std::move(shape);
    entries.push_back(std::move(object));
  }
  root["tensors"] = std::move(entries);

  std::error_code error;
  llvm::raw_fd_ostream out(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::errorCodeToError(error);
  out << llvm::json::Value(std::move(root)) << "\n";
  return llvm::Error::success();
}

} // namespace

llvm::Error mlir::llmcp::packWeights(llvm::StringRef modelDir,
                                     const WeightLayout &layout,
                                     const TensorMap &tensors,
                                     llvm::StringRef outputPath,
                                     llvm::StringRef manifestPath,
                                     const PackOptions &options) {
  SafetensorsIndex index;
  bool hasIndex = false;
  if (options.dryRun) {
    // Metadata has already been loaded by the caller. Validate that every
    // layout entry has a source tensor with the exact byte size.
    for (const WeightEntry &entry : layout.getEntries()) {
      auto it = tensors.find(entry.name);
      if (it == tensors.end())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "missing tensor in checkpoint: " +
                                           entry.name);
      if (it->second.dataSize != entry.sourceSize)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "size mismatch for " + entry.name + ": checkpoint " +
                std::to_string(it->second.dataSize) + ", layout source " +
                std::to_string(entry.sourceSize));
    }
    if (options.verbose)
      llvm::outs() << "validated " << layout.getEntries().size()
                   << " tensors\n";
    return llvm::Error::success();
  }

  // Re-read the per-shard index here so the packer does not depend on the
  // caller's TensorMap retaining shard names.
  llvm::SmallString<256> indexPath(modelDir);
  llvm::sys::path::append(indexPath, "model.safetensors.index.json");
  if (!llvm::sys::fs::exists(indexPath))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "model.safetensors.index.json not found");
  auto indexOrErr = readSafetensorsIndex(indexPath);
  if (!indexOrErr)
    return indexOrErr.takeError();
  index = std::move(*indexOrErr);

  auto outputOrErr =
      MappedFile::open(outputPath, /*writable=*/true, layout.getTotalBytes());
  if (!outputOrErr)
    return outputOrErr.takeError();
  MappedFile output = std::move(*outputOrErr);
  std::memset(output.data(), 0, layout.getHeaderBytes());

  llvm::StringMap<std::unique_ptr<MappedFile>> shards;
  for (const auto &item : index.weightMap) {
    llvm::StringRef shardName = item.getValue();
    if (shards.count(shardName))
      continue;
    llvm::SmallString<256> shardPath(modelDir);
    llvm::sys::path::append(shardPath, shardName);
    auto shardOrErr = MappedFile::open(shardPath, /*writable=*/false);
    if (!shardOrErr)
      return shardOrErr.takeError();
    shards[shardName] = std::make_unique<MappedFile>(std::move(*shardOrErr));
  }

  for (const WeightEntry &entry : layout.getEntries()) {
    auto infoIt = tensors.find(entry.name);
    if (infoIt == tensors.end())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "missing tensor in checkpoint: " +
                                         entry.name);
    const TensorInfo &info = infoIt->second;
    auto shardIt = index.weightMap.find(entry.name);
    if (shardIt == index.weightMap.end())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "tensor not present in safetensors index: " + entry.name);
    auto mappedIt = shards.find(shardIt->getValue());
    if (mappedIt == shards.end())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "shard not mapped: " +
                                         shardIt->getValue());
    MappedFile &shard = *mappedIt->second;
    if (info.dataOffset + info.dataSize > shard.size())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "tensor extends past shard: " +
                                         entry.name);
    if (entry.quantized) {
      if (llvm::Error error = quantizeTensorQ8_0(
              info, shard.data() + info.dataOffset, output.data() + entry.offset))
        return error;
    } else {
      std::memcpy(output.data() + entry.offset, shard.data() + info.dataOffset,
                  entry.size);
    }
    if (options.verbose)
      llvm::outs() << "  " << entry.offset << " +" << entry.size << " "
                   << entry.name << "\n";
  }

  if (!manifestPath.empty()) {
    if (llvm::Error error = writeManifest(manifestPath, modelDir, layout))
      return error;
  }
  return llvm::Error::success();
}
