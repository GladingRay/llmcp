//===- PackedWeights.h - Packed weight container format --------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_SUPPORT_PACKED_WEIGHTS_H
#define LLMCP_SUPPORT_PACKED_WEIGHTS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <cstring>

namespace llmcp {

inline constexpr uint64_t kPackedWeightsHeaderBytes = 64;

/// Little-endian file header. Tensor offsets include this header; the JSON
/// manifest follows the arena, preserving the offsets embedded in AOT code.
struct PackedWeightsHeader {
  uint64_t arenaBytes;
  uint64_t manifestOffset;
  uint64_t manifestBytes;

  void write(uint8_t *data) const {
    std::memset(data, 0, kPackedWeightsHeaderBytes);
    std::memcpy(data, "LLMCPWGT", 8);
    llvm::support::endian::write32le(data + 8, 1);
    llvm::support::endian::write32le(data + 12, kPackedWeightsHeaderBytes);
    llvm::support::endian::write64le(data + 16, arenaBytes);
    llvm::support::endian::write64le(data + 24, manifestOffset);
    llvm::support::endian::write64le(data + 32, manifestBytes);
  }

  static llvm::Expected<PackedWeightsHeader> read(llvm::StringRef file) {
    auto error = [](const char *message) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
    };
    if (file.size() < kPackedWeightsHeaderBytes ||
        file.take_front(8) != "LLMCPWGT")
      return error(
          "invalid packed weights header; repack weights with llmcp-pack");
    const char *data = file.data();
    if (llvm::support::endian::read32le(data + 8) != 1 ||
        llvm::support::endian::read32le(data + 12) != kPackedWeightsHeaderBytes)
      return error("unsupported packed weights format version or header size");
    PackedWeightsHeader header{llvm::support::endian::read64le(data + 16),
                               llvm::support::endian::read64le(data + 24),
                               llvm::support::endian::read64le(data + 32)};
    if (header.arenaBytes < kPackedWeightsHeaderBytes ||
        header.manifestOffset != header.arenaBytes ||
        header.manifestOffset > file.size() || header.manifestBytes == 0 ||
        header.manifestBytes != file.size() - header.manifestOffset)
      return error("invalid packed weights arena or manifest bounds");
    return header;
  }
};

} // namespace llmcp

#endif // LLMCP_SUPPORT_PACKED_WEIGHTS_H
