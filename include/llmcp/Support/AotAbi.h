//===- AotAbi.h - Shared AOT interface version ------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_SUPPORT_AOT_ABI_H
#define LLMCP_SUPPORT_AOT_ABI_H

#include <cstdint>

namespace llmcp {
/// Version 2 returns one row of logits from both generation entry points.
inline constexpr int32_t kAotAbiVersion = 2;
} // namespace llmcp

#endif // LLMCP_SUPPORT_AOT_ABI_H
