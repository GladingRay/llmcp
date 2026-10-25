//===- WeightPacker.h - Pack safetensors into the runtime arena ----------*- C++
//-*-===//
//
// Copies text-model tensors into the deterministic arena described by
// WeightLayout and writes a JSON manifest. The compiler and runtime use the
// same layout, so packed offsets are stable.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_MODEL_WEIGHTPACKER_H
#define LLMCP_MODEL_WEIGHTPACKER_H

#include "llmcp/Model/Safetensors.h"
#include "llmcp/Model/WeightLayout.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace llmcp {

struct PackOptions {
  bool dryRun = false;
  bool verbose = false;
};

/// Packs all tensors referenced by `layout` into `outputPath`.
///
/// `manifestPath` may be empty to skip the manifest. In dry-run mode the
/// tensors are validated but no arena file is written.
llvm::Error packWeights(llvm::StringRef modelDir, const WeightLayout &layout,
                        const TensorMap &tensors, llvm::StringRef outputPath,
                        llvm::StringRef manifestPath,
                        const PackOptions &options = {});

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_MODEL_WEIGHTPACKER_H
