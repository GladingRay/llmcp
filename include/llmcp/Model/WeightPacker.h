//===- WeightPacker.h - Pack safetensors into the runtime arena -*- C++ -*-===//
//
// Copies model tensors and their JSON manifest into one weight container.
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
/// The manifest is embedded in the output. In dry-run mode the tensors are
/// validated but no file is written.
llvm::Error packWeights(llvm::StringRef modelDir, const WeightLayout &layout,
                        const TensorMap &tensors, llvm::StringRef outputPath,
                        const PackOptions &options = {});

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_MODEL_WEIGHTPACKER_H
