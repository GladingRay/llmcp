//===- llmcp-pack.cpp - Pack Qwen3.5 weights into the runtime arena ------===//

#include "llmcp/Model/ModelConfig.h"
#include "llmcp/Model/Safetensors.h"
#include "llmcp/Model/WeightLayout.h"
#include "llmcp/Model/WeightPacker.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir::llmcp;

namespace {

llvm::cl::opt<std::string>
    modelPath("model", llvm::cl::desc("Path to the Qwen3.5 model directory"),
              llvm::cl::Required);
llvm::cl::opt<std::string>
    outputPath("out", llvm::cl::desc("Output packed arena (weights.bin)"));
llvm::cl::opt<std::string>
    manifestPath("manifest",
                 llvm::cl::desc("Output JSON manifest (default: <out>.json)"));
llvm::cl::opt<bool> dryRun("dry-run",
                           llvm::cl::desc("Validate only, do not write arena"),
                           llvm::cl::init(false));
llvm::cl::opt<bool> verbose("verbose", llvm::cl::desc("Print every tensor"),
                            llvm::cl::init(false));

int fail(llvm::Error error) {
  llvm::errs() << "llmcp-pack: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Pack a Qwen3.5 text checkpoint into the llmcp arena\n");

  if (!dryRun && outputPath.empty()) {
    llvm::errs() << "llmcp-pack: --out is required unless --dry-run is set\n";
    return 1;
  }

  const std::string modelDir = modelPath;
  auto configOrErr = ModelConfig::load(modelDir);
  if (!configOrErr)
    return fail(configOrErr.takeError());
  const ModelConfig &config = *configOrErr;

  auto tensorsOrErr = readModelTensors(modelDir);
  if (!tensorsOrErr)
    return fail(tensorsOrErr.takeError());
  TensorMap tensors = std::move(*tensorsOrErr);

  auto layoutOrErr = WeightLayout::build(config, tensors);
  if (!layoutOrErr)
    return fail(layoutOrErr.takeError());
  const WeightLayout &layout = *layoutOrErr;

  std::string manifest = manifestPath;
  if (manifest.empty() && !outputPath.empty())
    manifest = outputPath + ".json";

  PackOptions options;
  options.dryRun = dryRun;
  options.verbose = verbose;
  if (llvm::Error error =
          packWeights(modelDir, layout, tensors, outputPath, manifest, options))
    return fail(std::move(error));

  uint64_t totalTensorBytes = 0;
  for (const WeightEntry &entry : layout.getEntries())
    totalTensorBytes += entry.size;
  llvm::outs() << (dryRun ? "validated" : "packed") << " "
               << layout.getEntries().size() << " tensors, " << totalTensorBytes
               << " tensor bytes, arena " << layout.getTotalBytes() << " bytes";
  if (!dryRun)
    llvm::outs() << " -> " << outputPath;
  llvm::outs() << "\n";
  if (!manifest.empty())
    llvm::outs() << "manifest: " << manifest << "\n";
  return 0;
}
