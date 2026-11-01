//===- llmcp-inspect.cpp - Inspect a Qwen3.5 checkpoint -------------------===//
//
// Prints model configuration, tensor metadata, and arena layout.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Model/ModelConfig.h"
#include "llmcp/Model/Safetensors.h"
#include "llmcp/Model/WeightLayout.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir::llmcp;

namespace {

llvm::cl::opt<std::string>
    modelPath("model", llvm::cl::desc("Path to the Qwen3.5 model directory"),
              llvm::cl::Required);

llvm::cl::opt<bool> verbose("verbose",
                            llvm::cl::desc("Print every tensor in the layout"),
                            llvm::cl::init(false));

int fail(llvm::Error error) {
  llvm::errs() << "llmcp-inspect: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Inspect a Qwen3.5 model directory\n");

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

  uint64_t checkpointBytes = 0;
  for (const auto &entry : tensors)
    checkpointBytes += entry.second.dataSize;

  llvm::outs() << "model: " << modelDir << "\n";
  llvm::outs() << "layers: " << config.getNumLayers() << " ("
               << config.getNumLinearLayers() << " linear, "
               << config.getNumFullLayers() << " full)\n";
  llvm::outs() << "tensors: " << tensors.size() << "\n";
  llvm::outs() << "text hidden: " << config.text.hiddenSize
               << ", intermediate: " << config.text.intermediateSize
               << ", vocab: " << config.text.vocabSize << "\n";
  llvm::outs() << "q/kv heads: " << config.text.numAttentionHeads << "/"
               << config.text.numKeyValueHeads
               << ", head dim: " << config.text.headDim
               << ", rotary dim: " << config.getRotaryDim() << "\n";
  llvm::outs() << "checkpoint bytes: " << checkpointBytes << "\n";
  llvm::outs() << "arena bytes: " << layout.getTotalBytes() << " (header "
               << layout.getHeaderBytes() << ", group stride "
               << layout.getGroupStride() << ", groups "
               << layout.getNumGroups() << ")\n";
  llvm::outs() << "embedding offset: " << layout.getEmbeddingOffset()
               << ", final norm offset: " << layout.getFinalNormOffset()
               << "\n";

  if (verbose) {
    llvm::outs() << "tensors in layout:\n";
    for (const WeightEntry &entry : layout.getEntries())
      llvm::outs() << "  " << entry.offset << " +" << entry.size << " "
                   << entry.dtype << " " << entry.name << "\n";
  }

  return 0;
}
