//===- llmcp-inspect.cpp - Inspect a Qwen3.5 checkpoint
//--------------------===//
//
// Reads only configuration and safetensors headers; tensor data is not read.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Model/ModelConfig.h"
#include "llmcp/Model/Safetensors.h"
#include "llmcp/Model/WeightLayout.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

using namespace mlir::llmcp;
using llvm::StringRef;

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

llvm::Expected<std::vector<std::string>> collectShards(StringRef modelDir) {
  const std::string indexPath =
      (llvm::Twine(modelDir) + "/model.safetensors.index.json").str();
  auto indexOrErr = readSafetensorsIndex(indexPath);
  if (indexOrErr) {
    llvm::StringSet<> seen;
    std::vector<std::string> shards;
    for (const auto &entry : indexOrErr->weightMap) {
      StringRef file = entry.second;
      if (seen.insert(file).second)
        shards.push_back(file.str());
    }
    std::sort(shards.begin(), shards.end());
    return shards;
  }
  llvm::consumeError(indexOrErr.takeError());

  std::vector<std::string> shards;
  std::error_code ec;
  for (const auto &entry :
       std::filesystem::directory_iterator(modelDir.str(), ec)) {
    if (!entry.is_regular_file())
      continue;
    const std::string name = entry.path().filename().string();
    if (StringRef(name).ends_with(".safetensors"))
      shards.push_back(name);
  }
  if (ec)
    return llvm::createStringError(ec, "cannot scan model directory");
  std::sort(shards.begin(), shards.end());
  return shards;
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

  auto shardsOrErr = collectShards(modelDir);
  if (!shardsOrErr)
    return fail(shardsOrErr.takeError());
  if (shardsOrErr->empty())
    return fail(llvm::createStringError(llvm::inconvertibleErrorCode(),
                                        "no safetensors shards found"));

  TensorMap tensors;
  for (const std::string &shard : *shardsOrErr) {
    const std::string path = (llvm::Twine(modelDir) + "/" + shard).str();
    auto shardTensors = readSafetensorsFile(path);
    if (!shardTensors)
      return fail(shardTensors.takeError());
    for (const auto &entry : *shardTensors) {
      auto inserted = tensors.try_emplace(entry.first(), entry.second);
      if (!inserted.second)
        return fail(llvm::createStringError(llvm::inconvertibleErrorCode(),
                                            llvm::Twine("duplicate tensor '") +
                                                entry.first() + "'"));
    }
  }

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
