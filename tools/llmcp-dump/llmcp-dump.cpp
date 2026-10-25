//===- llmcp-dump.cpp - Dump the complete llmcp compute graph -------------===//
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/IR/LLMCP.h"
#include "llmcp/Model/ModelBuilder.h"
#include "llmcp/Model/ModelConfig.h"
#include "llmcp/Model/Safetensors.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>
#include <utility>

using namespace mlir;
using namespace mlir::llmcp;

namespace {

llvm::cl::opt<std::string>
    modelPath("model", llvm::cl::desc("Path to the Qwen3.5 model directory"),
              llvm::cl::Required);

llvm::cl::opt<std::string> outputPath(
    "o", llvm::cl::desc("Write the MLIR module to this file (default: stdout)"),
    llvm::cl::value_desc("path"));

llvm::cl::opt<bool> verifyModule(
    "verify-module",
    llvm::cl::desc("Verify the generated module before writing it"),
    llvm::cl::init(true));

llvm::cl::opt<bool>
    printStats("print-stats",
               llvm::cl::desc("Print graph statistics to stderr"),
               llvm::cl::init(false));

int fail(llvm::Error error) {
  llvm::errs() << "llmcp-dump: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Dump the llmcp compute graph\n");

  auto configOrErr = ModelConfig::load(modelPath);
  if (!configOrErr)
    return fail(configOrErr.takeError());

  auto tensorsOrErr = readModelTensors(modelPath);
  if (!tensorsOrErr)
    return fail(tensorsOrErr.takeError());

  MLIRContext context;
  auto moduleOrErr = ModelBuilder::build(*configOrErr, *tensorsOrErr, &context);
  if (!moduleOrErr)
    return fail(moduleOrErr.takeError());
  OwningOpRef<ModuleOp> module = std::move(*moduleOrErr);

  if (verifyModule && failed(module->verify()))
    return 1;

  if (printStats) {
    size_t numLinear = 0;
    size_t numRmsNorm = 0;
    size_t numDelta = 0;
    size_t numAttention = 0;
    size_t numConv = 0;
    size_t numGatedNorm = 0;
    size_t numEmbed = 0;
    size_t numLmHead = 0;
    size_t numSiluMul = 0;
    size_t numAdd = 0;
    size_t numViews = 0;
    module->walk([&](LinearOp) { ++numLinear; });
    module->walk([&](RmsNormOp) { ++numRmsNorm; });
    module->walk([&](GatedDeltaNetOp) { ++numDelta; });
    module->walk([&](AttentionOp) { ++numAttention; });
    module->walk([&](CausalConv1dOp) { ++numConv; });
    module->walk([&](GatedRmsNormOp) { ++numGatedNorm; });
    module->walk([&](EmbeddingOp) { ++numEmbed; });
    module->walk([&](LmHeadOp) { ++numLmHead; });
    module->walk([&](SiluMulOp) { ++numSiluMul; });
    module->walk([&](AddOp) { ++numAdd; });
    module->walk([&](memref::ViewOp) { ++numViews; });

    llvm::errs() << "functions: " << module->getBody()->getOperations().size()
                 << "\n";
    llvm::errs() << "linear: " << numLinear << ", rms_norm: " << numRmsNorm
                 << ", delta: " << numDelta << ", attention: " << numAttention
                 << ", conv: " << numConv << ", gated_norm: " << numGatedNorm
                 << ", embedding: " << numEmbed << ", lm_head: " << numLmHead
                 << ", silu_mul: " << numSiluMul << ", add: " << numAdd
                 << ", weight_views: " << numViews << "\n";
  }

  if (outputPath.empty()) {
    module->print(llvm::outs());
    return 0;
  }

  std::error_code ec;
  llvm::raw_fd_ostream output(outputPath, ec);
  if (ec)
    return fail(llvm::createStringError(ec, "cannot open output file"));
  module->print(output);
  return 0;
}
