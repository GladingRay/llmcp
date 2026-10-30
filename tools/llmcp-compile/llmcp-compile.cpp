//===- llmcp-compile.cpp - AOT compile the llmcp graph --------------------===//
//
// Lowers the generated llmcp graph through linalg/SCF/LLVM, then emits LLVM IR,
// an object file, or a shared library. The shared library exports the four
// llmcp entry points for a separate runtime driver.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/IR/LLMCP.h"
#include "llmcp/Model/ModelBuilder.h"
#include "llmcp/Model/ModelConfig.h"
#include "llmcp/Model/Safetensors.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace mlir::llmcp;

namespace {

cl::opt<std::string> modelPath("model",
                               cl::desc("Path to the Qwen3.5 model directory"),
                               cl::Required);
cl::opt<std::string> outputPath("out", cl::desc("Output file"), cl::Required);
cl::opt<std::string> emit("emit",
                          cl::desc("Output kind: llvm-ir, object, or shared"),
                          cl::init("shared"));
cl::opt<std::string> llmcpOptPath("llmcp-opt", cl::desc("Path to llmcp-opt"),
                                  cl::init(""));
cl::opt<std::string>
    mlirOptPath("mlir-opt", cl::desc("Path to mlir-opt"),
                cl::init("/opt/homebrew/opt/llvm@20/bin/mlir-opt"));
cl::opt<std::string>
    mlirTranslatePath("mlir-translate", cl::desc("Path to mlir-translate"),
                      cl::init("/opt/homebrew/opt/llvm@20/bin/mlir-translate"));
cl::opt<std::string> clangPath("clang", cl::desc("Path to clang"),
                               cl::init("/opt/homebrew/opt/llvm@20/bin/clang"));
cl::opt<std::string> runnerLibDir("runner-lib-dir",
                                  cl::desc("MLIR runner runtime directory"),
                                  cl::init("/opt/homebrew/opt/llvm@20/lib"));
cl::opt<std::string> optLevel("opt-level", cl::desc("Clang optimization level"),
                              cl::init("2"));
cl::opt<bool> keepTemps("keep-temps", cl::desc("Keep intermediate files"),
                        cl::init(false));

int fail(llvm::Error error) {
  llvm::errs() << "llmcp-compile: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

llvm::Expected<std::string> findTool(llvm::StringRef explicitPath,
                                     llvm::StringRef name,
                                     llvm::StringRef argv0) {
  if (!explicitPath.empty())
    return explicitPath.str();
  llvm::SmallString<256> executable(argv0);
  llvm::sys::fs::make_absolute(executable);
  llvm::SmallString<256> sibling(executable);
  llvm::sys::path::remove_filename(sibling);
  llvm::sys::path::append(sibling, name);
  if (llvm::sys::fs::can_execute(sibling))
    return sibling.str().str();
  auto found = llvm::sys::findProgramByName(name);
  if (!found)
    return llvm::errorCodeToError(found.getError());
  return *found;
}

llvm::Error runTool(llvm::StringRef program, llvm::ArrayRef<std::string> args) {
  std::vector<llvm::StringRef> refs;
  refs.push_back(program);
  for (const std::string &arg : args)
    refs.push_back(arg);
  std::string error;
  int result =
      llvm::sys::ExecuteAndWait(program, refs, std::nullopt, {}, 0, 0, &error);
  if (result != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "command failed: " + program.str() + " (" +
                                       error + ")");
  return llvm::Error::success();
}

std::string tempDir() {
  llvm::SmallString<256> dir;
  llvm::sys::fs::createUniqueDirectory("llmcp-compile", dir);
  return dir.str().str();
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "AOT compile the llmcp graph\n");

  if (emit != "llvm-ir" && emit != "object" && emit != "shared") {
    llvm::errs()
        << "llmcp-compile: --emit must be llvm-ir, object, or shared\n";
    return 1;
  }

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
  if (failed(module->verify()))
    return 1;

  const std::string directory = tempDir();
  const std::string graph = directory + "/graph.mlir";
  const std::string linalg = directory + "/linalg.mlir";
  const std::string llvmMlir = directory + "/llvm.mlir";
  const std::string llvmIr = directory + "/module.ll";

  {
    std::error_code ec;
    llvm::raw_fd_ostream out(graph, ec);
    if (ec)
      return fail(llvm::createStringError(ec, "cannot write graph"));
    module->print(out);
  }

  auto llmcpOptOrErr = findTool(llmcpOptPath, "llmcp-opt", argv[0]);
  if (!llmcpOptOrErr)
    return fail(llmcpOptOrErr.takeError());
  auto mlirOptOrErr = findTool(mlirOptPath, "mlir-opt", argv[0]);
  if (!mlirOptOrErr)
    return fail(mlirOptOrErr.takeError());
  auto mlirTranslateOrErr =
      findTool(mlirTranslatePath, "mlir-translate", argv[0]);
  if (!mlirTranslateOrErr)
    return fail(mlirTranslateOrErr.takeError());
  auto clangOrErr = findTool(clangPath, "clang", argv[0]);
  if (!clangOrErr)
    return fail(clangOrErr.takeError());

  if (llvm::Error error = runTool(
          *llmcpOptOrErr, {graph, "--llmcp-lower-to-linalg", "-o", linalg}))
    return fail(std::move(error));
  if (llvm::Error error = runTool(
          *llmcpOptOrErr, {linalg, "--llmcp-parallelize", "-o", linalg}))
    return fail(std::move(error));
  if (llvm::Error error =
          runTool(*mlirOptOrErr,
                  {linalg,
                   "--linalg-fuse-elementwise-ops",
                   "--one-shot-bufferize=bufferize-function-boundaries=1",
                   "--convert-linalg-to-parallel-loops",
                   "--convert-scf-to-openmp",
                   "--convert-openmp-to-llvm",
                   "--lower-affine",
                   "--convert-scf-to-cf",
                   "--convert-cf-to-llvm",
                   "--convert-arith-to-llvm",
                   "--convert-math-to-llvm",
                   "--convert-vector-to-scf",
                   "--convert-scf-to-cf",
                   "--lower-affine",
                   "--convert-cf-to-llvm",
                   "--convert-vector-to-llvm",
                   "--convert-arith-to-llvm",
                   "--convert-func-to-llvm",
                   "--convert-arith-to-llvm",
                   "--expand-strided-metadata",
                   "--finalize-memref-to-llvm",
                   "--reconcile-unrealized-casts",
                   "--convert-arith-to-llvm",
                   "-o",
                   llvmMlir}))
    return fail(std::move(error));
  if (llvm::Error error = runTool(*mlirTranslateOrErr,
                                  {"--mlir-to-llvmir", llvmMlir, "-o", llvmIr}))
    return fail(std::move(error));

  if (emit == "llvm-ir") {
    std::error_code ec = llvm::sys::fs::copy_file(llvmIr, outputPath);
    if (ec)
      return fail(llvm::errorCodeToError(ec));
  } else {
    std::vector<std::string> args;
    if (emit == "object")
      args = {"-c", llvmIr,    "-O" + optLevel, "-Wno-override-module",
              "-o", outputPath};
    else
      args = {"-shared", llvmIr,    "-O" + optLevel, "-Wno-override-module",
              "-o",      outputPath};
    if (emit == "shared") {
      args.push_back("-L/opt/homebrew/opt/libomp/lib");
      args.push_back("-lomp");
      args.push_back("-Wl,-rpath,/opt/homebrew/opt/libomp/lib");
      args.push_back("-L" + runnerLibDir);
      args.push_back("-lmlir_runner_utils");
      args.push_back("-lmlir_c_runner_utils");
      args.push_back("-Wl,-rpath," + runnerLibDir);
    }
#if defined(__aarch64__)
    args.push_back("-mcpu=native");
#endif
    if (llvm::Error error = runTool(*clangOrErr, args))
      return fail(std::move(error));
  }

  llvm::outs() << "compiled: " << outputPath << " (" << emit << ")\n";
  if (keepTemps)
    llvm::outs() << "temps: " << directory << "\n";
  else
    llvm::sys::fs::remove_directories(directory);
  return 0;
}
