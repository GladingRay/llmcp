//===- llmcp-matmul-benchmark.cpp - AOT matmul microbenchmark -------------===//
//
// Generates one llmcp.linear op, lowers it step by step to a shared library,
// then times the AOT function and reports GFLOPS.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include "llmcp/Runtime/AotRuntime.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir::llmcp::runtime;

namespace {

cl::opt<int64_t> mDim("m", cl::desc("Output rows"), cl::init(64));
cl::opt<int64_t> nDim("n", cl::desc("Output columns"), cl::init(64));
cl::opt<int64_t> kDim("k", cl::desc("Reduction dimension"), cl::init(64));
cl::opt<int64_t> iterations("iters", cl::desc("Timed iterations"),
                            cl::init(20));
cl::opt<bool> verify("verify", cl::desc("Check the first output element"),
                     cl::init(true));
cl::opt<bool> keepTemps("keep-temps", cl::desc("Keep intermediate files"),
                        cl::init(false));
cl::opt<bool> verbose("verbose", cl::desc("Print each lowering stage"),
                      cl::init(false));
cl::opt<int> threads("threads", cl::desc("OpenMP threads (0 = runtime default)"),
                     cl::init(0));
cl::opt<bool> noParallel("no-parallel", cl::desc("Disable OpenMP parallelization"),
                         cl::init(false));
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


int fail(StringRef message) {
  errs() << "llmcp-matmul-benchmark: " << message << "\n";
  return 1;
}

Expected<std::string> findTool(StringRef explicitPath, StringRef name,
                               StringRef argv0) {
  if (!explicitPath.empty())
    return explicitPath.str();
  SmallString<256> executable(argv0);
  sys::fs::make_absolute(executable);
  SmallString<256> sibling(executable);
  sys::path::remove_filename(sibling);
  sys::path::append(sibling, name);
  if (sys::fs::can_execute(sibling))
    return sibling.str().str();
  auto found = sys::findProgramByName(name);
  if (!found)
    return errorCodeToError(found.getError());
  return *found;
}

Error runTool(StringRef program, ArrayRef<std::string> args) {
  std::vector<StringRef> refs;
  refs.push_back(program);
  for (const std::string &arg : args)
    refs.push_back(arg);
  std::string error;
  int result =
      sys::ExecuteAndWait(program, refs, std::nullopt, {}, 0, 0, &error);
  if (result != 0)
    return createStringError(inconvertibleErrorCode(),
                             "command failed: " + program.str() + " (" + error +
                                 ")");
  return Error::success();
}

std::string generateMlir(int64_t m, int64_t n, int64_t k) {
  std::string ir;
  raw_string_ostream os(ir);
  os << "module {\n";
  os << "  func.func @linear_bench(%a: memref<" << m << "x" << k
     << "xbf16>, %b: memref<" << n << "x" << k << "xbf16>, %c: memref<" << m
     << "x" << n << "xbf16>) attributes {llvm.emit_c_interface} {\n";
  os << "    %c0 = arith.constant 0 : index\n";
  os << "    %c1 = arith.constant 1 : index\n";
  os << "    %cm = arith.constant " << m << " : index\n";
  os << "    %cn = arith.constant " << n << " : index\n";
  os << "    %at = bufferization.to_tensor %a restrict : memref<" << m << "x"
     << k << "xbf16> to tensor<" << m << "x" << k << "xbf16>\n";
  os << "    %out = llmcp.linear %at, %b : tensor<" << m << "x" << k
     << "xbf16>, memref<" << n << "x" << k << "xbf16> -> tensor<" << m << "x"
     << n << "xbf16>\n";
  os << "    scf.for %i = %c0 to %cm step %c1 {\n";
  os << "      scf.for %j = %c0 to %cn step %c1 {\n";
  os << "        %v = tensor.extract %out[%i, %j] : tensor<" << m << "x" << n
     << "xbf16>\n";
  os << "        memref.store %v, %c[%i, %j] : memref<" << m << "x" << n
     << "xbf16>\n";
  os << "      }\n    }\n    return\n  }\n}\n";
  return os.str();
}

float bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "llmcp-matmul-benchmark\n");
  if (mDim <= 0 || nDim <= 0 || kDim <= 0 || iterations <= 0)
    return fail("m, n, k and iters must be positive");

  auto llmcpOptOrErr = findTool(llmcpOptPath, "llmcp-opt", argv[0]);
  if (!llmcpOptOrErr)
    return fail(toString(llmcpOptOrErr.takeError()));
  auto mlirOptOrErr = findTool(mlirOptPath, "mlir-opt", argv[0]);
  if (!mlirOptOrErr)
    return fail(toString(mlirOptOrErr.takeError()));
  auto translateOrErr = findTool(mlirTranslatePath, "mlir-translate", argv[0]);
  if (!translateOrErr)
    return fail(toString(translateOrErr.takeError()));
  auto clangOrErr = findTool(clangPath, "clang", argv[0]);
  if (!clangOrErr)
    return fail(toString(clangOrErr.takeError()));

  SmallString<256> directory;
  if (std::error_code ec =
          sys::fs::createUniqueDirectory("llmcp-matmul-benchmark", directory))
    return fail(ec.message());
  std::string source = (directory + "/input.mlir").str();
  std::string linalg = (directory + "/linalg.mlir").str();
  std::string parallel = (directory + "/parallel.mlir").str();
  std::string llvmMlir = (directory + "/llvm.mlir").str();
  std::string llvmIr = (directory + "/module.ll").str();
  std::string dylib = (directory + "/liblinear.dylib").str();

  {
    std::ofstream out(source);
    out << generateMlir(mDim, nDim, kDim);
  }

  if (verbose)
    std::cout << "stage 1: llmcp.linear -> linalg\n";
  if (Error error = runTool(*llmcpOptOrErr,
                            {source, "--llmcp-lower-to-linalg", "-o", linalg}))
    return fail(toString(std::move(error)));
  if (verbose)
    std::cout << "stage 2: linalg -> parallel scf\n";
  std::string inputForLowering = linalg;
  if (!noParallel) {
    if (Error error = runTool(
            *llmcpOptOrErr,
            {linalg, "--llmcp-parallelize", "-o", parallel}))
      return fail(toString(std::move(error)));
    inputForLowering = parallel;
  }

  std::vector<std::string> mlirArgs = {
      inputForLowering, "--one-shot-bufferize=bufferize-function-boundaries=1"};
  if (noParallel)
    mlirArgs.push_back("--convert-linalg-to-loops");
  else {
    mlirArgs.push_back("--convert-linalg-to-parallel-loops");
    mlirArgs.push_back("--convert-scf-to-openmp");
    mlirArgs.push_back("--convert-openmp-to-llvm");
  }
  for (const char *arg :
       {"--lower-affine", "--convert-scf-to-cf", "--convert-cf-to-llvm",
        "--convert-arith-to-llvm", "--convert-math-to-llvm",
        "--convert-vector-to-scf", "--convert-scf-to-cf", "--lower-affine",
        "--convert-cf-to-llvm", "--convert-vector-to-llvm",
        "--convert-arith-to-llvm", "--convert-func-to-llvm",
        "--convert-arith-to-llvm", "--expand-strided-metadata",
        "--finalize-memref-to-llvm", "--reconcile-unrealized-casts",
        "--convert-arith-to-llvm"})
    mlirArgs.push_back(arg);
  mlirArgs.push_back("-o");
  mlirArgs.push_back(llvmMlir);
  if (Error error = runTool(*mlirOptOrErr, mlirArgs))
    return fail(toString(std::move(error)));
  if (verbose)
    std::cout << "stage 3: LLVM dialect -> LLVM IR\n";
  if (Error error = runTool(*translateOrErr,
                            {"--mlir-to-llvmir", llvmMlir, "-o", llvmIr}))
    return fail(toString(std::move(error)));
  if (verbose)
    std::cout << "stage 4: LLVM IR -> shared library\n";
  std::vector<std::string> clangArgs = {
      "-shared", llvmIr, "-O2", "-L/opt/homebrew/opt/llvm@20/lib",
      "-lmlir_runner_utils", "-lmlir_c_runner_utils",
      "-Wl,-rpath,/opt/homebrew/opt/llvm@20/lib"};
  if (!noParallel) {
    clangArgs.push_back("-L/opt/homebrew/opt/libomp/lib");
    clangArgs.push_back("-lomp");
    clangArgs.push_back("-Wl,-rpath,/opt/homebrew/opt/libomp/lib");
  }
  clangArgs.push_back("-o");
  clangArgs.push_back(dylib);
#if defined(__aarch64__)
  clangArgs.push_back("-mcpu=native");
#endif
  if (Error error = runTool(*clangOrErr, clangArgs))
    return fail(toString(std::move(error)));

  if (threads > 0)
    setenv("OMP_NUM_THREADS", std::to_string(threads).c_str(), 1);
  void *handle = dlopen(dylib.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    return fail(dlerror());
  using Fn = void (*)(void *, void *, void *);
  auto fn = reinterpret_cast<Fn>(dlsym(handle, "_mlir_ciface_linear_bench"));
  if (!fn)
    return fail(dlerror());

  std::vector<uint16_t> a(static_cast<size_t>(mDim * kDim), 0x3f80);
  std::vector<uint16_t> b(static_cast<size_t>(nDim * kDim), 0x3f80);
  std::vector<uint16_t> c(static_cast<size_t>(mDim * nDim), 0);
  MemRef<uint16_t, 2> ra{a.data(), a.data(), 0, {mDim, kDim}, {kDim, 1}};
  MemRef<uint16_t, 2> rb{b.data(), b.data(), 0, {nDim, kDim}, {kDim, 1}};
  MemRef<uint16_t, 2> rc{c.data(), c.data(), 0, {mDim, nDim}, {nDim, 1}};

  fn(&ra, &rb, &rc); // warmup / first run
  auto begin = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < iterations; ++i)
    fn(&ra, &rb, &rc);
  auto end = std::chrono::steady_clock::now();
  double seconds = std::chrono::duration<double>(end - begin).count();
  double flops = 2.0 * static_cast<double>(mDim) * nDim * kDim * iterations;
  double gflops = flops / seconds / 1.0e9;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "m: " << mDim << "\n";
  std::cout << "n: " << nDim << "\n";
  std::cout << "k: " << kDim << "\n";
  std::cout << "iters: " << iterations << "\n";
  if (threads > 0)
    std::cout << "threads: " << threads << "\n";
  std::cout << "seconds: " << seconds << "\n";
  std::cout << "gflops: " << gflops << "\n";
  if (verify) {
    float expected = static_cast<float>(kDim);
    float actual = bf16ToFloat(c[0]);
    std::cout << "first: " << actual << " (expected " << expected << ")\n";
  }

  dlclose(handle);
  if (keepTemps)
    std::cout << "temps: " << directory.str().str() << "\n";
  else
    sys::fs::remove_directories(directory);
  return 0;
}
