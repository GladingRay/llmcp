//===- llmcp-benchmark.cpp - Benchmark the AOT model ---------------------===//
//
// Measures prefill and decode throughput of llmcp_model_aot.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "llmcp/Runtime/AotRuntime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace llvm;
using llmcp::CacheLineAlignedVector;
using namespace mlir::llmcp::runtime;

namespace {

cl::opt<std::string> weightsPath("weights", cl::desc("Packed arena"),
                                 cl::init("build/weights.bin"));
cl::opt<std::string> manifestPath("manifest", cl::desc("Arena manifest"),
                                  cl::init("build/weights.json"));
cl::opt<std::string> aotPath("aot", cl::desc("AOT shared library"),
                             cl::Required);
cl::opt<int64_t> prefillTokens("prefill-tokens",
                               cl::desc("Number of prefill tokens"),
                               cl::init(32));
cl::opt<int64_t> decodeTokens("decode-tokens",
                              cl::desc("Number of decode tokens"), cl::init(8));
cl::opt<int64_t> tokenId("token", cl::desc("Synthetic token id"),
                         cl::init(9419));
cl::opt<bool> verbose("verbose", cl::desc("Print per-phase detail"),
                      cl::init(false));

int64_t argmax(const CacheLineAlignedVector<float> &values) {
  return static_cast<int64_t>(std::max_element(values.begin(), values.end()) -
                              values.begin());
}

double seconds(std::chrono::steady_clock::time_point begin,
               std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "llmcp-benchmark\n");

  if (prefillTokens <= 0 || decodeTokens < 0) {
    errs() << "llmcp-benchmark: prefill-tokens must be > 0 and "
              "decode-tokens >= 0\n";
    return 1;
  }

  Arena arena;
  if (Error error = arena.load(weightsPath, manifestPath)) {
    errs() << "llmcp-benchmark: " << toString(std::move(error)) << "\n";
    return 1;
  }

  AotRuntime runtime;
  if (!runtime.init(aotPath, prefillTokens + decodeTokens + 8))
    return 1;

  std::vector<int64_t> prompt(static_cast<size_t>(prefillTokens), tokenId);
  CacheLineAlignedVector<float> logits;

  auto prefillBegin = std::chrono::steady_clock::now();
  if (!runtime.forward(arena, prompt, 0, prefillTokens, logits))
    return 1;
  auto prefillEnd = std::chrono::steady_clock::now();
  double prefillSeconds = seconds(prefillBegin, prefillEnd);

  std::vector<int64_t> generated;
  generated.reserve(static_cast<size_t>(decodeTokens));
  CacheLineAlignedVector<float> lastLogits(logits.end() - 248320, logits.end());
  auto decodeBegin = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < decodeTokens; ++i) {
    int64_t next = argmax(lastLogits);
    generated.push_back(next);
    std::vector<int64_t> one{next};
    int64_t position = prefillTokens + i;
    if (!runtime.forward(arena, one, position, position + 1, logits))
      return 1;
    lastLogits.assign(logits.end() - 248320, logits.end());
  }
  auto decodeEnd = std::chrono::steady_clock::now();
  double decodeSeconds = seconds(decodeBegin, decodeEnd);
  double totalSeconds = prefillSeconds + decodeSeconds;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "prefill_tokens: " << prefillTokens << "\n";
  std::cout << "decode_tokens: " << decodeTokens << "\n";
  std::cout << "prefill_seconds: " << prefillSeconds << "\n";
  std::cout << "decode_seconds: " << decodeSeconds << "\n";
  std::cout << "total_seconds: " << totalSeconds << "\n";
  std::cout << "prefill_tokens_per_second: " << (prefillTokens / prefillSeconds)
            << "\n";
  std::cout << "decode_tokens_per_second: "
            << (decodeSeconds > 0 ? decodeTokens / decodeSeconds : 0.0) << "\n";
  if (verbose) {
    std::cout << "generated_tokens:";
    for (int64_t id : generated)
      std::cout << " " << id;
    std::cout << "\n";
  }
  return 0;
}
