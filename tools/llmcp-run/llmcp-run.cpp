//===- llmcp-run.cpp - Minimal C++ runner for packed Qwen3.5 weights -----===//
//
// Loads the arena produced by llmcp-pack and executes one-token prefill using
// the same math as the llmcp ops. This is the first executable baseline before
// the AOT lowering is wired in.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

using namespace llvm;

namespace {

cl::opt<std::string> weightsPath("weights", cl::desc("Packed arena"),
                                 cl::init("build/weights.bin"));
cl::opt<std::string> manifestPath("manifest", cl::desc("Arena manifest"),
                                  cl::init("build/weights.json"));
cl::opt<std::string> goldenPath("golden", cl::desc("Optional f32 logits file"));
cl::opt<long long> tokenId("tokens", cl::desc("Token ID for one-token prefill"),
                           cl::init(9419));
cl::opt<std::string>
    prompt("text", cl::desc("Prompt for the minimal tokenizer"), cl::init(""));
cl::opt<std::string> tokenizerPath(
    "tokenizer", cl::desc("HuggingFace tokenizer.json"),
    cl::init("/Users/guorui/Work/models/qwen3.5-4b/tokenizer.json"));
cl::opt<bool> dumpLogits("dump-logits", cl::desc("Write f32 logits to stdout"),
                         cl::init(false));
cl::opt<bool> debug("debug", cl::desc("Print per-layer stats"),
                    cl::init(false));
cl::opt<bool> dumpCosSin("dump-cos-sin",
                         cl::desc("Print generated cos/sin values"),
                         cl::init(false));
cl::opt<long long> position("position", cl::desc("RoPE position"), cl::init(0));

struct Shape {
  std::vector<int64_t> dims;
  int64_t elements() const {
    int64_t n = 1;
    for (int64_t d : dims)
      n *= d;
    return n;
  }
};

struct Weight {
  const uint8_t *data = nullptr;
  Shape shape;
  std::string dtype;
};

struct Tokenizer {
  std::unordered_map<std::string, int64_t> vocab;
  std::unordered_map<std::string, int64_t> ranks;

  Error load(StringRef path) {
    auto buffer = MemoryBuffer::getFile(path);
    if (!buffer)
      return errorCodeToError(buffer.getError());
    auto json = json::parse((*buffer)->getBuffer());
    if (!json)
      return json.takeError();
    auto *root = json->getAsObject();
    auto *model = root ? root->getObject("model") : nullptr;
    auto *vocabObject = model ? model->getObject("vocab") : nullptr;
    if (!vocabObject)
      return createStringError(inconvertibleErrorCode(),
                               "tokenizer vocab missing");
    for (const auto &item : *vocabObject) {
      int64_t id = item.second.getAsInteger().value_or(-1);
      if (id >= 0)
        vocab[item.first.str()] = id;
    }
    auto *merges = model->getArray("merges");
    if (!merges)
      return createStringError(inconvertibleErrorCode(),
                               "tokenizer merges missing");
    int64_t rank = 0;
    for (const json::Value &value : *merges) {
      auto merge = value.getAsString();
      if (!merge)
        continue;
      size_t space = merge->find(' ');
      if (space == StringRef::npos)
        continue;
      std::string key = merge->substr(0, space).str() + '\x1f' +
                        merge->substr(space + 1).str();
      ranks[key] = rank++;
    }
    return Error::success();
  }

  Expected<std::vector<int64_t>> encode(StringRef text) const {
    std::vector<std::string> pieces;
    std::string current;
    for (char c : text) {
      if (c == ' ' || c == '\t' || c == '\n') {
        if (!current.empty()) {
          pieces.push_back(current);
          current.clear();
        }
      } else {
        current.push_back(c);
      }
    }
    if (!current.empty())
      pieces.push_back(current);

    std::vector<int64_t> ids;
    for (const std::string &piece : pieces) {
      std::vector<std::string> symbols;
      for (char c : piece)
        symbols.emplace_back(1, c);
      while (symbols.size() > 1) {
        int64_t bestRank = INT64_MAX;
        size_t bestIndex = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
          auto it = ranks.find(symbols[i] + '\x1f' + symbols[i + 1]);
          if (it != ranks.end() && it->second < bestRank) {
            bestRank = it->second;
            bestIndex = i;
          }
        }
        if (bestRank == INT64_MAX)
          break;
        std::vector<std::string> merged;
        for (size_t i = 0; i < symbols.size();) {
          if (i + 1 < symbols.size() && i == bestIndex) {
            merged.push_back(symbols[i] + symbols[i + 1]);
            i += 2;
          } else {
            merged.push_back(symbols[i]);
            ++i;
          }
        }
        symbols = std::move(merged);
      }
      for (const std::string &symbol : symbols) {
        auto it = vocab.find(symbol);
        if (it == vocab.end())
          return createStringError(inconvertibleErrorCode(),
                                   "token not in vocab: " + symbol);
        ids.push_back(it->second);
      }
    }
    return ids;
  }
};

struct Arena {
  int fd = -1;
  const uint8_t *base = nullptr;
  uint64_t size = 0;
  std::unordered_map<std::string, Weight> weights;

  ~Arena() {
    if (base)
      munmap(const_cast<uint8_t *>(base), size);
    if (fd >= 0)
      close(fd);
  }

  Error load(StringRef bin, StringRef manifest) {
    fd = open(bin.str().c_str(), O_RDONLY);
    if (fd < 0)
      return errorCodeToError(std::error_code(errno, std::generic_category()));
    struct stat st;
    if (fstat(fd, &st) != 0)
      return errorCodeToError(std::error_code(errno, std::generic_category()));
    size = static_cast<uint64_t>(st.st_size);
    base = static_cast<const uint8_t *>(
        mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (base == MAP_FAILED)
      return errorCodeToError(std::error_code(errno, std::generic_category()));

    auto buffer = MemoryBuffer::getFile(manifest);
    if (!buffer)
      return errorCodeToError(buffer.getError());
    auto json = json::parse((*buffer)->getBuffer());
    if (!json)
      return json.takeError();
    auto *object = json->getAsObject();
    if (!object)
      return createStringError(inconvertibleErrorCode(),
                               "manifest is not object");
    auto *tensors = object->getArray("tensors");
    if (!tensors)
      return createStringError(inconvertibleErrorCode(),
                               "manifest has no tensors");
    for (const json::Value &value : *tensors) {
      auto *entry = value.getAsObject();
      if (!entry)
        continue;
      Weight weight;
      std::string name = entry->getString("name")->str();
      uint64_t offset = entry->getInteger("offset").value_or(0);
      uint64_t bytes = entry->getInteger("size").value_or(0);
      weight.dtype = entry->getString("dtype")->str();
      for (const json::Value &dim : *entry->getArray("shape"))
        weight.shape.dims.push_back(dim.getAsInteger().value_or(0));
      if (offset + bytes > size)
        return createStringError(inconvertibleErrorCode(),
                                 "tensor outside arena: " + name);
      weight.data = base + offset;
      weights[name] = std::move(weight);
    }
    return Error::success();
  }

  const Weight &get(StringRef name) const {
    auto it = weights.find(name.str());
    assert(it != weights.end() && "missing weight");
    return it->second;
  }
};

float bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t floatToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t rounding = 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

float readBf16(const uint8_t *base, int64_t index) {
  uint16_t value;
  std::memcpy(&value, base + index * 2, sizeof(value));
  return bf16ToFloat(value);
}

float readScalar(const Weight &weight, int64_t index) {
  if (weight.dtype == "F32") {
    float value;
    std::memcpy(&value, weight.data + index * 4, sizeof(value));
    return value;
  }
  return readBf16(weight.data, index);
}

std::vector<float> matvec(const Weight &weight, const std::vector<float> &x) {
  int64_t out = weight.shape.dims[0];
  int64_t in = weight.shape.dims[1];
  std::vector<float> y(static_cast<size_t>(out), 0.0f);
  constexpr int64_t kBlock = 512;
#ifdef __APPLE__
  for (int64_t start = 0; start < out; start += kBlock) {
    int64_t rows = std::min(kBlock, out - start);
    std::vector<float> block(static_cast<size_t>(rows * in));
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t c = 0; c < in; ++c)
        block[r * in + c] = readBf16(weight.data, (start + r) * in + c);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(rows),
                static_cast<int>(in), 1.0f, block.data(), static_cast<int>(in),
                x.data(), 1, 0.0f, y.data() + start, 1);
  }
#else
  for (int64_t r = 0; r < out; ++r) {
    float sum = 0.0f;
    for (int64_t c = 0; c < in; ++c)
      sum += readBf16(weight.data, r * in + c) * x[c];
    y[r] = sum;
  }
#endif
  for (float &value : y)
    value = bf16ToFloat(floatToBf16(value));
  return y;
}

std::vector<float> rmsNorm(const std::vector<float> &x, const Weight &weight,
                           float eps, bool zeroCentered = true) {
  float sum = 0.0f;
  for (float value : x)
    sum += value * value;
  float rstd = 1.0f / std::sqrt(sum / static_cast<float>(x.size()) + eps);
  std::vector<float> y(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    float w = readBf16(weight.data, static_cast<int64_t>(i));
    y[i] = x[i] * rstd * (zeroCentered ? (1.0f + w) : w);
  }
  for (float &value : y)
    value = bf16ToFloat(floatToBf16(value));
  return y;
}

std::vector<float> linear(const std::vector<float> &x, const Weight &weight) {
  return matvec(weight, x);
}

float maxAbs(const std::vector<float> &v) {
  float m = 0.0f;
  for (float x : v)
    m = std::max(m, std::fabs(x));
  return m;
}

std::vector<float> add(const std::vector<float> &a,
                       const std::vector<float> &b) {
  std::vector<float> y(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    y[i] = bf16ToFloat(floatToBf16(a[i] + b[i]));
  return y;
}

std::vector<float> mlp(const std::vector<float> &x, const Arena &arena,
                       StringRef prefix) {
  std::string base = prefix.str();
  auto gate = linear(x, arena.get(base + ".mlp.gate_proj.weight"));
  auto up = linear(x, arena.get(base + ".mlp.up_proj.weight"));
  std::vector<float> act(gate.size());
  for (size_t i = 0; i < gate.size(); ++i) {
    float silu = gate[i] / (1.0f + std::exp(-gate[i]));
    act[i] = bf16ToFloat(floatToBf16(silu * up[i]));
  }
  return linear(act, arena.get(base + ".mlp.down_proj.weight"));
}

std::vector<float> fullAttention(const std::vector<float> &x,
                                 const Arena &arena, StringRef prefix,
                                 int64_t heads, int64_t kvHeads,
                                 int64_t headDim) {
  std::string base = prefix.str();
  auto qg = linear(x, arena.get(base + ".self_attn.q_proj.weight"));
  auto v = linear(x, arena.get(base + ".self_attn.v_proj.weight"));
  std::vector<float> context(static_cast<size_t>(heads * headDim));
  int64_t group = heads / kvHeads;
  for (int64_t h = 0; h < heads; ++h) {
    int64_t kv = h / group;
    for (int64_t d = 0; d < headDim; ++d) {
      float gate = qg[h * 2 * headDim + headDim + d];
      float sigmoid = 1.0f / (1.0f + std::exp(-gate));
      context[h * headDim + d] =
          bf16ToFloat(floatToBf16(v[kv * headDim + d] * sigmoid));
    }
  }
  return linear(context, arena.get(base + ".self_attn.o_proj.weight"));
}

float dot(const std::vector<float> &a, size_t aOff, const std::vector<float> &b,
          size_t bOff, int64_t n) {
  float sum = 0.0f;
  for (int64_t i = 0; i < n; ++i)
    sum += a[aOff + i] * b[bOff + i];
  return sum;
}

std::vector<float> linearAttention(const std::vector<float> &x,
                                   const Arena &arena, StringRef prefix,
                                   int64_t numK, int64_t numV, int64_t headK,
                                   int64_t headV, int64_t kernel, float eps) {
  std::string base = prefix.str();
  auto qkv = linear(x, arena.get(base + ".linear_attn.in_proj_qkv.weight"));
  auto z = linear(x, arena.get(base + ".linear_attn.in_proj_z.weight"));
  auto a = linear(x, arena.get(base + ".linear_attn.in_proj_a.weight"));
  auto b = linear(x, arena.get(base + ".linear_attn.in_proj_b.weight"));
  if (debug)
    outs() << "layer0 raw qkv " << maxAbs(qkv) << " z " << maxAbs(z) << " a "
           << maxAbs(a) << " b " << maxAbs(b) << "\n";
  const Weight &conv = arena.get(base + ".linear_attn.conv1d.weight");
  // Zero conv state, seq_len=1: only the last tap contributes.
  for (int64_t i = 0; i < static_cast<int64_t>(qkv.size()); ++i) {
    float tap = readBf16(conv.data, i * kernel + (kernel - 1));
    qkv[i] = bf16ToFloat(floatToBf16(qkv[i] * tap));
    qkv[i] = qkv[i] / (1.0f + std::exp(-qkv[i]));
  }
  int64_t qDim = numK * headK;
  int64_t vDim = numV * headV;
  std::vector<float> q(qDim), k(qDim), v(vDim);
  for (int64_t i = 0; i < qDim; ++i) {
    q[i] = qkv[i];
    k[i] = qkv[qDim + i];
  }
  for (int64_t i = 0; i < vDim; ++i)
    v[i] = qkv[2 * qDim + i];
  if (debug)
    outs() << "layer0 conv qkv " << maxAbs(qkv) << " q " << maxAbs(q) << " k "
           << maxAbs(k) << " v " << maxAbs(v) << "\n";

  auto l2 = [](std::vector<float> &value, int64_t off, int64_t n) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i)
      sum += value[off + i] * value[off + i];
    float inv = 1.0f / std::sqrt(sum + 1e-6f);
    for (int64_t i = 0; i < n; ++i)
      value[off + i] *= inv;
  };
  for (int64_t h = 0; h < numK; ++h) {
    l2(q, h * headK, headK);
    l2(k, h * headK, headK);
    for (int64_t d = 0; d < headK; ++d)
      q[h * headK + d] /= std::sqrt(static_cast<float>(headK));
  }

  const Weight &aLog = arena.get(base + ".linear_attn.A_log");
  const Weight &dt = arena.get(base + ".linear_attn.dt_bias");
  std::vector<float> out(vDim);
  int64_t group = numV / numK;
  for (int64_t h = 0; h < numV; ++h) {
    int64_t kh = h / group;
    float beta = 1.0f / (1.0f + std::exp(-b[h]));
    float aVal = a[h];
    float dtVal = readScalar(dt, h);
    float softplus = std::log1p(std::exp(aVal + dtVal));
    float aLogVal = readScalar(aLog, h);
    float decay = std::exp(-std::exp(aLogVal) * softplus);
    float qk = dot(q, kh * headK, k, kh * headK, headK);
    if (debug && h == 0)
      outs() << "layer0 h0 beta " << beta << " decay " << decay << " qk " << qk
             << "\n";
    for (int64_t d = 0; d < headV; ++d)
      out[h * headV + d] =
          bf16ToFloat(floatToBf16(decay * qk * beta * v[h * headV + d]));
    std::string normName = base + ".linear_attn.norm.weight";
    const Weight &norm = arena.get(normName);
    float sum = 0.0f;
    for (int64_t d = 0; d < headV; ++d)
      sum += out[h * headV + d] * out[h * headV + d];
    float rstd = 1.0f / std::sqrt(sum / static_cast<float>(headV) + eps);
    for (int64_t d = 0; d < headV; ++d) {
      float w = readScalar(norm, d);
      float val = out[h * headV + d] * rstd;
      val = bf16ToFloat(floatToBf16(val));
      float gate = z[h * headV + d] / (1.0f + std::exp(-z[h * headV + d]));
      out[h * headV + d] = bf16ToFloat(floatToBf16(val * w * gate));
    }
  }
  if (debug) {
    outs() << "layer0 gated out " << maxAbs(out) << "\n";
    const Weight &proj = arena.get(base + ".linear_attn.out_proj.weight");
    outs() << "layer0 out_proj shape " << proj.shape.dims[0] << "x"
           << proj.shape.dims[1] << " dtype " << proj.dtype << "\n";
  }
  return linear(out, arena.get(base + ".linear_attn.out_proj.weight"));
}

struct Config {
  int64_t hidden = 2560;
  int64_t layers = 32;
  int64_t heads = 16;
  int64_t kvHeads = 4;
  int64_t headDim = 256;
  int64_t linearK = 16;
  int64_t linearV = 32;
  int64_t linearHeadK = 128;
  int64_t linearHeadV = 128;
  int64_t convKernel = 4;
  float eps = 1e-6f;
};

std::pair<std::vector<float>, std::vector<float>>
makeCosSin(int64_t pos, int64_t rotaryDim, float theta) {
  std::vector<float> cos(rotaryDim), sin(rotaryDim);
  for (int64_t d = 0; d < rotaryDim / 2; ++d) {
    float invFreq = 1.0f / std::pow(theta, static_cast<float>(2 * d) /
                                               static_cast<float>(rotaryDim));
    float angle = static_cast<float>(pos) * invFreq;
    cos[d] = cos[d + rotaryDim / 2] = std::cos(angle);
    sin[d] = sin[d + rotaryDim / 2] = std::sin(angle);
  }
  return {cos, sin};
}

std::vector<int64_t> topK(const std::vector<float> &values, size_t k) {
  std::vector<int64_t> indices(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    indices[i] = static_cast<int64_t>(i);
  std::partial_sort(
      indices.begin(), indices.begin() + k, indices.end(),
      [&](int64_t a, int64_t b) { return values[a] > values[b]; });
  indices.resize(k);
  return indices;
}

std::vector<float> runModel(const Arena &arena, int64_t token,
                            const Config &config) {
  std::vector<float> hidden(config.hidden);
  const Weight &embedding =
      arena.get("model.language_model.embed_tokens.weight");
  for (int64_t i = 0; i < config.hidden; ++i)
    hidden[i] = readBf16(embedding.data, token * config.hidden + i);
  if (debug)
    outs() << "embed " << hidden[0] << " " << hidden[1] << "\n";

  for (int64_t layer = 0; layer < config.layers; ++layer) {
    std::string base = "model.language_model.layers." + std::to_string(layer);
    auto norm = rmsNorm(hidden, arena.get(base + ".input_layernorm.weight"),
                        config.eps);
    std::vector<float> mixed;
    bool linearLayer = (layer % 4) != 3;
    if (debug && layer == 0)
      outs() << "norm max " << maxAbs(norm) << "\n";
    if (linearLayer) {
      mixed = linearAttention(norm, arena, base, config.linearK, config.linearV,
                              config.linearHeadK, config.linearHeadV,
                              config.convKernel, config.eps);
    } else {
      mixed = fullAttention(norm, arena, base, config.heads, config.kvHeads,
                            config.headDim);
    }
    if (debug && layer == 0)
      outs() << "mixed max " << maxAbs(mixed) << "\n";
    hidden = add(hidden, mixed);
    auto norm2 =
        rmsNorm(hidden, arena.get(base + ".post_attention_layernorm.weight"),
                config.eps);
    auto down = mlp(norm2, arena, base);
    if (debug && layer == 0)
      outs() << "down max " << maxAbs(down) << "\n";
    hidden = add(hidden, down);
    if (debug && layer < 4)
      outs() << "layer " << layer << " " << hidden[0] << " " << hidden[1]
             << "\n";
  }
  hidden = rmsNorm(hidden, arena.get("model.language_model.norm.weight"),
                   config.eps);
  return matvec(embedding, hidden);
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "llmcp-run\n");
  Arena arena;
  if (Error error = arena.load(weightsPath, manifestPath)) {
    errs() << "llmcp-run: " << toString(std::move(error)) << "\n";
    return 1;
  }
  Config config;
  int64_t selectedToken = tokenId;
  if (!prompt.empty()) {
    Tokenizer tokenizer;
    if (Error error = tokenizer.load(tokenizerPath)) {
      errs() << "llmcp-run: " << toString(std::move(error)) << "\n";
      return 1;
    }
    auto idsOrErr = tokenizer.encode(prompt);
    if (!idsOrErr) {
      errs() << "llmcp-run: " << toString(idsOrErr.takeError()) << "\n";
      return 1;
    }
    if (idsOrErr->size() != 1) {
      errs() << "llmcp-run: expected one token, got " << idsOrErr->size()
             << "\n";
      return 1;
    }
    selectedToken = (*idsOrErr)[0];
    outs() << "token: " << selectedToken << "\n";
  }
  int64_t rotaryDim = 64;
  auto [cosValues, sinValues] = makeCosSin(position, rotaryDim, 10000000.0f);
  if (dumpCosSin) {
    for (int64_t d = 0; d < rotaryDim; ++d)
      outs() << "cos[" << d << "]=" << cosValues[d] << " sin[" << d
             << "]=" << sinValues[d] << "\n";
  }
  auto logits = runModel(arena, selectedToken, config);
  std::vector<float> golden;
  if (!goldenPath.empty()) {
    auto buffer = MemoryBuffer::getFile(goldenPath);
    if (!buffer) {
      errs() << "llmcp-run: " << buffer.getError().message() << "\n";
      return 1;
    }
    StringRef bytes = (*buffer)->getBuffer();
    golden.resize(bytes.size() / sizeof(float));
    std::memcpy(golden.data(), bytes.data(), bytes.size());
    float maxAbs = 0.0f, sumAbs = 0.0f;
    for (size_t i = 0; i < golden.size(); ++i) {
      float diff = std::fabs(logits[i] - golden[i]);
      maxAbs = std::max(maxAbs, diff);
      sumAbs += diff;
    }
    auto oursTop = topK(logits, 5);
    auto refTop = topK(golden, 5);
    int overlap = 0;
    for (int64_t a : oursTop)
      overlap += std::find(refTop.begin(), refTop.end(), a) != refTop.end();
    outs() << "logits: " << logits.size() << ", max_abs " << maxAbs
           << ", mean_abs " << (sumAbs / golden.size()) << "\n";
    outs() << "top1 ours=" << oursTop[0] << " hf=" << refTop[0]
           << ", top5 overlap=" << overlap << "/5\n";
  } else {
    outs() << "logits: " << logits.size() << "\n";
  }
  if (dumpLogits) {
    for (float value : logits)
      outs() << value << "\n";
  }
  return 0;
}
