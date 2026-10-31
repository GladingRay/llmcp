//===- llmcp-run.cpp - C++ runner for packed Qwen3.5 weights --------------===//
//
// Loads the arena produced by llmcp-pack and executes prefill/greedy decode
// using the same math as the llmcp ops.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "llmcp/Runtime/AotRuntime.h"
#include "llmcp/Support/CacheLineAlignedVector.h"

#include "tokenizers_cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
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
using llmcp::CacheLineAlignedVector;
using namespace mlir::llmcp::runtime;

namespace {

cl::opt<std::string> weightsPath("weights", cl::desc("Packed arena"),
                                 cl::init("build/weights.bin"));
cl::opt<std::string> manifestPath("manifest", cl::desc("Arena manifest"),
                                  cl::init("build/weights.json"));
cl::opt<std::string> goldenPath("golden", cl::desc("Optional f32 logits file"));
cl::opt<long long> tokenId("tokens", cl::desc("Token ID when --text is empty"),
                           cl::init(9419));
cl::opt<std::string> prompt("text", cl::desc("Prompt text"), cl::init(""));
cl::opt<std::string> tokenizerPath(
    "tokenizer", cl::desc("HuggingFace tokenizer.json"),
    cl::init("/Users/guorui/Work/models/qwen3.5-4b/tokenizer.json"));
cl::opt<bool> chat("chat", cl::desc("Wrap --text in the Qwen chat template"),
                   cl::init(false));
cl::opt<std::string> systemPrompt("system", cl::desc("Optional system message"),
                                  cl::init(""));
cl::opt<bool> noThinking("no-thinking", cl::desc("Use an empty <think> block"),
                         cl::init(false));
cl::opt<long long> maxNewTokens("max-new-tokens",
                                cl::desc("Greedy decode token count"),
                                cl::init(0));
cl::opt<std::string>
    aotPath("aot", cl::desc("Optional AOT shared library (llmcp_model_aot)"),
            cl::init(""));
cl::opt<bool> dumpLogits("dump-logits", cl::desc("Write last-token logits"),
                         cl::init(false));
cl::opt<bool> debug("debug", cl::desc("Print per-layer stats"),
                    cl::init(false));
cl::opt<bool>
    tokenizeOnly("tokenize-only",
                 cl::desc("Only encode --text, do not load the model"),
                 cl::init(false));
cl::opt<long long>
    expectToken("expect-token",
                cl::desc("Fail unless the first encoded token matches"),
                cl::init(-1));
cl::opt<bool> dumpCosSin("dump-cos-sin",
                         cl::desc("Print generated cos/sin values"),
                         cl::init(false));

using Vec = CacheLineAlignedVector<float>;
using Matrix = std::vector<Vec>;

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

float roundBf16(float value) { return bf16ToFloat(floatToBf16(value)); }

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

Matrix matmul(const Weight &weight, const Matrix &x) {
  int64_t rows = static_cast<int64_t>(x.size());
  int64_t in = static_cast<int64_t>(x[0].size());
  int64_t out = weight.shape[0];
  Vec flatX(static_cast<size_t>(rows * in));
  for (int64_t r = 0; r < rows; ++r)
    std::copy(x[r].begin(), x[r].end(), flatX.begin() + r * in);
  Vec flatY(static_cast<size_t>(rows * out), 0.0f);

  constexpr int64_t kBlock = 256;
#ifdef __APPLE__
  for (int64_t start = 0; start < out; start += kBlock) {
    int64_t count = std::min(kBlock, out - start);
    Vec block(static_cast<size_t>(count * in));
    for (int64_t r = 0; r < count; ++r)
      for (int64_t c = 0; c < in; ++c)
        block[r * in + c] = readScalar(weight, (start + r) * in + c);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(rows),
                static_cast<int>(count), static_cast<int>(in), 1.0f,
                flatX.data(), static_cast<int>(in), block.data(),
                static_cast<int>(in), 0.0f, flatY.data() + start,
                static_cast<int>(out));
  }
#else
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t o = 0; o < out; ++o) {
      float sum = 0.0f;
      for (int64_t c = 0; c < in; ++c)
        sum += x[r][c] * readScalar(weight, o * in + c);
      flatY[r * out + o] = sum;
    }
#endif

  Matrix y(static_cast<size_t>(rows), Vec(static_cast<size_t>(out)));
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t o = 0; o < out; ++o)
      y[r][o] = roundBf16(flatY[r * out + o]);
  return y;
}

Matrix rmsNorm(const Matrix &x, const Weight &weight, float eps,
               bool zeroCentered = true) {
  Matrix y(x.size(), Vec(x[0].size()));
  for (size_t r = 0; r < x.size(); ++r) {
    float sum = 0.0f;
    for (float value : x[r])
      sum += value * value;
    float rstd = 1.0f / std::sqrt(sum / static_cast<float>(x[r].size()) + eps);
    for (size_t i = 0; i < x[r].size(); ++i) {
      float w = readScalar(weight, static_cast<int64_t>(i));
      y[r][i] = roundBf16(x[r][i] * rstd * (zeroCentered ? 1.0f + w : w));
    }
  }
  return y;
}

Matrix add(const Matrix &a, const Matrix &b) {
  Matrix y(a.size(), Vec(a[0].size()));
  for (size_t r = 0; r < a.size(); ++r)
    for (size_t c = 0; c < a[r].size(); ++c)
      y[r][c] = roundBf16(a[r][c] + b[r][c]);
  return y;
}

Matrix mlp(const Matrix &x, const Arena &arena, StringRef prefix) {
  std::string base = prefix.str();
  Matrix gate = matmul(arena.get(base + ".mlp.gate_proj.weight"), x);
  Matrix up = matmul(arena.get(base + ".mlp.up_proj.weight"), x);
  for (size_t r = 0; r < gate.size(); ++r)
    for (size_t c = 0; c < gate[r].size(); ++c)
      gate[r][c] =
          roundBf16((gate[r][c] / (1.0f + std::exp(-gate[r][c]))) * up[r][c]);
  return matmul(arena.get(base + ".mlp.down_proj.weight"), gate);
}

void applyRope(Vec &values, const Vec &cos, const Vec &sin, int64_t rotary) {
  Vec original = values;
  int64_t half = rotary / 2;
  for (int64_t d = 0; d < rotary; ++d) {
    bool second = d >= half;
    int64_t partner = second ? d - half : d + half;
    float sign = second ? 1.0f : -1.0f;
    values[d] = original[d] * cos[d] + sign * original[partner] * sin[d];
  }
}

void normalize(Vec &values) {
  float sum = 0.0f;
  for (float value : values)
    sum += value * value;
  float inv = 1.0f / std::sqrt(sum + 1e-6f);
  for (float &value : values)
    value *= inv;
}

struct FullCache {
  std::vector<Vec> keys;
  std::vector<Vec> values;
  int64_t filled = 0;
};

struct LinearState {
  std::vector<Vec> conv;
  Vec delta;
  Vec cumulativeDecay;
};

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
  int64_t rotary = 64;
  float theta = 10000000.0f;
  float eps = 1e-6f;
  int64_t numLinear() const { return layers / 4 * 3; }
  int64_t numFull() const { return layers / 4; }
};

struct Runtime {
  std::vector<FullCache> full;
  std::vector<LinearState> linear;
  int64_t capacity = 0;

  void init(const Config &config, int64_t maxSequence) {
    capacity = maxSequence;
    full.resize(config.numFull());
    for (FullCache &cache : full) {
      cache.keys.assign(capacity, Vec(config.kvHeads * config.headDim, 0.0f));
      cache.values.assign(capacity, Vec(config.kvHeads * config.headDim, 0.0f));
    }
    linear.resize(config.numLinear());
    for (LinearState &state : linear) {
      state.conv.assign(config.convKernel - 1,
                        Vec(2 * config.linearK * config.linearHeadK +
                                config.linearV * config.linearHeadV,
                            0.0f));
      state.delta.assign(
          config.linearV * config.linearHeadK * config.linearHeadV, 0.0f);
      state.cumulativeDecay.assign(config.linearV, 0.0f);
    }
  }
};

Matrix fullAttention(const Matrix &x, const Arena &arena, StringRef prefix,
                     Runtime &runtime, int64_t fullIndex, int64_t startPos,
                     const Config &config) {
  std::string base = prefix.str();
  Matrix qg = matmul(arena.get(base + ".self_attn.q_proj.weight"), x);
  Matrix k = matmul(arena.get(base + ".self_attn.k_proj.weight"), x);
  Matrix v = matmul(arena.get(base + ".self_attn.v_proj.weight"), x);
  const Weight &qNorm = arena.get(base + ".self_attn.q_norm.weight");
  const Weight &kNorm = arena.get(base + ".self_attn.k_norm.weight");
  FullCache &cache = runtime.full[fullIndex];
  Matrix output(x.size(), Vec(config.heads * config.headDim));

  for (size_t t = 0; t < x.size(); ++t) {
    int64_t global = startPos + static_cast<int64_t>(t);
    auto [cos, sin] = makeCosSin(global, config.rotary, config.theta);
    for (int64_t h = 0; h < config.heads; ++h) {
      Vec query(qg[t].begin() + h * 2 * config.headDim,
                qg[t].begin() + h * 2 * config.headDim + config.headDim);
      Vec gate(qg[t].begin() + h * 2 * config.headDim + config.headDim,
               qg[t].begin() + h * 2 * config.headDim + 2 * config.headDim);
      float qSum = 0.0f;
      for (float value : query)
        qSum += value * value;
      float qRstd = 1.0f / std::sqrt(qSum / config.headDim + config.eps);
      for (int64_t d = 0; d < config.headDim; ++d)
        query[d] = query[d] * qRstd * (1.0f + readScalar(qNorm, d));
      applyRope(query, cos, sin, config.rotary);
      for (int64_t d = 0; d < config.headDim; ++d)
        query[d] = roundBf16(query[d]);

      Vec values(config.kvHeads * config.headDim);
      for (int64_t kv = 0; kv < config.kvHeads; ++kv) {
        Vec key(k[t].begin() + kv * config.headDim,
                k[t].begin() + (kv + 1) * config.headDim);
        float kSum = 0.0f;
        for (float value : key)
          kSum += value * value;
        float kRstd = 1.0f / std::sqrt(kSum / config.headDim + config.eps);
        for (int64_t d = 0; d < config.headDim; ++d)
          key[d] = key[d] * kRstd * (1.0f + readScalar(kNorm, d));
        applyRope(key, cos, sin, config.rotary);
        for (int64_t d = 0; d < config.headDim; ++d) {
          cache.keys[global][kv * config.headDim + d] = roundBf16(key[d]);
          cache.values[global][kv * config.headDim + d] =
              roundBf16(v[t][kv * config.headDim + d]);
        }
      }

      int64_t kvHead = h / (config.heads / config.kvHeads);
      constexpr int64_t kBlock = 64;
      Vec context(config.headDim, 0.0f);
      Vec scores(kBlock, 0.0f);
      float runningMax = -1.0e30f;
      float runningSum = 0.0f;
      for (int64_t blockStart = 0; blockStart <= global;
           blockStart += kBlock) {
        int64_t blockEnd = std::min(blockStart + kBlock, global + 1);
        float tileMax = -1.0e30f;
        for (int64_t p = blockStart; p < blockEnd; ++p) {
          float sum = 0.0f;
          for (int64_t d = 0; d < config.headDim; ++d)
            sum += query[d] * cache.keys[p][kvHead * config.headDim + d];
          scores[p - blockStart] =
              sum / std::sqrt(static_cast<float>(config.headDim));
          tileMax = std::max(tileMax, scores[p - blockStart]);
        }

        float newMax = std::max(runningMax, tileMax);
        float correction = std::exp(runningMax - newMax);
        runningSum *= correction;
        for (int64_t d = 0; d < config.headDim; ++d)
          context[d] *= correction;

        for (int64_t p = blockStart; p < blockEnd; ++p) {
          float probability = std::exp(scores[p - blockStart] - newMax);
          runningSum += probability;
          for (int64_t d = 0; d < config.headDim; ++d)
            context[d] += probability *
                          cache.values[p][kvHead * config.headDim + d];
        }
        runningMax = newMax;
      }
      for (int64_t d = 0; d < config.headDim; ++d) {
        float gateValue = 1.0f / (1.0f + std::exp(-gate[d]));
        output[t][h * config.headDim + d] =
            roundBf16(context[d] / runningSum * gateValue);
      }
    }
  }
  cache.filled =
      std::max(cache.filled, startPos + static_cast<int64_t>(x.size()));
  return matmul(arena.get(base + ".self_attn.o_proj.weight"), output);
}

Matrix linearAttention(const Matrix &x, const Arena &arena, StringRef prefix,
                       Runtime &runtime, int64_t linearIndex,
                       const Config &config, bool decode) {
  std::string base = prefix.str();
  Matrix qkv = matmul(arena.get(base + ".linear_attn.in_proj_qkv.weight"), x);
  Matrix z = matmul(arena.get(base + ".linear_attn.in_proj_z.weight"), x);
  Matrix a = matmul(arena.get(base + ".linear_attn.in_proj_a.weight"), x);
  Matrix b = matmul(arena.get(base + ".linear_attn.in_proj_b.weight"), x);
  const Weight &convWeight = arena.get(base + ".linear_attn.conv1d.weight");
  LinearState &state = runtime.linear[linearIndex];
  int64_t channels = static_cast<int64_t>(qkv[0].size());
  int64_t stateRows = config.convKernel - 1;
  Matrix padded(stateRows + x.size(), Vec(channels, 0.0f));
  for (int64_t i = 0; i < stateRows; ++i)
    padded[i] = state.conv[i];
  for (size_t t = 0; t < x.size(); ++t)
    padded[stateRows + t] = qkv[t];

  for (size_t t = 0; t < x.size(); ++t)
    for (int64_t c = 0; c < channels; ++c) {
      float sum = 0.0f;
      for (int64_t k = 0; k < config.convKernel; ++k)
        sum += readScalar(convWeight, c * config.convKernel + k) *
               padded[t + k][c];
      sum = roundBf16(sum);
      qkv[t][c] = roundBf16(sum / (1.0f + std::exp(-sum)));
    }
  for (int64_t i = 0; i < stateRows; ++i)
    state.conv[i] = padded[padded.size() - stateRows + i];

  int64_t qDim = config.linearK * config.linearHeadK;
  int64_t vDim = config.linearV * config.linearHeadV;
  Matrix output(x.size(), Vec(vDim));
  const Weight &aLog = arena.get(base + ".linear_attn.A_log");
  const Weight &dt = arena.get(base + ".linear_attn.dt_bias");
  const Weight &normWeight = arena.get(base + ".linear_attn.norm.weight");
  size_t seq = x.size();

  for (int64_t h = 0; h < config.linearV; ++h) {
    int64_t kh = h / (config.linearV / config.linearK);
    std::vector<Vec> qh(seq, Vec(config.linearHeadK));
    std::vector<Vec> khv(seq, Vec(config.linearHeadK));
    std::vector<Vec> vh(seq, Vec(config.linearHeadV));
    Vec beta(seq), gate(seq), cum(seq);
    for (size_t t = 0; t < seq; ++t) {
      for (int64_t i = 0; i < config.linearHeadK; ++i) {
        qh[t][i] = qkv[t][kh * config.linearHeadK + i];
        khv[t][i] = qkv[t][qDim + kh * config.linearHeadK + i];
      }
      normalize(qh[t]);
      normalize(khv[t]);
      float qScale = 1.0f / std::sqrt(static_cast<float>(config.linearHeadK));
      for (float &value : qh[t])
        value *= qScale;
      for (int64_t i = 0; i < config.linearHeadV; ++i)
        vh[t][i] = qkv[t][2 * qDim + h * config.linearHeadV + i];
      beta[t] = 1.0f / (1.0f + std::exp(-b[t][h]));
      float dtValue = readScalar(dt, h);
      float softplus = std::log1p(std::exp(a[t][h] + dtValue));
      gate[t] = -std::exp(readScalar(aLog, h)) * softplus;
    }
    for (size_t t = 0; t < seq; ++t)
      cum[t] = std::exp(gate[t] + (t == 0 ? 0.0f : std::log(cum[t - 1])));

    float *delta =
        state.delta.data() + h * config.linearHeadK * config.linearHeadV;

    if (!decode && seq > 1) {
      std::vector<Vec> ut(seq, Vec(seq, 0.0f));
      for (size_t i = 0; i < seq; ++i)
        for (size_t j = 0; j < i; ++j) {
          float dot = 0.0f;
          for (int64_t d = 0; d < config.linearHeadK; ++d)
            dot += khv[i][d] * beta[i] * khv[j][d];
          float decay = cum[i] / cum[j];
          ut[i][j] = dot * decay;
        }
      std::vector<Vec> newValue(seq, Vec(config.linearHeadV, 0.0f));
      std::vector<Vec> kCum(seq, Vec(config.linearHeadK, 0.0f));
      for (size_t i = 0; i < seq; ++i) {
        for (int64_t d = 0; d < config.linearHeadV; ++d)
          newValue[i][d] = vh[i][d] * beta[i];
        for (int64_t d = 0; d < config.linearHeadK; ++d)
          kCum[i][d] = khv[i][d] * beta[i] * cum[i];
        for (size_t j = 0; j < i; ++j) {
          for (int64_t d = 0; d < config.linearHeadV; ++d)
            newValue[i][d] += ut[i][j] * newValue[j][d];
          for (int64_t d = 0; d < config.linearHeadK; ++d)
            kCum[i][d] += ut[i][j] * kCum[j][d];
        }
      }
      std::vector<Vec> vNew(seq, Vec(config.linearHeadV, 0.0f));
      for (size_t i = 0; i < seq; ++i) {
        vNew[i] = newValue[i];
        for (int64_t ki = 0; ki < config.linearHeadK; ++ki)
          for (int64_t vi = 0; vi < config.linearHeadV; ++vi)
            vNew[i][vi] -= kCum[i][ki] * delta[ki * config.linearHeadV + vi];
      }
      std::vector<Vec> intra(seq, Vec(seq, 0.0f));
      for (size_t i = 0; i < seq; ++i)
        for (size_t j = 0; j <= i; ++j) {
          float dot = 0.0f;
          for (int64_t d = 0; d < config.linearHeadK; ++d)
            dot += qh[i][d] * khv[j][d];
          intra[i][j] = dot * (cum[i] / cum[j]);
        }
      for (size_t i = 0; i < seq; ++i) {
        Vec out(config.linearHeadV, 0.0f);
        for (int64_t ki = 0; ki < config.linearHeadK; ++ki)
          for (int64_t vi = 0; vi < config.linearHeadV; ++vi)
            out[vi] += qh[i][ki] * cum[i] * delta[ki * config.linearHeadV + vi];
        for (size_t j = 0; j <= i; ++j)
          for (int64_t vi = 0; vi < config.linearHeadV; ++vi)
            out[vi] += intra[i][j] * vNew[j][vi];
        for (int64_t vi = 0; vi < config.linearHeadV; ++vi)
          output[i][h * config.linearHeadV + vi] = roundBf16(out[vi]);
      }
      for (int64_t ki = 0; ki < config.linearHeadK; ++ki)
        for (int64_t vi = 0; vi < config.linearHeadV; ++vi)
          delta[ki * config.linearHeadV + vi] =
              roundBf16(delta[ki * config.linearHeadV + vi] * cum[seq - 1]);
      for (size_t i = 0; i < seq; ++i) {
        float keyDecay = cum[seq - 1] / cum[i];
        for (int64_t ki = 0; ki < config.linearHeadK; ++ki)
          for (int64_t vi = 0; vi < config.linearHeadV; ++vi)
            delta[ki * config.linearHeadV + vi] =
                roundBf16(delta[ki * config.linearHeadV + vi] +
                          khv[i][ki] * keyDecay * vNew[i][vi]);
      }
    } else {
      for (size_t t = 0; t < seq; ++t) {
        float decay = std::exp(gate[t]);
        for (int64_t ki = 0; ki < config.linearHeadK; ++ki) {
          float kVal = khv[t][ki];
          for (int64_t vi = 0; vi < config.linearHeadV; ++vi) {
            float &value = delta[ki * config.linearHeadV + vi];
            value = roundBf16(value * decay);
            float kv = value * kVal;
            float deltaValue = roundBf16((vh[t][vi] - kv) * beta[t]);
            value = roundBf16(value + kVal * deltaValue);
          }
        }
        for (int64_t vi = 0; vi < config.linearHeadV; ++vi) {
          float sum = 0.0f;
          for (int64_t ki = 0; ki < config.linearHeadK; ++ki)
            sum += delta[ki * config.linearHeadV + vi] * qh[t][ki];
          output[t][h * config.linearHeadV + vi] = roundBf16(sum);
        }
      }
    }

    for (size_t t = 0; t < seq; ++t) {
      float sumSq = 0.0f;
      for (int64_t vi = 0; vi < config.linearHeadV; ++vi)
        sumSq += output[t][h * config.linearHeadV + vi] *
                 output[t][h * config.linearHeadV + vi];
      float rstd = 1.0f / std::sqrt(sumSq / config.linearHeadV + config.eps);
      for (int64_t vi = 0; vi < config.linearHeadV; ++vi) {
        float g = z[t][h * config.linearHeadV + vi];
        g = g / (1.0f + std::exp(-g));
        float value = output[t][h * config.linearHeadV + vi] * rstd;
        value = roundBf16(value) * readScalar(normWeight, vi);
        output[t][h * config.linearHeadV + vi] = roundBf16(value * g);
      }
    }
  }
  return matmul(arena.get(base + ".linear_attn.out_proj.weight"), output);
}

float maxAbs(const Vec &v) {
  float m = 0.0f;
  for (float x : v)
    m = std::max(m, std::fabs(x));
  return m;
}

float maxAbs(const Matrix &m) {
  float result = 0.0f;
  for (const Vec &row : m)
    result = std::max(result, maxAbs(row));
  return result;
}

Vec runModel(const Arena &arena, const std::vector<int64_t> &tokens,
             int64_t startPos, Runtime &runtime, const Config &config,
             bool decode) {
  Matrix hidden(tokens.size(), Vec(config.hidden));
  const Weight &embedding =
      arena.get("model.language_model.embed_tokens.weight");
  for (size_t t = 0; t < tokens.size(); ++t)
    for (int64_t i = 0; i < config.hidden; ++i)
      hidden[t][i] = readBf16(embedding.data, tokens[t] * config.hidden + i);

  int64_t linearIndex = 0, fullIndex = 0;
  for (int64_t layer = 0; layer < config.layers; ++layer) {
    std::string base = "model.language_model.layers." + std::to_string(layer);
    Matrix residual = hidden;
    Matrix norm = rmsNorm(hidden, arena.get(base + ".input_layernorm.weight"),
                          config.eps);
    Matrix mixed;
    if (layer % 4 != 3) {
      mixed = linearAttention(norm, arena, base, runtime, linearIndex, config,
                              decode);
      ++linearIndex;
    } else {
      mixed = fullAttention(norm, arena, base, runtime, fullIndex, startPos,
                            config);
      ++fullIndex;
    }
    if (debug && layer < 4)
      outs() << "layer " << layer << " mixed max " << maxAbs(mixed) << " first "
             << mixed[0][0] << "\n";
    hidden = add(residual, mixed);
    residual = hidden;
    norm = rmsNorm(hidden, arena.get(base + ".post_attention_layernorm.weight"),
                   config.eps);
    Matrix down = mlp(norm, arena, base);
    if (debug && layer == 0)
      outs() << "layer0 down max " << maxAbs(down) << "\n";
    hidden = add(residual, down);
    if (debug && layer < 6)
      outs() << "layer " << layer << " hidden " << hidden[0][0] << " "
             << hidden[0][1] << " max " << maxAbs(hidden) << "\n";
  }
  hidden = rmsNorm(hidden, arena.get("model.language_model.norm.weight"),
                   config.eps);
  Matrix logits = matmul(embedding, hidden);
  if (debug)
    outs() << "final hidden max " << maxAbs(hidden) << " logits max "
           << maxAbs(logits.back()) << "\n";
  return logits.back();
}

std::vector<int64_t> topK(const Vec &values, size_t k) {
  std::vector<int64_t> indices(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    indices[i] = static_cast<int64_t>(i);
  std::partial_sort(
      indices.begin(), indices.begin() + k, indices.end(),
      [&](int64_t a, int64_t b) { return values[a] > values[b]; });
  indices.resize(k);
  return indices;
}

int64_t argmax(const Vec &values) {
  return static_cast<int64_t>(std::max_element(values.begin(), values.end()) -
                              values.begin());
}

} // namespace

std::string makeChatPrompt(StringRef user, StringRef system, bool thinking) {
  std::string prompt;
  if (!system.empty()) {
    prompt += "<|im_start|>system\n";
    prompt += system.str();
    prompt += "<|im_end|>\n";
  }
  prompt += "<|im_start|>user\n";
  prompt += user.str();
  prompt += "<|im_end|>\n";
  prompt += "<|im_start|>assistant\n";
  if (thinking)
    prompt += "<think>\n";
  else
    prompt += "<think>\n\n</think>\n\n";
  return prompt;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "llmcp-run\n");

  auto blob = MemoryBuffer::getFile(tokenizerPath);
  if (!blob) {
    errs() << "llmcp-run: " << blob.getError().message() << "\n";
    return 1;
  }
  auto tokenizer =
      tokenizers::Tokenizer::FromBlobJSON((*blob)->getBuffer().str());

  std::vector<int64_t> tokens;
  if (!prompt.empty()) {
    std::string text =
        chat ? makeChatPrompt(prompt, systemPrompt, !noThinking) : prompt;
    auto ids = tokenizer->Encode(text);
    for (int32_t id : ids)
      tokens.push_back(id);
  } else {
    tokens.push_back(tokenId);
  }
  int32_t eosToken = tokenizer->TokenToId("<|im_end|>");
  if (tokenizeOnly) {
    for (size_t i = 0; i < tokens.size(); ++i)
      outs() << (i ? "," : "") << tokens[i];
    outs() << "\n";
    if (expectToken >= 0 && (tokens.empty() || tokens[0] != expectToken))
      return 1;
    return 0;
  }

  if (!aotPath.empty()) {
    Arena aotArena;
    if (Error error = aotArena.load(weightsPath, manifestPath)) {
      errs() << "llmcp-run: " << toString(std::move(error)) << "\n";
      return 1;
    }
    AotRuntime aot;
    if (!aot.init(aotPath,
                  static_cast<int64_t>(tokens.size()) + maxNewTokens + 16))
      return 1;
    Vec allLogits;
    if (!aot.forward(aotArena, tokens, 0, tokens.size(), allLogits))
      return 1;
    Vec logits(allLogits.end() - 248320, allLogits.end());
    std::vector<int64_t> generated;
    if (maxNewTokens > 0) {
      outs() << "output: ";
      outs().flush();
    }
    for (int64_t i = 0; i < maxNewTokens; ++i) {
      int64_t next = argmax(logits);
      if (next == eosToken)
        break;
      generated.push_back(next);
      std::vector<int32_t> oneI32{static_cast<int32_t>(next)};
      outs() << tokenizer->Decode(oneI32);
      outs().flush();
      std::vector<int64_t> one{next};
      int64_t pos = static_cast<int64_t>(tokens.size() + generated.size() - 1);
      if (!aot.forward(aotArena, one, pos, pos + 1, allLogits))
        return 1;
      logits.assign(allLogits.end() - 248320, allLogits.end());
    }
    if (!generated.empty()) {
      outs() << "\n";
      outs() << "tokens:";
      for (int64_t id : generated)
        outs() << " " << id;
      outs() << "\n";
    } else {
      outs() << "aot logits: " << logits.size() << " next " << argmax(logits)
             << "\n";
    }
    if (dumpLogits)
      for (float value : logits)
        outs() << value << "\n";
    return 0;
  }

  Arena arena;
  if (Error error = arena.load(weightsPath, manifestPath)) {
    errs() << "llmcp-run: " << toString(std::move(error)) << "\n";
    return 1;
  }
  Config config;
  Runtime runtime;
  runtime.init(config, static_cast<int64_t>(tokens.size()) + maxNewTokens + 16);

  Vec logits = runModel(arena, tokens, 0, runtime, config, /*decode=*/false);
  std::vector<int64_t> generated;
  if (maxNewTokens > 0) {
    outs() << "output: ";
    outs().flush();
  }
  for (int64_t i = 0; i < maxNewTokens; ++i) {
    int64_t next = argmax(logits);
    if (next == eosToken)
      break;
    generated.push_back(next);
    std::vector<int32_t> oneI32{static_cast<int32_t>(next)};
    outs() << tokenizer->Decode(oneI32);
    outs().flush();
    std::vector<int64_t> one{next};
    logits = runModel(
        arena, one, static_cast<int64_t>(tokens.size() + generated.size() - 1),
        runtime, config, /*decode=*/true);
  }

  if (!generated.empty()) {
    outs() << "\n";
    outs() << "tokens:";
    for (int64_t id : generated)
      outs() << " " << id;
    outs() << "\n";
  }

  Vec golden;
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
  } else if (generated.empty()) {
    outs() << "next: " << argmax(logits) << "\n";
    outs() << "logits: " << logits.size() << "\n";
  }

  if (dumpLogits) {
    for (float value : logits)
      outs() << value << "\n";
  }
  if (dumpCosSin) {
    auto [cos, sin] = makeCosSin(0, config.rotary, config.theta);
    for (int64_t d = 0; d < config.rotary; ++d)
      outs() << "cos[" << d << "]=" << cos[d] << " sin[" << d << "]=" << sin[d]
             << "\n";
  }
  return 0;
}
