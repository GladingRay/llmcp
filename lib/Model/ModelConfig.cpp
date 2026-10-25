//===- ModelConfig.cpp - Qwen3.5 model configuration ----------------------===//
//
//===----------------------------------------------------------------------===//

#include "llmcp/Model/ModelConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cmath>
#include <optional>
#include <string>
#include <utility>

using namespace mlir::llmcp;
using llvm::StringRef;
using llvm::json::Object;
using llvm::json::Value;

namespace {

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Expected<int64_t> readInteger(const Object &object, StringRef key) {
  if (std::optional<int64_t> value = object.getInteger(key))
    return *value;
  return makeError(llvm::Twine("missing or invalid integer field '") + key +
                   "'");
}

llvm::Expected<std::string> readString(const Object &object, StringRef key) {
  if (std::optional<StringRef> value = object.getString(key))
    return value->str();
  return makeError(llvm::Twine("missing or invalid string field '") + key +
                   "'");
}

int64_t readIntegerOr(const Object &object, StringRef key, int64_t fallback) {
  return object.getInteger(key).value_or(fallback);
}

double readNumberOr(const Object &object, StringRef key, double fallback) {
  return object.getNumber(key).value_or(fallback);
}

bool readBoolOr(const Object &object, StringRef key, bool fallback) {
  return object.getBoolean(key).value_or(fallback);
}

} // namespace

llvm::StringRef mlir::llmcp::stringifyLayerKind(LayerKind kind) {
  switch (kind) {
  case LayerKind::LinearAttention:
    return "linear_attention";
  case LayerKind::FullAttention:
    return "full_attention";
  }
  llvm_unreachable("unknown LayerKind");
}

llvm::Expected<ModelConfig> ModelConfig::load(StringRef modelDir) {
  std::string path = (llvm::Twine(modelDir) + "/config.json").str();
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return llvm::errorCodeToError(bufferOrErr.getError());
  return parse((*bufferOrErr)->getBuffer(), modelDir);
}

llvm::Expected<ModelConfig> ModelConfig::parse(StringRef json,
                                               StringRef modelDir) {
  auto parsed = llvm::json::parse(json);
  if (!parsed)
    return parsed.takeError();

  const Object *root = parsed->getAsObject();
  if (!root)
    return makeError("config.json must contain a top-level object");

  const Object *text = root->getObject("text_config");
  if (!text)
    return makeError("config.json must contain a 'text_config' object");

  ModelConfig config;
  config.modelDir = modelDir.str();

  TextConfig &tc = config.text;
  auto hiddenSize = readInteger(*text, "hidden_size");
  if (!hiddenSize)
    return hiddenSize.takeError();
  tc.hiddenSize = *hiddenSize;

  auto intermediateSize = readInteger(*text, "intermediate_size");
  if (!intermediateSize)
    return intermediateSize.takeError();
  tc.intermediateSize = *intermediateSize;

  auto numHiddenLayers = readInteger(*text, "num_hidden_layers");
  if (!numHiddenLayers)
    return numHiddenLayers.takeError();
  tc.numHiddenLayers = *numHiddenLayers;

  auto numAttentionHeads = readInteger(*text, "num_attention_heads");
  if (!numAttentionHeads)
    return numAttentionHeads.takeError();
  tc.numAttentionHeads = *numAttentionHeads;

  auto numKeyValueHeads = readInteger(*text, "num_key_value_heads");
  if (!numKeyValueHeads)
    return numKeyValueHeads.takeError();
  tc.numKeyValueHeads = *numKeyValueHeads;

  auto vocabSize = readInteger(*text, "vocab_size");
  if (!vocabSize)
    return vocabSize.takeError();
  tc.vocabSize = *vocabSize;

  tc.headDim = readIntegerOr(*text, "head_dim", 0);
  if (tc.headDim == 0 && tc.numAttentionHeads != 0)
    tc.headDim = tc.hiddenSize / tc.numAttentionHeads;

  tc.maxPositionEmbeddings =
      readIntegerOr(*text, "max_position_embeddings", 0);
  tc.fullAttentionInterval =
      readIntegerOr(*text, "full_attention_interval", 4);

  auto linearKeyHeads = readInteger(*text, "linear_num_key_heads");
  if (!linearKeyHeads)
    return linearKeyHeads.takeError();
  tc.linearNumKeyHeads = *linearKeyHeads;

  auto linearValueHeads = readInteger(*text, "linear_num_value_heads");
  if (!linearValueHeads)
    return linearValueHeads.takeError();
  tc.linearNumValueHeads = *linearValueHeads;

  auto linearKeyHeadDim = readInteger(*text, "linear_key_head_dim");
  if (!linearKeyHeadDim)
    return linearKeyHeadDim.takeError();
  tc.linearKeyHeadDim = *linearKeyHeadDim;

  auto linearValueHeadDim = readInteger(*text, "linear_value_head_dim");
  if (!linearValueHeadDim)
    return linearValueHeadDim.takeError();
  tc.linearValueHeadDim = *linearValueHeadDim;

  tc.linearConvKernelDim =
      readIntegerOr(*text, "linear_conv_kernel_dim", 4);

  tc.attentionBias = readBoolOr(*text, "attention_bias", false);
  tc.attnOutputGate = readBoolOr(*text, "attn_output_gate", true);

  auto hiddenAct = readString(*text, "hidden_act");
  if (!hiddenAct)
    return hiddenAct.takeError();
  tc.hiddenAct = *hiddenAct;

  tc.rmsNormEps = readNumberOr(*text, "rms_norm_eps", 1.0e-6);

  const Object *rope = text->getObject("rope_parameters");
  if (rope) {
    tc.ropeTheta = readNumberOr(*rope, "rope_theta", tc.ropeTheta);
    tc.partialRotaryFactor =
        readNumberOr(*rope, "partial_rotary_factor",
                     tc.partialRotaryFactor);
    tc.mropeInterleaved =
        readBoolOr(*rope, "mrope_interleaved", tc.mropeInterleaved);
    if (const auto *section = rope->getArray("mrope_section")) {
      for (const Value &value : *section) {
        auto entry = value.getAsInteger();
        if (!entry)
          return makeError("rope_parameters.mrope_section must contain "
                           "integers");
        tc.mropeSection.push_back(*entry);
      }
    }
  }
  if (tc.mropeSection.empty()) {
    if (const auto *section = text->getArray("mrope_section")) {
      for (const Value &value : *section) {
        auto entry = value.getAsInteger();
        if (!entry)
          return makeError("mrope_section must contain integers");
        tc.mropeSection.push_back(*entry);
      }
    }
  }
  if (tc.mropeSection.empty())
    tc.mropeSection = {11, 11, 10};

  const auto *layerTypes = text->getArray("layer_types");
  if (layerTypes) {
    for (const Value &value : *layerTypes) {
      auto name = value.getAsString();
      if (!name)
        return makeError("layer_types must contain strings");
      if (*name == "linear_attention") {
        config.layerTypes.push_back(LayerKind::LinearAttention);
      } else if (*name == "full_attention") {
        config.layerTypes.push_back(LayerKind::FullAttention);
      } else {
        return makeError(llvm::Twine("unknown layer type '") + *name + "'");
      }
    }
  } else {
    if (tc.fullAttentionInterval <= 0)
      return makeError("full_attention_interval must be positive");
    for (int64_t i = 0; i < tc.numHiddenLayers; ++i) {
      const bool isFull = ((i + 1) % tc.fullAttentionInterval) == 0;
      config.layerTypes.push_back(isFull ? LayerKind::FullAttention
                                         : LayerKind::LinearAttention);
    }
  }

  config.tieWordEmbeddings =
      root->getBoolean("tie_word_embeddings")
          .value_or(text->getBoolean("tie_word_embeddings").value_or(true));

  if (llvm::Error error = config.verify())
    return std::move(error);
  return std::move(config);
}

llvm::Error ModelConfig::verify() const {
  const TextConfig &tc = text;
  if (tc.numHiddenLayers <= 0)
    return makeError("num_hidden_layers must be positive");
  if (static_cast<int64_t>(layerTypes.size()) != tc.numHiddenLayers)
    return makeError(llvm::Twine("layer_types has ") +
                     llvm::Twine(layerTypes.size()) +
                     " entries, expected " +
                     llvm::Twine(tc.numHiddenLayers));
  if (tc.hiddenSize <= 0 || tc.intermediateSize <= 0 ||
      tc.vocabSize <= 0)
    return makeError("hidden/intermediate/vocab sizes must be positive");
  if (tc.headDim <= 0 || tc.numAttentionHeads <= 0 ||
      tc.numKeyValueHeads <= 0)
    return makeError("attention head configuration must be positive");
  if (tc.numAttentionHeads % tc.numKeyValueHeads != 0)
    return makeError("num_attention_heads must be divisible by "
                     "num_key_value_heads");
  if (tc.linearNumKeyHeads <= 0 || tc.linearNumValueHeads <= 0 ||
      tc.linearKeyHeadDim <= 0 || tc.linearValueHeadDim <= 0)
    return makeError("linear attention head configuration must be positive");
  if (tc.linearNumValueHeads % tc.linearNumKeyHeads != 0)
    return makeError("linear_num_value_heads must be divisible by "
                     "linear_num_key_heads");
  if (tc.linearConvKernelDim <= 0)
    return makeError("linear_conv_kernel_dim must be positive");
  if (tc.partialRotaryFactor <= 0.0 || tc.partialRotaryFactor > 1.0)
    return makeError("partial_rotary_factor must be in (0, 1]");
  if (tc.rmsNormEps <= 0.0)
    return makeError("rms_norm_eps must be positive");
  if (tc.hiddenAct != "silu")
    return makeError(llvm::Twine("unsupported hidden_act '") +
                     tc.hiddenAct + "' (only 'silu' is supported)");
  if (tc.fullAttentionInterval <= 0)
    return makeError("full_attention_interval must be positive");
  if (tc.numHiddenLayers % tc.fullAttentionInterval != 0)
    return makeError("num_hidden_layers must be divisible by "
                     "full_attention_interval");

  const int64_t rotaryDim = getRotaryDim();
  if (rotaryDim <= 0 || (rotaryDim % 2) != 0)
    return makeError("rotary dimension must be positive and even");
  int64_t sectionSum = 0;
  for (int64_t section : tc.mropeSection) {
    if (section < 0)
      return makeError("mrope_section entries must be non-negative");
    sectionSum += section;
  }
  if (sectionSum != rotaryDim / 2)
    return makeError(llvm::Twine("mrope_section sums to ") +
                     llvm::Twine(sectionSum) + ", expected " +
                     llvm::Twine(rotaryDim / 2));

  if (getNumLinearLayers() == 0)
    return makeError("model must contain at least one linear attention layer");
  if (getNumFullLayers() == 0)
    return makeError("model must contain at least one full attention layer");

  return llvm::Error::success();
}

int64_t ModelConfig::getNumLinearLayers() const {
  return llvm::count_if(layerTypes, [](LayerKind kind) {
    return kind == LayerKind::LinearAttention;
  });
}

int64_t ModelConfig::getNumFullLayers() const {
  return static_cast<int64_t>(layerTypes.size()) - getNumLinearLayers();
}

int64_t ModelConfig::getRotaryDim() const {
  return static_cast<int64_t>(text.headDim * text.partialRotaryFactor);
}

int64_t ModelConfig::getQProjOutDim() const {
  return text.numAttentionHeads * text.headDim * 2;
}

int64_t ModelConfig::getKvDim() const {
  return text.numKeyValueHeads * text.headDim;
}

int64_t ModelConfig::getLinearQkvDim() const {
  return text.linearNumKeyHeads * text.linearKeyHeadDim * 2 +
         text.linearNumValueHeads * text.linearValueHeadDim;
}

int64_t ModelConfig::getLinearValueDim() const {
  return text.linearNumValueHeads * text.linearValueHeadDim;
}

bool ModelConfig::isLinearLayer(int64_t index) const {
  return index >= 0 && index < static_cast<int64_t>(layerTypes.size()) &&
         layerTypes[index] == LayerKind::LinearAttention;
}
