//===- WeightLayout.cpp - Deterministic weight arena layout ---------------===//
//
// Computes deterministic byte offsets for the packed weight arena.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Model/WeightLayout.h"
#include "llmcp/Support/PackedWeights.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <utility>

using namespace mlir::llmcp;
using llvm::ArrayRef;
using llvm::StringRef;

namespace {

constexpr uint64_t kTensorAlignment = 64;
constexpr uint64_t kHeaderBytes = ::llmcp::kPackedWeightsHeaderBytes;

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

uint64_t alignTo(uint64_t value) {
  return (value + kTensorAlignment - 1) & ~(kTensorAlignment - 1);
}

std::string makeName(StringRef prefix, StringRef suffix) {
  return (llvm::Twine(prefix) + suffix).str();
}

llvm::Error checkShape(StringRef name, ArrayRef<int64_t> actual,
                       ArrayRef<int64_t> expected) {
  if (actual.size() != expected.size())
    return makeError(llvm::Twine("tensor '") + name + "' has rank " +
                     llvm::Twine(actual.size()) + ", expected " +
                     llvm::Twine(expected.size()));
  for (size_t i = 0; i < expected.size(); ++i) {
    if (actual[i] != expected[i])
      return makeError(llvm::Twine("tensor '") + name + "' dimension " +
                       llvm::Twine(i) + " is " + llvm::Twine(actual[i]) +
                       ", expected " + llvm::Twine(expected[i]));
  }
  return llvm::Error::success();
}

} // namespace

llvm::Error WeightLayout::addTensor(StringRef name,
                                    ArrayRef<int64_t> expectedShape,
                                    const TensorMap &tensors,
                                    uint64_t &cursor) {
  auto it = tensors.find(name);
  if (it == tensors.end())
    return makeError(llvm::Twine("missing tensor '") + name + "'");
  if (llvm::Error error = checkShape(name, it->second.shape, expectedShape))
    return error;

  const uint64_t offset = alignTo(cursor);
  WeightEntry entry;
  entry.name = name.str();
  entry.offset = offset;
  entry.sourceSize = it->second.dataSize;
  entry.size = it->second.dataSize;
  entry.dtype = it->second.dtype;
  entry.shape = it->second.shape;
  if (quantization_ == QuantizationMode::Q8_0 &&
      entry.name != "model.language_model.embed_tokens.weight" &&
      entry.shape.size() == 2 && entry.shape[1] % kQ8_0BlockSize == 0 &&
      (entry.dtype == "BF16" || entry.dtype == "F32")) {
    auto quantizedSize = getQ8_0TensorSize(entry.shape);
    if (!quantizedSize)
      return quantizedSize.takeError();
    entry.size = *quantizedSize;
    entry.dtype = "Q8_0";
    entry.quantized = true;
  }
  entries_.push_back(std::move(entry));
  offsets_[name] = offset;
  cursor = offset + entry.size;
  return llvm::Error::success();
}

llvm::Error WeightLayout::addLayerTensor(StringRef prefix, StringRef suffix,
                                         ArrayRef<int64_t> expectedShape,
                                         const TensorMap &tensors,
                                         uint64_t &cursor) {
  return addTensor(makeName(prefix, suffix), expectedShape, tensors, cursor);
}

llvm::Expected<WeightLayout>
WeightLayout::build(const ModelConfig &config, const TensorMap &tensors,
                    QuantizationMode quantization) {
  const int64_t numLayers = config.getNumLayers();
  const int64_t groupSize = config.text.fullAttentionInterval;
  if (groupSize <= 0 || numLayers % groupSize != 0)
    return makeError(
        "num_hidden_layers must be divisible by full_attention_interval");
  if (!config.tieWordEmbeddings)
    return makeError("untied lm_head is not supported yet");

  const int64_t numGroups = numLayers / groupSize;
  for (int64_t group = 1; group < numGroups; ++group) {
    for (int64_t i = 0; i < groupSize; ++i) {
      if (config.layerTypes[group * groupSize + i] != config.layerTypes[i])
        return makeError("all layer groups must have the same mixer pattern");
    }
  }

  WeightLayout layout;
  layout.quantization_ = quantization;
  layout.headerBytes_ = kHeaderBytes;
  layout.numGroups_ = numGroups;
  uint64_t cursor = kHeaderBytes;

  const auto &tc = config.text;
  const int64_t hidden = tc.hiddenSize;
  const int64_t intermediate = tc.intermediateSize;
  const int64_t vocab = tc.vocabSize;
  const int64_t qkvDim = config.getLinearQkvDim();
  const int64_t valueDim = config.getLinearValueDim();
  const int64_t numValueHeads = tc.linearNumValueHeads;
  const int64_t valueHeadDim = tc.linearValueHeadDim;
  const int64_t qProjOut = config.getQProjOutDim();
  const int64_t kvDim = config.getKvDim();
  const int64_t numHeads = tc.numAttentionHeads;
  const int64_t headDim = tc.headDim;

  const llvm::SmallVector<int64_t, 4> embedShape{vocab, hidden};
  if (llvm::Error error =
          layout.addTensor("model.language_model.embed_tokens.weight",
                           embedShape, tensors, cursor))
    return std::move(error);
  layout.embeddingOffset_ =
      *layout.getOffset("model.language_model.embed_tokens.weight");

  const llvm::SmallVector<int64_t, 4> normShape{hidden};
  const llvm::SmallVector<int64_t, 4> mlpUpShape{intermediate, hidden};
  const llvm::SmallVector<int64_t, 4> mlpDownShape{hidden, intermediate};

  for (int64_t group = 0; group < numGroups; ++group) {
    const uint64_t groupStart = cursor;
    for (int64_t i = 0; i < groupSize; ++i) {
      const int64_t layer = group * groupSize + i;
      const std::string prefix = (llvm::Twine("model.language_model.layers.") +
                                  llvm::Twine(layer) + ".")
                                     .str();

      if (config.isLinearLayer(layer)) {
        if (llvm::Error error = layout.addLayerTensor(
                prefix, "input_layernorm.weight", normShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error =
                layout.addLayerTensor(prefix, "post_attention_layernorm.weight",
                                      normShape, tensors, cursor))
          return std::move(error);

        const llvm::SmallVector<int64_t, 4> qkvShape{qkvDim, hidden};
        const llvm::SmallVector<int64_t, 4> zShape{valueDim, hidden};
        const llvm::SmallVector<int64_t, 4> abShape{numValueHeads, hidden};
        const llvm::SmallVector<int64_t, 4> abVectorShape{numValueHeads};
        const llvm::SmallVector<int64_t, 4> valueHeadShape{valueHeadDim};
        const llvm::SmallVector<int64_t, 4> outShape{hidden, valueDim};
        const llvm::SmallVector<int64_t, 4> convShape{qkvDim, 1,
                                                      tc.linearConvKernelDim};

        if (llvm::Error error =
                layout.addLayerTensor(prefix, "linear_attn.in_proj_qkv.weight",
                                      qkvShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error =
                layout.addLayerTensor(prefix, "linear_attn.in_proj_z.weight",
                                      zShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error =
                layout.addLayerTensor(prefix, "linear_attn.in_proj_a.weight",
                                      abShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error =
                layout.addLayerTensor(prefix, "linear_attn.in_proj_b.weight",
                                      abShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error =
                layout.addLayerTensor(prefix, "linear_attn.conv1d.weight",
                                      convShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error = layout.addLayerTensor(
                prefix, "linear_attn.dt_bias", abVectorShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error = layout.addLayerTensor(
                prefix, "linear_attn.A_log", abVectorShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error =
                layout.addLayerTensor(prefix, "linear_attn.norm.weight",
                                      valueHeadShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error =
                layout.addLayerTensor(prefix, "linear_attn.out_proj.weight",
                                      outShape, tensors, cursor))
          return std::move(error);
      } else {
        if (llvm::Error error = layout.addLayerTensor(
                prefix, "input_layernorm.weight", normShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error =
                layout.addLayerTensor(prefix, "post_attention_layernorm.weight",
                                      normShape, tensors, cursor))
          return std::move(error);

        const llvm::SmallVector<int64_t, 4> qShape{qProjOut, hidden};
        const llvm::SmallVector<int64_t, 4> kvShape{kvDim, hidden};
        const llvm::SmallVector<int64_t, 4> oShape{hidden, numHeads * headDim};
        const llvm::SmallVector<int64_t, 4> headShape{headDim};

        if (llvm::Error error = layout.addLayerTensor(
                prefix, "self_attn.q_proj.weight", qShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error = layout.addLayerTensor(
                prefix, "self_attn.k_proj.weight", kvShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error = layout.addLayerTensor(
                prefix, "self_attn.v_proj.weight", kvShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error = layout.addLayerTensor(
                prefix, "self_attn.o_proj.weight", oShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error = layout.addLayerTensor(
                prefix, "self_attn.q_norm.weight", headShape, tensors, cursor))
          return std::move(error);
        if (llvm::Error error = layout.addLayerTensor(
                prefix, "self_attn.k_norm.weight", headShape, tensors, cursor))
          return std::move(error);
      }

      if (llvm::Error error = layout.addLayerTensor(
              prefix, "mlp.gate_proj.weight", mlpUpShape, tensors, cursor))
        return std::move(error);
      if (llvm::Error error = layout.addLayerTensor(
              prefix, "mlp.up_proj.weight", mlpUpShape, tensors, cursor))
        return std::move(error);
      if (llvm::Error error = layout.addLayerTensor(
              prefix, "mlp.down_proj.weight", mlpDownShape, tensors, cursor))
        return std::move(error);
    }

    const uint64_t groupBytes = alignTo(cursor - groupStart);
    if (group == 0) {
      layout.groupStride_ = groupBytes;
    } else if (layout.groupStride_ != groupBytes) {
      return makeError("layer groups have different weight sizes");
    }
    cursor = groupStart + layout.groupStride_;
  }

  if (llvm::Error error = layout.addTensor("model.language_model.norm.weight",
                                           normShape, tensors, cursor))
    return std::move(error);
  layout.finalNormOffset_ =
      *layout.getOffset("model.language_model.norm.weight");

  layout.totalBytes_ = alignTo(cursor);
  return layout;
}

std::optional<uint64_t> WeightLayout::getOffset(StringRef name) const {
  auto it = offsets_.find(name);
  if (it == offsets_.end())
    return std::nullopt;
  return it->second;
}
