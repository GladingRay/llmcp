#ifndef LLMCP_TESTS_UNITTESTS_TESTMODEL_H
#define LLMCP_TESTS_UNITTESTS_TESTMODEL_H

#include "llmcp/Model/ModelConfig.h"
#include "llmcp/Model/Safetensors.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <string>
#include <utility>

namespace llmcp_test {

using namespace mlir::llmcp;

inline void addTensor(TensorMap &tensors, llvm::StringRef name,
                      llvm::StringRef dtype, llvm::ArrayRef<int64_t> shape) {
  TensorInfo info;
  info.name = name.str();
  info.dtype = dtype.str();
  info.shape.assign(shape.begin(), shape.end());

  auto elementBytes = getSafetensorsDTypeSize(dtype);
  EXPECT_TRUE(!!elementBytes);
  if (!elementBytes) {
    llvm::consumeError(elementBytes.takeError());
    return;
  }
  info.dataSize = info.getNumElements() * *elementBytes;
  tensors.try_emplace(name, std::move(info));
}

inline ModelConfig makeTinyConfig() {
  ModelConfig config;
  config.modelDir = "/tmp/qwen3.5-tiny";
  config.tieWordEmbeddings = true;

  TextConfig &tc = config.text;
  tc.hiddenSize = 8;
  tc.intermediateSize = 16;
  tc.numHiddenLayers = 4;
  tc.numAttentionHeads = 2;
  tc.numKeyValueHeads = 1;
  tc.headDim = 4;
  tc.vocabSize = 32;
  tc.maxPositionEmbeddings = 64;
  tc.linearNumKeyHeads = 2;
  tc.linearNumValueHeads = 4;
  tc.linearKeyHeadDim = 2;
  tc.linearValueHeadDim = 2;
  tc.linearConvKernelDim = 2;
  tc.fullAttentionInterval = 4;
  tc.partialRotaryFactor = 0.5;
  tc.mropeSection = {1};
  tc.hiddenAct = "silu";
  config.layerTypes = {LayerKind::LinearAttention, LayerKind::LinearAttention,
                       LayerKind::LinearAttention, LayerKind::FullAttention};
  return config;
}

inline TensorMap makeTinyTensors() {
  TensorMap tensors;
  addTensor(tensors, "model.language_model.embed_tokens.weight", "BF16",
            {32, 8});
  addTensor(tensors, "model.language_model.norm.weight", "BF16", {8});

  for (int64_t layer = 0; layer < 4; ++layer) {
    const bool isLinear = layer != 3;
    const std::string prefix =
        (llvm::Twine("model.language_model.layers.") + llvm::Twine(layer) + ".")
            .str();

    addTensor(tensors, prefix + "input_layernorm.weight", "BF16", {8});
    addTensor(tensors, prefix + "post_attention_layernorm.weight", "BF16", {8});

    if (isLinear) {
      addTensor(tensors, prefix + "linear_attn.in_proj_qkv.weight", "BF16",
                {16, 8});
      addTensor(tensors, prefix + "linear_attn.in_proj_z.weight", "BF16",
                {8, 8});
      addTensor(tensors, prefix + "linear_attn.in_proj_a.weight", "BF16",
                {4, 8});
      addTensor(tensors, prefix + "linear_attn.in_proj_b.weight", "BF16",
                {4, 8});
      addTensor(tensors, prefix + "linear_attn.conv1d.weight", "BF16",
                {16, 1, 2});
      addTensor(tensors, prefix + "linear_attn.dt_bias", "BF16", {4});
      addTensor(tensors, prefix + "linear_attn.A_log", "F32", {4});
      addTensor(tensors, prefix + "linear_attn.norm.weight", "F32", {2});
      addTensor(tensors, prefix + "linear_attn.out_proj.weight", "BF16",
                {8, 8});
    } else {
      addTensor(tensors, prefix + "self_attn.q_proj.weight", "BF16", {16, 8});
      addTensor(tensors, prefix + "self_attn.k_proj.weight", "BF16", {4, 8});
      addTensor(tensors, prefix + "self_attn.v_proj.weight", "BF16", {4, 8});
      addTensor(tensors, prefix + "self_attn.o_proj.weight", "BF16", {8, 8});
      addTensor(tensors, prefix + "self_attn.q_norm.weight", "BF16", {4});
      addTensor(tensors, prefix + "self_attn.k_norm.weight", "BF16", {4});
    }

    addTensor(tensors, prefix + "mlp.gate_proj.weight", "BF16", {16, 8});
    addTensor(tensors, prefix + "mlp.up_proj.weight", "BF16", {16, 8});
    addTensor(tensors, prefix + "mlp.down_proj.weight", "BF16", {8, 16});
  }
  return tensors;
}

} // namespace llmcp_test

#endif // LLMCP_TESTS_UNITTESTS_TESTMODEL_H
