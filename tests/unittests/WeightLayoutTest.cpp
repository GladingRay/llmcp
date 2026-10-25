#include "llmcp/Model/WeightLayout.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <string>
#include <utility>

using namespace mlir::llmcp;

namespace {

void addTensor(TensorMap &tensors, llvm::StringRef name, llvm::StringRef dtype,
               llvm::ArrayRef<int64_t> shape) {
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

ModelConfig makeConfig() {
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

TensorMap makeTensors() {
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

void expectLayoutError(llvm::Expected<WeightLayout> layout) {
  if (layout) {
    ADD_FAILURE() << "expected WeightLayout::build to fail";
    return;
  }
  llvm::consumeError(layout.takeError());
}

} // namespace

TEST(WeightLayoutTest, BuildsAlignedArena) {
  ModelConfig config = makeConfig();
  TensorMap tensors = makeTensors();

  auto layout = WeightLayout::build(config, tensors);
  ASSERT_TRUE(!!layout) << llvm::toString(layout.takeError());

  EXPECT_EQ(layout->getHeaderBytes(), 64u);
  EXPECT_EQ(layout->getEmbeddingOffset(), 64u);
  EXPECT_EQ(layout->getNumGroups(), 1);
  EXPECT_GT(layout->getGroupStride(), 0u);
  EXPECT_GT(layout->getTotalBytes(), 64u);
  EXPECT_EQ(layout->getTotalBytes() % 64, 0u);
  EXPECT_EQ(layout->getEntries().size(), 55u);

  for (const WeightEntry &entry : layout->getEntries())
    EXPECT_EQ(entry.offset % 64, 0u) << entry.name;

  auto qkv = layout->getOffset(
      "model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
  ASSERT_TRUE(qkv.has_value());
  EXPECT_GE(*qkv, layout->getEmbeddingOffset());

  EXPECT_FALSE(layout->getOffset("does.not.exist").has_value());
}

TEST(WeightLayoutTest, RejectsMissingTensor) {
  ModelConfig config = makeConfig();
  TensorMap tensors = makeTensors();
  ASSERT_TRUE(
      tensors.erase("model.language_model.layers.2.mlp.down_proj.weight"));
  expectLayoutError(WeightLayout::build(config, tensors));
}

TEST(WeightLayoutTest, RejectsWrongShape) {
  ModelConfig config = makeConfig();
  TensorMap tensors = makeTensors();
  auto it = tensors.find("model.language_model.embed_tokens.weight");
  ASSERT_NE(it, tensors.end());
  it->second.shape = {31, 8};
  expectLayoutError(WeightLayout::build(config, tensors));
}
