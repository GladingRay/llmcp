#include "llmcp/Model/ModelConfig.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <string>

using namespace mlir::llmcp;

namespace {

const char *kConfig = R"({
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "tie_word_embeddings": true,
  "text_config": {
    "attention_bias": false,
    "attn_output_gate": true,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9216,
    "layer_types": [
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention"
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "mamba_ssm_dtype": "float32",
    "num_attention_heads": 16,
    "num_hidden_layers": 4,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    },
    "tie_word_embeddings": true,
    "vocab_size": 248320
  }
})";

void expectError(llvm::Expected<ModelConfig> config) {
  if (config) {
    ADD_FAILURE() << "expected ModelConfig::parse to fail";
    return;
  }
  llvm::consumeError(config.takeError());
}

} // namespace

TEST(ModelConfigTest, ParsesQwen35TextConfig) {
  auto config = ModelConfig::parse(kConfig, "/tmp/qwen3.5-4b");
  ASSERT_TRUE(!!config) << llvm::toString(config.takeError());

  EXPECT_EQ(config->text.hiddenSize, 2560);
  EXPECT_EQ(config->text.intermediateSize, 9216);
  EXPECT_EQ(config->text.numHiddenLayers, 4);
  EXPECT_EQ(config->text.numAttentionHeads, 16);
  EXPECT_EQ(config->text.numKeyValueHeads, 4);
  EXPECT_EQ(config->text.headDim, 256);
  EXPECT_EQ(config->text.linearNumKeyHeads, 16);
  EXPECT_EQ(config->text.linearNumValueHeads, 32);
  EXPECT_EQ(config->text.linearConvKernelDim, 4);
  EXPECT_EQ(config->getRotaryDim(), 64);
  EXPECT_EQ(config->getQProjOutDim(), 8192);
  EXPECT_EQ(config->getKvDim(), 1024);
  EXPECT_EQ(config->getLinearQkvDim(), 8192);
  EXPECT_EQ(config->getLinearValueDim(), 4096);
  EXPECT_TRUE(config->tieWordEmbeddings);
  EXPECT_TRUE(config->isLinearLayer(0));
  EXPECT_TRUE(config->isLinearLayer(2));
  EXPECT_FALSE(config->isLinearLayer(3));
  EXPECT_EQ(config->getNumLinearLayers(), 3);
  EXPECT_EQ(config->getNumFullLayers(), 1);
}

TEST(ModelConfigTest, DerivesLayerTypes) {
  std::string json = kConfig;
  const std::string layerTypes = R"("layer_types": [
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention"
    ],)";
  const size_t pos = json.find(layerTypes);
  ASSERT_NE(pos, std::string::npos);
  json.erase(pos, layerTypes.size());

  auto config = ModelConfig::parse(json, "/tmp/qwen3.5-4b");
  ASSERT_TRUE(!!config) << llvm::toString(config.takeError());
  ASSERT_EQ(config->layerTypes.size(), 4u);
  EXPECT_TRUE(config->isLinearLayer(0));
  EXPECT_TRUE(config->isLinearLayer(2));
  EXPECT_FALSE(config->isLinearLayer(3));
}

TEST(ModelConfigTest, RejectsMissingHiddenSize) {
  std::string json = kConfig;
  const std::string field = R"("hidden_size": 2560,)";
  const size_t pos = json.find(field);
  ASSERT_NE(pos, std::string::npos);
  json.erase(pos, field.size());
  expectError(ModelConfig::parse(json, "/tmp/qwen3.5-4b"));
}

TEST(ModelConfigTest, RejectsBadMropeSection) {
  std::string json = kConfig;
  const std::string section = R"("mrope_section": [11, 11, 10])";
  const size_t pos = json.find(section);
  ASSERT_NE(pos, std::string::npos);
  json.replace(pos, section.size(), R"("mrope_section": [1, 1, 1])");
  expectError(ModelConfig::parse(json, "/tmp/qwen3.5-4b"));
}

TEST(ModelConfigTest, RejectsNonSiluActivation) {
  std::string json = kConfig;
  const std::string field = R"("hidden_act": "silu")";
  const size_t pos = json.find(field);
  ASSERT_NE(pos, std::string::npos);
  json.replace(pos, field.size(), R"("hidden_act": "gelu")");
  expectError(ModelConfig::parse(json, "/tmp/qwen3.5-4b"));
}
