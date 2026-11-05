//===- ModelBuilderTest.cpp - Unit tests for ModelBuilder -----------------===//
//
// Unit tests for graph construction and operation counts.
//
//===----------------------------------------------------------------------===//

#include "TestModel.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCP.h"
#include "llmcp/Model/ModelBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::llmcp;

TEST(ModelBuilderTest, BuildsCompleteGraph) {
  ModelConfig config = llmcp_test::makeTinyConfig();
  TensorMap tensors = llmcp_test::makeTinyTensors();

  MLIRContext context;
  auto moduleOrErr = ModelBuilder::build(config, tensors, &context);
  ASSERT_TRUE(!!moduleOrErr) << llvm::toString(moduleOrErr.takeError());
  OwningOpRef<ModuleOp> module = std::move(*moduleOrErr);

  ASSERT_TRUE(succeeded(module->verify()));

  auto numLayers =
      module->getOperation()->getAttrOfType<IntegerAttr>("llmcp.num_layers");
  ASSERT_TRUE(numLayers);
  EXPECT_EQ(numLayers.getInt(), 4);

  auto forward = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module.get(), "llmcp_forward_prefill"));
  ASSERT_TRUE(forward);

  size_t numLinear = 0;
  size_t numRmsNorm = 0;
  size_t numDelta = 0;
  size_t numAttention = 0;
  size_t numConv = 0;
  size_t numGatedNorm = 0;
  size_t numSiluMul = 0;
  size_t numAdd = 0;
  std::vector<int64_t> deltaIndices;
  std::vector<int64_t> attentionIndices;
  forward.walk([&](LinearOp) { ++numLinear; });
  forward.walk([&](RmsNormOp) { ++numRmsNorm; });
  forward.walk([&](GatedDeltaNetOp op) {
    ++numDelta;
    deltaIndices.push_back(op.getLayerIndex());
  });
  forward.walk([&](AttentionOp op) {
    ++numAttention;
    attentionIndices.push_back(op.getLayerIndex());
  });
  forward.walk([&](CausalConv1dOp) { ++numConv; });
  forward.walk([&](GatedRmsNormOp) { ++numGatedNorm; });
  forward.walk([&](SiluMulOp) { ++numSiluMul; });
  forward.walk([&](AddOp) { ++numAdd; });

  EXPECT_EQ(numLinear, 31u);
  EXPECT_EQ(numRmsNorm, 9u);
  EXPECT_EQ(numDelta, 3u);
  EXPECT_EQ(numAttention, 1u);
  EXPECT_EQ(numConv, 3u);
  EXPECT_EQ(numGatedNorm, 3u);
  EXPECT_EQ(numSiluMul, 4u);
  EXPECT_EQ(numAdd, 8u);

  ASSERT_EQ(deltaIndices.size(), 3u);
  EXPECT_EQ(deltaIndices[0], 0);
  EXPECT_EQ(deltaIndices[1], 1);
  EXPECT_EQ(deltaIndices[2], 2);
  ASSERT_EQ(attentionIndices.size(), 1u);
  EXPECT_EQ(attentionIndices[0], 0); // first full-attention state slot

  for (const char *name :
       {"llmcp_embed", "llmcp_forward_prefill", "llmcp_forward_decode",
        "llmcp_lm_head", "llmcp_model", "llmcp_model_prefill",
        "llmcp_model_decode"}) {
    EXPECT_TRUE(SymbolTable::lookupSymbolIn(module.get(), name));
  }

  for (const auto &entry : {std::pair<StringRef, StringRef>{
                                "llmcp_model_prefill", "llmcp_forward_prefill"},
                            {"llmcp_model_decode", "llmcp_forward_decode"}}) {
    auto func = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupSymbolIn(module.get(), entry.first));
    ASSERT_TRUE(func);
    bool found = false;
    func.walk([&](func::CallOp call) {
      if (call.getCallee() == entry.second)
        found = true;
    });
    EXPECT_TRUE(found) << entry.first.str() << " must call "
                       << entry.second.str();
  }

  auto model = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module.get(), "llmcp_model"));
  ASSERT_TRUE(model);
  size_t numCalls = 0;
  model.walk([&](func::CallOp) { ++numCalls; });
  EXPECT_EQ(numCalls, 3u);

  std::string text;
  llvm::raw_string_ostream os(text);
  module->print(os);
  EXPECT_NE(text.find("llmcp.embedding"), std::string::npos);
  EXPECT_NE(text.find("llmcp.attention"), std::string::npos);
  EXPECT_NE(text.find("llmcp.gated_delta_net"), std::string::npos);
  EXPECT_NE(text.find("llmcp.lm_head"), std::string::npos);
}

TEST(ModelBuilderTest, RejectsInvalidConfig) {
  ModelConfig config = llmcp_test::makeTinyConfig();
  config.text.hiddenAct = "gelu";
  TensorMap tensors = llmcp_test::makeTinyTensors();

  MLIRContext context;
  auto moduleOrErr = ModelBuilder::build(config, tensors, &context);
  EXPECT_FALSE(!!moduleOrErr);
  if (moduleOrErr)
    return;
  llvm::consumeError(moduleOrErr.takeError());
}
