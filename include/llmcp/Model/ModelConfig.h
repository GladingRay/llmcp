//===- ModelConfig.h - Qwen3.5 text model configuration ---------*- C++ -*-===//
//
// Parses and validates the text_config section of a Qwen3.5 checkpoint.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_MODEL_MODELCONFIG_H
#define LLMCP_MODEL_MODELCONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace llmcp {

/// Token mixer used by a decoder layer.
enum class LayerKind {
  LinearAttention,
  FullAttention,
};

llvm::StringRef stringifyLayerKind(LayerKind kind);

/// Text-model hyperparameters needed by the compiler and the runtime.
struct TextConfig {
  int64_t hiddenSize = 0;
  int64_t intermediateSize = 0;
  int64_t numHiddenLayers = 0;
  int64_t numAttentionHeads = 0;
  int64_t numKeyValueHeads = 0;
  int64_t headDim = 0;
  int64_t vocabSize = 0;
  int64_t maxPositionEmbeddings = 0;

  int64_t linearNumKeyHeads = 0;
  int64_t linearNumValueHeads = 0;
  int64_t linearKeyHeadDim = 0;
  int64_t linearValueHeadDim = 0;
  int64_t linearConvKernelDim = 0;
  int64_t fullAttentionInterval = 4;

  double rmsNormEps = 1.0e-6;
  double ropeTheta = 10000000.0;
  double partialRotaryFactor = 0.25;

  std::vector<int64_t> mropeSection;

  std::string hiddenAct = "silu";
  bool attentionBias = false;
  bool attnOutputGate = true;
  bool mropeInterleaved = true;
};

/// The parts of the checkpoint configuration that affect the compiled graph.
struct ModelConfig {
  std::string modelDir;
  TextConfig text;
  std::vector<LayerKind> layerTypes;
  bool tieWordEmbeddings = true;

  /// Reads `<modelDir>/config.json`.
  static llvm::Expected<ModelConfig> load(llvm::StringRef modelDir);

  /// Parses a config.json buffer. Exposed for unit tests.
  static llvm::Expected<ModelConfig> parse(llvm::StringRef json,
                                          llvm::StringRef modelDir);

  /// Checks shape relationships and the layer-type pattern.
  llvm::Error verify() const;

  int64_t getNumLayers() const { return text.numHiddenLayers; }
  int64_t getNumLinearLayers() const;
  int64_t getNumFullLayers() const;
  int64_t getRotaryDim() const;
  int64_t getQProjOutDim() const;
  int64_t getKvDim() const;
  int64_t getLinearQkvDim() const;
  int64_t getLinearValueDim() const;
  bool isLinearLayer(int64_t index) const;
};

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_MODEL_MODELCONFIG_H
