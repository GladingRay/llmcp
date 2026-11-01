//===- ModelBuilder.h - Build the llmcp model graph -------------*- C++ -*-===//
//
// Builds the complete llmcp prefill graph from a model configuration.
//
//===----------------------------------------------------------------------===//

#ifndef LLMCP_MODEL_MODELBUILDER_H
#define LLMCP_MODEL_MODELBUILDER_H

#include "llmcp/Model/ModelConfig.h"
#include "llmcp/Model/Safetensors.h"
#include "llmcp/Model/WeightLayout.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace llmcp {

/// Builds the complete llmcp compute graph for a Qwen3.5 text model.
///
/// The generated module contains:
///   `llmcp_embed`, `llmcp_forward`, `llmcp_lm_head`, and `llmcp_model`.
/// `llmcp_forward` is unrolled: one op sequence per decoder layer, so the dump
/// is directly inspectable. Weight operands are `memref.view` slices of one
/// byte-addressed weight arena.
class ModelBuilder {
public:
  static llvm::Expected<OwningOpRef<ModuleOp>>
  build(const ModelConfig &config, const TensorMap &tensors,
        MLIRContext *context,
        QuantizationMode quantization = QuantizationMode::None);
};

} // namespace llmcp
} // namespace mlir

#endif // LLMCP_MODEL_MODELBUILDER_H
