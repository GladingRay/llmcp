//===- LLMCPOps.cpp - LLMCP operation implementations ---------------------===//
//
// Verifiers and builders for LLMCP operations.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/IR/LLMCPOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

#define GET_OP_CLASSES
#include "llmcp/Dialect/LLMCP/IR/LLMCPOps.cpp.inc"

using namespace mlir;
using namespace mlir::llmcp;
using llvm::StringRef;

namespace {

/// Returns true when two dimensions are compatible, treating a dynamic
/// dimension as compatible with anything.
bool areCompatible(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

} // namespace

LogicalResult LinearOp::verify() {
  auto inputType = dyn_cast<ShapedType>(getInput().getType());
  auto weightType = dyn_cast<MemRefType>(getWeight().getType());
  auto outputType = dyn_cast<ShapedType>(getOutput().getType());

  if (!inputType || !inputType.hasRank() || inputType.getRank() != 2)
    return emitOpError("input must be a rank-2 tensor");
  if (!weightType || weightType.getRank() != 2)
    return emitOpError("weight must be a rank-2 memref");
  if (!outputType || !outputType.hasRank() || outputType.getRank() != 2)
    return emitOpError("output must be a rank-2 tensor");

  const bool transposeB = getTransposeB();
  const StringRef format = getWeightFormat();
  if (format == "q8_0") {
    if (transposeB)
      return emitOpError("q8_0 weights do not support transpose_b");
    if (!weightType.getElementType().isInteger(8))
      return emitOpError("q8_0 weight must be i8");
    const int64_t inputK = inputType.getDimSize(1);
    if (!ShapedType::isDynamic(inputK) && inputK % 32 != 0)
      return emitOpError("q8_0 requires K to be a multiple of 32");
    const int64_t expectedBytes = ShapedType::isDynamic(inputK)
                                      ? ShapedType::kDynamic
                                      : (inputK / 32) * 34;
    if (!areCompatible(weightType.getDimSize(1), expectedBytes))
      return emitOpError("q8_0 weight byte width must be (K/32)*34");
    if (!areCompatible(weightType.getDimSize(0), outputType.getDimSize(1)))
      return emitOpError("output inner dimension must match weight rows");
    if (!areCompatible(inputType.getDimSize(0), outputType.getDimSize(0)))
      return emitOpError("output outer dimension must match input");
    if (outputType.getElementType() != inputType.getElementType())
      return emitOpError("output element type must match input");
    return success();
  }
  if (format != "bf16")
    return emitOpError("unsupported weight_format: " + format);
  const int64_t inputK = inputType.getDimSize(1);
  const int64_t weightK =
      transposeB ? weightType.getDimSize(0) : weightType.getDimSize(1);
  const int64_t weightN =
      transposeB ? weightType.getDimSize(1) : weightType.getDimSize(0);

  if (!areCompatible(inputType.getDimSize(0), outputType.getDimSize(0)))
    return emitOpError("output outer dimension must match input");
  if (!areCompatible(inputK, weightK))
    return emitOpError("input and weight inner dimensions must match");
  if (!areCompatible(weightN, outputType.getDimSize(1)))
    return emitOpError("output inner dimension must match weight");

  if (inputType.getElementType() != weightType.getElementType())
    return emitOpError("input and weight element types must match");
  if (outputType.getElementType() != inputType.getElementType())
    return emitOpError("output element type must match input");

  return success();
}

LogicalResult RmsNormOp::verify() {
  auto inputType = dyn_cast<ShapedType>(getInput().getType());
  auto weightType = dyn_cast<MemRefType>(getWeight().getType());
  auto outputType = dyn_cast<ShapedType>(getOutput().getType());

  if (!inputType || !inputType.hasRank() || inputType.getRank() < 1)
    return emitOpError("input must be a ranked tensor with rank >= 1");
  if (!weightType || weightType.getRank() != 1)
    return emitOpError("weight must be a rank-1 memref");
  if (!outputType || !outputType.hasRank())
    return emitOpError("output must be a ranked tensor");

  if (inputType.getRank() != outputType.getRank())
    return emitOpError("output rank must match input rank");
  for (int64_t i = 0, e = inputType.getRank(); i < e; ++i)
    if (!areCompatible(inputType.getDimSize(i), outputType.getDimSize(i)))
      return emitOpError("output shape must match input shape");

  const int64_t inputLastDim = inputType.getDimSize(inputType.getRank() - 1);
  if (!areCompatible(inputLastDim, weightType.getDimSize(0)))
    return emitOpError("weight length must match the last input dimension");

  if (inputType.getElementType() != weightType.getElementType())
    return emitOpError("input and weight element types must match");
  if (inputType.getElementType() != outputType.getElementType())
    return emitOpError("output element type must match input");

  if (getEpsAttr().getValueAsDouble() <= 0.0)
    return emitOpError("eps must be positive");

  return success();
}

namespace {

/// Verifies that two ranked shaped values have the same rank and compatible
/// dimensions.
bool haveCompatibleShape(ShapedType lhs, ShapedType rhs) {
  if (lhs.getRank() != rhs.getRank())
    return false;
  for (int64_t i = 0, e = lhs.getRank(); i < e; ++i)
    if (!areCompatible(lhs.getDimSize(i), rhs.getDimSize(i)))
      return false;
  return true;
}

} // namespace

LogicalResult AddOp::verify() {
  auto lhs = dyn_cast<ShapedType>(getLhs().getType());
  auto rhs = dyn_cast<ShapedType>(getRhs().getType());
  auto result = dyn_cast<ShapedType>(getResult().getType());
  if (!lhs || !rhs || !result)
    return emitOpError("operands and result must be ranked tensors");
  if (!haveCompatibleShape(lhs, rhs) || !haveCompatibleShape(lhs, result))
    return emitOpError("operands and result must have the same shape");
  if (lhs.getElementType() != rhs.getElementType() ||
      lhs.getElementType() != result.getElementType())
    return emitOpError("operands and result must have the same element type");
  return success();
}

LogicalResult SiluMulOp::verify() {
  auto gate = dyn_cast<ShapedType>(getGate().getType());
  auto up = dyn_cast<ShapedType>(getUp().getType());
  auto result = dyn_cast<ShapedType>(getResult().getType());
  if (!gate || !up || !result)
    return emitOpError("operands and result must be ranked tensors");
  if (!haveCompatibleShape(gate, up) || !haveCompatibleShape(gate, result))
    return emitOpError("operands and result must have the same shape");
  if (gate.getElementType() != up.getElementType() ||
      gate.getElementType() != result.getElementType())
    return emitOpError("operands and result must have the same element type");
  return success();
}

LogicalResult EmbeddingOp::verify() {
  auto tokens = dyn_cast<ShapedType>(getTokens().getType());
  auto table = dyn_cast<MemRefType>(getTable().getType());
  auto result = dyn_cast<ShapedType>(getResult().getType());
  if (!tokens || !tokens.hasRank() || tokens.getRank() != 1)
    return emitOpError("tokens must be a rank-1 tensor");
  if (!table || table.getRank() != 2)
    return emitOpError("table must be a rank-2 memref");
  if (!result || !result.hasRank() || result.getRank() != 2)
    return emitOpError("result must be a rank-2 tensor");
  if (!areCompatible(result.getDimSize(0), tokens.getDimSize(0)))
    return emitOpError("result and tokens must have the same length");
  if (!areCompatible(result.getDimSize(1), table.getDimSize(1)))
    return emitOpError("result hidden dimension must match the table");
  if (result.getElementType() != table.getElementType())
    return emitOpError("result element type must match the table");
  return success();
}

LogicalResult LmHeadOp::verify() {
  auto hidden = dyn_cast<ShapedType>(getHidden().getType());
  auto table = dyn_cast<MemRefType>(getTable().getType());
  auto logits = dyn_cast<ShapedType>(getLogits().getType());
  if (!hidden || !hidden.hasRank() || hidden.getRank() != 2)
    return emitOpError("hidden must be a rank-2 tensor");
  if (!table || table.getRank() != 2)
    return emitOpError("table must be a rank-2 memref");
  if (!logits || !logits.hasRank() || logits.getRank() != 2)
    return emitOpError("logits must be a rank-2 tensor");
  if (!areCompatible(logits.getDimSize(0), hidden.getDimSize(0)))
    return emitOpError("logits and hidden must have the same length");
  if (getWeightFormat() == "q8_0") {
    if (!table.getElementType().isInteger(8))
      return emitOpError("q8_0 table must be i8");
    const int64_t hiddenDim = hidden.getDimSize(1);
    if (!ShapedType::isDynamic(hiddenDim) && hiddenDim % 32 != 0)
      return emitOpError("q8_0 requires hidden size to be a multiple of 32");
    const int64_t expectedBytes = ShapedType::isDynamic(hiddenDim)
                                      ? ShapedType::kDynamic
                                      : (hiddenDim / 32) * 34;
    if (!areCompatible(table.getDimSize(1), expectedBytes))
      return emitOpError("q8_0 table byte width must be (K/32)*34");
    if (!areCompatible(logits.getDimSize(1), table.getDimSize(0)))
      return emitOpError("logits vocabulary dimension must match the table");
    return success();
  }
  if (!areCompatible(hidden.getDimSize(1), table.getDimSize(1)))
    return emitOpError("hidden dimension must match the table");
  if (!areCompatible(logits.getDimSize(1), table.getDimSize(0)))
    return emitOpError("logits vocabulary dimension must match the table");
  if (getWeightFormat() != "bf16")
    return emitOpError("unsupported weight_format: " + getWeightFormat());
  if (hidden.getElementType() != table.getElementType())
    return emitOpError("hidden element type must match the table");
  if (!logits.getElementType().isF32())
    return emitOpError("logits must be f32");
  return success();
}

LogicalResult CausalConv1dOp::verify() {
  auto input = dyn_cast<ShapedType>(getInput().getType());
  auto weight = dyn_cast<MemRefType>(getWeight().getType());
  auto state = dyn_cast<MemRefType>(getState().getType());
  auto result = dyn_cast<ShapedType>(getResult().getType());
  if (!input || !input.hasRank() || input.getRank() != 2)
    return emitOpError("input must be a rank-2 tensor");
  if (!weight || weight.getRank() != 3)
    return emitOpError("weight must be a rank-3 memref");
  if (!state || state.getRank() != 3)
    return emitOpError("state must be a rank-3 memref");
  if (!result || !result.hasRank() || result.getRank() != 2)
    return emitOpError("result must be a rank-2 tensor");

  const int64_t channels = input.getDimSize(1);
  if (!areCompatible(channels, weight.getDimSize(0)))
    return emitOpError("input channels must match weight channels");
  if (!areCompatible(channels, state.getDimSize(2)))
    return emitOpError("input channels must match state channels");
  if (!areCompatible(getKernelSize(), weight.getDimSize(2)))
    return emitOpError("kernel_size must match the weight kernel dimension");
  if (!areCompatible(state.getDimSize(1), getKernelSize() - 1))
    return emitOpError("state must hold kernel_size - 1 tokens");
  if (getLayerIndex() < 0 || getLayerIndex() >= state.getDimSize(0))
    return emitOpError("layer_index is out of range for the conv state");
  if (!haveCompatibleShape(input, result))
    return emitOpError("result shape must match input shape");
  if (input.getElementType() != weight.getElementType() ||
      input.getElementType() != state.getElementType() ||
      input.getElementType() != result.getElementType())
    return emitOpError("input, weight, state, and result must have the same "
                       "element type");
  return success();
}

LogicalResult GatedDeltaNetOp::verify() {
  auto conv = dyn_cast<ShapedType>(getConv().getType());
  auto a = dyn_cast<ShapedType>(getA().getType());
  auto b = dyn_cast<ShapedType>(getB().getType());
  auto aLog = dyn_cast<MemRefType>(getALog().getType());
  auto dtBias = dyn_cast<MemRefType>(getDtBias().getType());
  auto state = dyn_cast<MemRefType>(getState().getType());
  auto output = dyn_cast<ShapedType>(getOutput().getType());
  if (!conv || !conv.hasRank() || conv.getRank() != 2)
    return emitOpError("conv must be a rank-2 tensor");
  if (!a || !a.hasRank() || a.getRank() != 2 || !b || !b.hasRank() ||
      b.getRank() != 2)
    return emitOpError("a and b must be rank-2 tensors");
  if (!aLog || aLog.getRank() != 1)
    return emitOpError("a_log must be a rank-1 memref");
  if (!dtBias || dtBias.getRank() != 1)
    return emitOpError("dt_bias must be a rank-1 memref");
  if (!state || state.getRank() != 4)
    return emitOpError("state must be a rank-4 memref");
  if (!output || !output.hasRank() || output.getRank() != 2)
    return emitOpError("output must be a rank-2 tensor");

  const int64_t numK = getNumKHeads();
  const int64_t numV = getNumVHeads();
  const int64_t headK = getHeadKDim();
  const int64_t headV = getHeadVDim();
  const int64_t qkv = 2 * numK * headK + numV * headV;
  if (!areCompatible(conv.getDimSize(1), qkv))
    return emitOpError("conv last dimension does not match head configuration");
  if (!areCompatible(a.getDimSize(1), numV) ||
      !areCompatible(b.getDimSize(1), numV))
    return emitOpError("a and b last dimension must equal num_v_heads");
  if (!areCompatible(aLog.getDimSize(0), numV) ||
      !areCompatible(dtBias.getDimSize(0), numV))
    return emitOpError("a_log and dt_bias length must equal num_v_heads");
  if (!areCompatible(state.getDimSize(1), numV) ||
      !areCompatible(state.getDimSize(2), headK) ||
      !areCompatible(state.getDimSize(3), headV))
    return emitOpError("state shape must be [layers, num_v_heads, head_k_dim, "
                       "head_v_dim]");
  if (!areCompatible(output.getDimSize(1), numV * headV))
    return emitOpError("output last dimension must equal num_v_heads * "
                       "head_v_dim");
  if (!areCompatible(conv.getDimSize(0), output.getDimSize(0)))
    return emitOpError("conv and output sequence lengths must match");
  if (getLayerIndex() < 0 || getLayerIndex() >= state.getDimSize(0))
    return emitOpError("layer_index is out of range for the delta state");
  const StringRef mode = getMode();
  if (mode != "prefill" && mode != "decode")
    return emitOpError("mode must be 'prefill' or 'decode'");
  return success();
}

LogicalResult AttentionOp::verify() {
  auto qg = dyn_cast<ShapedType>(getQg().getType());
  auto k = dyn_cast<ShapedType>(getK().getType());
  auto v = dyn_cast<ShapedType>(getV().getType());
  auto qNorm = dyn_cast<MemRefType>(getQNorm().getType());
  auto kNorm = dyn_cast<MemRefType>(getKNorm().getType());
  auto cos = dyn_cast<ShapedType>(getCos().getType());
  auto sin = dyn_cast<ShapedType>(getSin().getType());
  auto cache = dyn_cast<MemRefType>(getKvCache().getType());
  auto output = dyn_cast<ShapedType>(getOutput().getType());
  if (!qg || !qg.hasRank() || qg.getRank() != 2)
    return emitOpError("qg must be a rank-2 tensor");
  if (!k || !k.hasRank() || k.getRank() != 2 || !v || !v.hasRank() ||
      v.getRank() != 2)
    return emitOpError("k and v must be rank-2 tensors");
  if (!qNorm || qNorm.getRank() != 1 || !kNorm || kNorm.getRank() != 1)
    return emitOpError("q_norm and k_norm must be rank-1 memrefs");
  if (!cos || !cos.hasRank() || cos.getRank() != 2 || !sin || !sin.hasRank() ||
      sin.getRank() != 2)
    return emitOpError("cos and sin must be rank-2 tensors");
  if (!cache || cache.getRank() != 5)
    return emitOpError("kv_cache must be a rank-5 memref");
  if (!output || !output.hasRank() || output.getRank() != 2)
    return emitOpError("output must be a rank-2 tensor");

  const int64_t numHeads = getNumHeads();
  const int64_t numKvHeads = getNumKvHeads();
  const int64_t headDim = getHeadDim();
  const int64_t rotaryDim = getRotaryDim();
  if (!areCompatible(qg.getDimSize(1), numHeads * headDim * 2))
    return emitOpError("qg last dimension must equal num_heads * head_dim * 2");
  if (!areCompatible(k.getDimSize(1), numKvHeads * headDim) ||
      !areCompatible(v.getDimSize(1), numKvHeads * headDim))
    return emitOpError("k and v last dimension must equal num_kv_heads * "
                       "head_dim");
  if (!areCompatible(qNorm.getDimSize(0), headDim) ||
      !areCompatible(kNorm.getDimSize(0), headDim))
    return emitOpError("q_norm and k_norm length must equal head_dim");
  if (!areCompatible(cos.getDimSize(1), rotaryDim) ||
      !areCompatible(sin.getDimSize(1), rotaryDim))
    return emitOpError("cos and sin last dimension must equal rotary_dim");
  if (!areCompatible(qg.getDimSize(0), k.getDimSize(0)) ||
      !areCompatible(qg.getDimSize(0), v.getDimSize(0)) ||
      !areCompatible(qg.getDimSize(0), cos.getDimSize(0)) ||
      !areCompatible(qg.getDimSize(0), sin.getDimSize(0)))
    return emitOpError("qg, k, v, cos, and sin sequence lengths must match");
  if (!areCompatible(cache.getDimSize(1), 2) ||
      !areCompatible(cache.getDimSize(2), numKvHeads) ||
      !areCompatible(cache.getDimSize(4), headDim))
    return emitOpError("kv_cache shape must be [layers, 2, num_kv_heads, "
                       "capacity, head_dim]");
  if (getLayerIndex() < 0 || getLayerIndex() >= cache.getDimSize(0))
    return emitOpError("layer_index is out of range for the kv cache");
  if (!areCompatible(output.getDimSize(0), qg.getDimSize(0)) ||
      !areCompatible(output.getDimSize(1), numHeads * headDim))
    return emitOpError("output shape must be [sequence, num_heads * head_dim]");
  const StringRef mode = getMode();
  if (mode != "prefill" && mode != "decode")
    return emitOpError("mode must be 'prefill' or 'decode'");
  return success();
}

LogicalResult GatedRmsNormOp::verify() {
  auto input = dyn_cast<ShapedType>(getInput().getType());
  auto gate = dyn_cast<ShapedType>(getGate().getType());
  auto weight = dyn_cast<MemRefType>(getWeight().getType());
  auto result = dyn_cast<ShapedType>(getResult().getType());
  if (!input || !input.hasRank() || input.getRank() != 2)
    return emitOpError("input must be a rank-2 tensor");
  if (!gate || !gate.hasRank() || gate.getRank() != 2)
    return emitOpError("gate must be a rank-2 tensor");
  if (!weight || weight.getRank() != 1)
    return emitOpError("weight must be a rank-1 memref");
  if (!result || !result.hasRank() || result.getRank() != 2)
    return emitOpError("result must be a rank-2 tensor");
  if (!haveCompatibleShape(input, gate) || !haveCompatibleShape(input, result))
    return emitOpError("input, gate, and result shapes must match");
  if (!areCompatible(weight.getDimSize(0), getGroupSize()))
    return emitOpError("weight length must equal group_size");
  const int64_t lastDim = input.getDimSize(1);
  if (!ShapedType::isDynamic(lastDim) && (lastDim % getGroupSize()) != 0)
    return emitOpError("last input dimension must be divisible by group_size");
  if (getEpsAttr().getValueAsDouble() <= 0.0)
    return emitOpError("eps must be positive");
  return success();
}
