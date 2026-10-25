//===- LLMCPOps.cpp - LLMCP operation implementations
//----------------------===//
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
