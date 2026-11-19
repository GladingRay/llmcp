//===- Q8Matmul.cpp - Packed Q8 SDOT prefill GEMM -------------------------===//
//
// Reads the pre-packed four-column weights for an SDOT-by-element microkernel.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/MatmulLoweringInternal.h"
#include "llmcp/Support/Q8PackedLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::llmcp;

#if defined(__aarch64__) && !defined(__AARCH64EB__) &&                         \
    defined(LLMCP_ENABLE_SDOT)
namespace {

constexpr int64_t kQuantBlock = 32;
constexpr int64_t kLanes = kQ8PackedLanes;
constexpr int64_t kRowsPerTile = 4;
constexpr int64_t kGroupsPerTile = 2;

static_assert(kQ8PackedBlockBytes == kLanes * (kQuantBlock + 2));

class PackedQ8Matmul {
public:
  PackedQ8Matmul(PatternRewriter &builder, Location loc)
      : builder(builder), loc(loc), f32(builder.getF32Type()),
        i8(builder.getI8Type()), i32(builder.getI32Type()),
        i8x16(VectorType::get({16}, i8)), i8x8(VectorType::get({8}, i8)),
        i32x4(VectorType::get({4}, i32)), f32x4(VectorType::get({4}, f32)),
        f16x4(VectorType::get({4}, builder.getF16Type())), zero(index(0)),
        zeroF(
            builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(f32))) {}

  Value emit(Value input, Value weight, Value output, Value rows,
             int64_t blockCount, int64_t channels, StringRef activation);

private:
  Value index(int64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc, value);
  }

  Value add(Value base, int64_t offset) {
    return builder.create<arith::AddIOp>(loc, base, index(offset));
  }

  scf::ForOp loop(Value begin, Value end, int64_t step = 1,
                  ValueRange init = {}, bool parallel = false) {
    auto op = builder.create<scf::ForOp>(loc, begin, end, index(step), init);
    if (parallel)
      op->setAttr("llmcp.parallel", builder.getUnitAttr());
    builder.setInsertionPointToStart(op.getBody());
    return op;
  }

  Value load(VectorType type, Value source, ValueRange indices) {
    return builder.create<vector::LoadOp>(loc, type, source, indices);
  }

  Value splat(VectorType type, Value scalar) {
    return builder.create<vector::BroadcastOp>(loc, type, scalar);
  }

  void packActivations(Value input, Value rows, Value blocks, Value data,
                       Value scales);
  void emitTile(Value row, Value group, Value blocks, int64_t rowCount,
                Value aData, Value aScales, Value weight, Value output,
                Type outputElement, StringRef activation, int64_t blockCount);

  PatternRewriter &builder;
  Location loc;
  Type f32;
  Type i8;
  Type i32;
  VectorType i8x16;
  VectorType i8x8;
  VectorType i32x4;
  VectorType f32x4;
  VectorType f16x4;
  Value zero;
  Value zeroF;
};

void PackedQ8Matmul::packActivations(Value input, Value rows, Value blocks,
                                     Value data, Value scales) {
  auto rowLoop = loop(zero, rows, 1, {}, /*parallel=*/true);
  Value row = rowLoop.getInductionVar();
  Value rowTile = builder.create<arith::DivUIOp>(loc, row, index(kRowsPerTile));
  Value rowInTile =
      builder.create<arith::RemUIOp>(loc, row, index(kRowsPerTile));
  Value packedOffset =
      builder.create<arith::MulIOp>(loc, rowInTile, index(kQuantBlock));
  auto blockLoop = loop(zero, blocks);
  Value block = blockLoop.getInductionVar();
  Value offset = builder.create<arith::MulIOp>(loc, block, index(kQuantBlock));
  auto sourceType = cast<RankedTensorType>(input.getType());
  auto inputVector =
      VectorType::get({kQuantBlock}, sourceType.getElementType());
  auto floatVector = VectorType::get({kQuantBlock}, f32);
  auto byteVector = VectorType::get({kQuantBlock}, i8);
  auto intVector = VectorType::get({kQuantBlock}, i32);
  auto rowMap = AffineMap::getMinorIdentityMap(2, 1, builder.getContext());
  Value values = builder.create<vector::TransferReadOp>(
      loc, inputVector, input, ValueRange{row, offset}, rowMap,
      ArrayRef<bool>{true});
  if (sourceType.getElementType().isBF16())
    values = builder.create<arith::ExtFOp>(loc, floatVector, values);
  Value abs = builder.create<math::AbsFOp>(loc, values);
  Value maxAbs = builder.create<vector::ReductionOp>(
      loc, vector::CombiningKind::MAXNUMF, abs);
  Value range =
      builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(127));
  Value scale = builder.create<arith::DivFOp>(loc, maxAbs, range);
  builder.create<memref::StoreOp>(loc, scale, scales,
                                  ValueRange{rowTile, block, rowInTile});

  Value isZero = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                               maxAbs, zeroF);
  Value oneF =
      builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(1));
  Value reciprocal = builder.create<arith::DivFOp>(loc, oneF, scale);
  Value inverse =
      builder.create<arith::SelectOp>(loc, isZero, zeroF, reciprocal);
  Value scaled =
      builder.create<arith::MulFOp>(loc, values, splat(floatVector, inverse));
  Value rounded = builder.create<math::RoundEvenOp>(loc, scaled);
  Value integers = builder.create<arith::FPToSIOp>(loc, intVector, rounded);
  Value bytes = builder.create<arith::TruncIOp>(loc, byteVector, integers);
  builder.create<vector::StoreOp>(loc, bytes, data,
                                  ValueRange{rowTile, block, packedOffset});
  builder.setInsertionPointAfter(rowLoop);
}

void PackedQ8Matmul::emitTile(Value row, Value group, Value blocks,
                              int64_t rowCount, Value aData, Value aScales,
                              Value weight, Value output, Type outputElement,
                              StringRef activation, int64_t blockCount) {
  Value rowTile = builder.create<arith::DivUIOp>(loc, row, index(kRowsPerTile));
  Value firstRow =
      rowCount == kRowsPerTile
          ? zero
          : builder.create<arith::RemUIOp>(loc, row, index(kRowsPerTile));
  Value zeroF32x4 = splat(f32x4, zeroF);
  Value zeroI32x4 =
      builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(i32x4));
  SmallVector<Value> init(rowCount * kGroupsPerTile, zeroF32x4);
  auto blockLoop = loop(zero, blocks, 1, init);
  Value block = blockLoop.getInductionVar();
  SmallVector<Value> dots(rowCount * kGroupsPerTile, zeroI32x4);
  Value recordBytes =
      builder.create<arith::MulIOp>(loc, block, index(kQ8PackedBlockBytes));
  SmallVector<Value> records;
  for (int64_t g = 0; g < kGroupsPerTile; ++g) {
    Value globalGroup = add(group, g);
    Value groupBase = builder.create<arith::MulIOp>(
        loc, globalGroup, index(blockCount * kQ8PackedBlockBytes));
    records.push_back(
        builder.create<arith::AddIOp>(loc, groupBase, recordBytes).getResult());
  }

  // Each half of a 32-element block is four four-byte activation words.
  for (int64_t half = 0; half < kQuantBlock / 16; ++half) {
    SmallVector<Value> aWords;
    for (int64_t r = 0; r < rowCount; ++r) {
      Value aOffset = builder.create<arith::MulIOp>(loc, add(firstRow, r),
                                                    index(kQuantBlock));
      Value a = load(i8x16, aData, {rowTile, block, add(aOffset, half * 16)});
      aWords.push_back(builder.create<vector::BitCastOp>(loc, i32x4, a));
    }
    for (int64_t word = 0; word < 4; ++word) {
      SmallVector<Value, kGroupsPerTile> weights;
      for (int64_t g = 0; g < kGroupsPerTile; ++g) {
        int64_t offset =
            kQ8PackedScaleBytes + (half * 4 + word) * kQ8PackedChunkBytes;
        weights.push_back(load(i8x16, weight, {add(records[g], offset)}));
      }
      for (int64_t r = 0; r < rowCount; ++r) {
        SmallVector<int64_t, 4> mask(4, word);
        Value a =
            builder.create<vector::ShuffleOp>(loc, aWords[r], aWords[r], mask);
        a = builder.create<vector::BitCastOp>(loc, i8x16, a);
        for (int64_t g = 0; g < kGroupsPerTile; ++g) {
          Value &dot = dots[r * kGroupsPerTile + g];
          dot = builder
                    .create<LLVM::CallIntrinsicOp>(
                        loc, i32x4,
                        builder.getStringAttr("llvm.aarch64.neon.sdot"),
                        ValueRange{dot, weights[g], a})
                    .getResult(0);
        }
      }
    }
  }

  SmallVector<Value, kGroupsPerTile> weightScales;
  for (int64_t g = 0; g < kGroupsPerTile; ++g) {
    Value scaleBytes = load(i8x8, weight, {records[g]});
    Value scales = builder.create<vector::BitCastOp>(loc, f16x4, scaleBytes);
    weightScales.push_back(builder.create<arith::ExtFOp>(loc, f32x4, scales));
  }
  SmallVector<Value> accumulators;
  for (int64_t r = 0; r < rowCount; ++r) {
    Value scale = builder.create<memref::LoadOp>(
        loc, aScales, ValueRange{rowTile, block, add(firstRow, r)});
    Value broadcast = splat(f32x4, scale);
    for (int64_t g = 0; g < kGroupsPerTile; ++g) {
      int64_t tileIndex = r * kGroupsPerTile + g;
      Value dot = builder.create<arith::SIToFPOp>(loc, f32x4, dots[tileIndex]);
      Value factor =
          builder.create<arith::MulFOp>(loc, broadcast, weightScales[g]);
      accumulators.push_back(builder.create<math::FmaOp>(
          loc, dot, factor, blockLoop.getRegionIterArg(tileIndex)));
    }
  }
  builder.create<scf::YieldOp>(loc, accumulators);
  builder.setInsertionPointAfter(blockLoop);

  for (int64_t r = 0; r < rowCount; ++r) {
    for (int64_t g = 0; g < kGroupsPerTile; ++g) {
      Value result = blockLoop.getResult(r * kGroupsPerTile + g);
      Value column =
          builder.create<arith::MulIOp>(loc, add(group, g), index(kLanes));
      if (activation == "none") {
        if (outputElement.isBF16())
          result = builder.create<arith::TruncFOp>(
              loc, VectorType::get({kLanes}, outputElement), result);
        builder.create<vector::StoreOp>(loc, result, output,
                                        ValueRange{add(row, r), column});
        continue;
      }
      for (int64_t lane = 0; lane < kLanes; ++lane) {
        Value scalar = builder.create<vector::ExtractOp>(loc, result, lane);
        scalar = applyMatmulActivation(builder, loc, scalar, activation);
        if (outputElement.isBF16())
          scalar = builder.create<arith::TruncFOp>(loc, outputElement, scalar);
        builder.create<memref::StoreOp>(
            loc, scalar, output, ValueRange{add(row, r), add(column, lane)});
      }
    }
  }
}

Value PackedQ8Matmul::emit(Value input, Value weight, Value output, Value rows,
                           int64_t blockCount, int64_t channels,
                           StringRef activation) {
  auto outputType = cast<MemRefType>(output.getType());
  auto outputElement = outputType.getElementType();
  Value blocks = index(blockCount);
  Value tileRows =
      builder.create<arith::CeilDivUIOp>(loc, rows, index(kRowsPerTile));
  Value tail = builder.create<arith::RemUIOp>(loc, rows, index(kRowsPerTile));
  Value mainRows = builder.create<arith::SubIOp>(loc, rows, tail);
  // Activations change every call, so pack only the A panels.
  constexpr int64_t dyn = ShapedType::kDynamic;
  Value aData = builder.create<memref::AllocOp>(
      loc, MemRefType::get({dyn, blockCount, kRowsPerTile * kQuantBlock}, i8),
      ValueRange{tileRows}, builder.getI64IntegerAttr(64));
  Value aScales = builder.create<memref::AllocOp>(
      loc, MemRefType::get({dyn, blockCount, kRowsPerTile}, f32),
      ValueRange{tileRows}, builder.getI64IntegerAttr(64));
  packActivations(input, rows, blocks, aData, aScales);

  Value groups = index(channels / kLanes);
  auto columnTile = loop(zero, groups, kGroupsPerTile, {}, /*parallel=*/true);
  Value group = columnTile.getInductionVar();
  auto rowTile = loop(zero, mainRows, kRowsPerTile);
  emitTile(rowTile.getInductionVar(), group, blocks, kRowsPerTile, aData,
           aScales, weight, output, outputElement, activation, blockCount);
  builder.setInsertionPointAfter(rowTile);

  auto inputType = cast<RankedTensorType>(input.getType());
  if (inputType.isDynamicDim(0) || inputType.getDimSize(0) % kRowsPerTile) {
    auto tailRow = loop(mainRows, rows);
    emitTile(tailRow.getInductionVar(), group, blocks, 1, aData, aScales,
             weight, output, outputElement, activation, blockCount);
    builder.setInsertionPointAfter(tailRow);
  }
  builder.setInsertionPointAfter(columnTile);
  return output;
}

} // namespace
#endif

Value mlir::llmcp::emitQ8PackedMatmul(PatternRewriter &rewriter, Location loc,
                                      Value input, Value weight,
                                      RankedTensorType resultType,
                                      StringRef activation) {
#if defined(__aarch64__) && !defined(__AARCH64EB__) &&                         \
    defined(LLMCP_ENABLE_SDOT)
  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<MemRefType>(weight.getType());
  if (inputType.getRank() != 2 || weightType.getRank() != 1 ||
      !weightType.getElementType().isInteger(8) ||
      (!inputType.getElementType().isBF16() &&
       !inputType.getElementType().isF32()) ||
      (!resultType.getElementType().isBF16() &&
       !resultType.getElementType().isF32()))
    return Value();
  int64_t channels = resultType.getDimSize(1);
  int64_t reduction = inputType.getDimSize(1);
  if (ShapedType::isDynamic(channels) || ShapedType::isDynamic(reduction) ||
      channels <= 0 || channels % (kGroupsPerTile * kLanes) != 0 ||
      reduction == 0 || reduction % kQuantBlock != 0)
    return Value();

  Value rows = rewriter.create<tensor::DimOp>(loc, input, 0);
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output = rewriter.create<memref::AllocOp>(
      loc, outputType, ValueRange{rows}, rewriter.getI64IntegerAttr(128));
  return PackedQ8Matmul(rewriter, loc)
      .emit(input, weight, output, rows, reduction / kQuantBlock, channels,
            activation);
#else
  return Value();
#endif
}
