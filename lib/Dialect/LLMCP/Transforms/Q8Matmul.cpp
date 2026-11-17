//===- Q8Matmul.cpp - Packed Q8 SDOT prefill GEMM -------------------------===//
//
// Packs weight and activation panels for an SDOT-by-element microkernel.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/Transforms/MatmulLoweringInternal.h"

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
constexpr int64_t kLanes = 4;
constexpr int64_t kRowsPerTile = 4;
constexpr int64_t kGroupsPerTile = 2;
constexpr int64_t kColumnsPerPanel = 64;
constexpr int64_t kRowsPerPanel = 32;
constexpr int64_t kMinRows = 16;
constexpr int64_t kMinColumns = 64;

static_assert(kRowsPerPanel % kRowsPerTile == 0);
static_assert(kColumnsPerPanel % (kGroupsPerTile * kLanes) == 0);

class PackedQ8Matmul {
public:
  PackedQ8Matmul(PatternRewriter &builder, Location loc)
      : builder(builder), loc(loc), f32(builder.getF32Type()),
        i8(builder.getI8Type()), i32(builder.getI32Type()),
        i8x16(VectorType::get({16}, i8)), i32x4(VectorType::get({4}, i32)),
        f32x4(VectorType::get({4}, f32)), zero(index(0)),
        zeroF(
            builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(f32))) {}

  void emit(Value input, Value weight, Value output, Value rows,
            StringRef activation);

private:
  Value index(int64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc, value);
  }

  Value add(Value base, int64_t offset) {
    return builder.create<arith::AddIOp>(loc, base, index(offset));
  }

  Value min(Value lhs, Value rhs) {
    return builder.create<arith::MinUIOp>(loc, lhs, rhs);
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
  void packWeights(Value weight, Value panelStart, Value groups, int64_t blocks,
                   Value data, Value scales);
  void emitTile(Value row, Value panelStart, Value group, Value blocks,
                int64_t rowCount, Value aData, Value aScales, Value bData,
                Value bScales, Value output, Type outputElement,
                StringRef activation);

  PatternRewriter &builder;
  Location loc;
  Type f32;
  Type i8;
  Type i32;
  VectorType i8x16;
  VectorType i32x4;
  VectorType f32x4;
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

void PackedQ8Matmul::packWeights(Value weight, Value panelStart, Value groups,
                                 int64_t blockCount, Value data, Value scales) {
  auto groupLoop = loop(zero, groups);
  Value group = groupLoop.getInductionVar();
  Value globalGroup = builder.create<arith::AddIOp>(loc, panelStart, group);
  Value firstColumn =
      builder.create<arith::MulIOp>(loc, globalGroup, index(kLanes));
  Value tile =
      builder.create<arith::DivUIOp>(loc, group, index(kGroupsPerTile));
  Value groupInTile =
      builder.create<arith::RemUIOp>(loc, group, index(kGroupsPerTile));
  Value groupOffset =
      builder.create<arith::MulIOp>(loc, groupInTile, index(16));
  auto blockLoop = loop(zero, index(blockCount));
  Value block = blockLoop.getInductionVar();
  Value scaleOffset = builder.create<arith::MulIOp>(loc, block, index(2));
  Value dataOffset =
      builder.create<arith::MulIOp>(loc, block, index(kQuantBlock));
  dataOffset = add(dataOffset, blockCount * 2);
  auto i16 = builder.getI16Type();
  auto half4 = VectorType::get({kLanes}, builder.getF16Type());
  Value shift =
      builder.create<arith::ConstantOp>(loc, builder.getI16IntegerAttr(8));

  SmallVector<Value, kLanes> weightScales;
  SmallVector<Value, kLanes> columns[kQuantBlock / 16];
  for (int64_t lane = 0; lane < kLanes; ++lane) {
    Value column = add(firstColumn, lane);
    Value lowByte = builder.create<memref::LoadOp>(
        loc, weight, ValueRange{column, scaleOffset});
    Value highByte = builder.create<memref::LoadOp>(
        loc, weight, ValueRange{column, add(scaleOffset, 1)});
    Value low = builder.create<arith::ExtUIOp>(loc, i16, lowByte);
    Value high = builder.create<arith::ExtUIOp>(loc, i16, highByte);
    high = builder.create<arith::ShLIOp>(loc, high, shift);
    Value bits = builder.create<arith::OrIOp>(loc, low, high);
    weightScales.push_back(
        builder.create<arith::BitcastOp>(loc, builder.getF16Type(), bits));
    for (int64_t halfIndex = 0; halfIndex < kQuantBlock / 16; ++halfIndex) {
      Value bytes =
          load(i8x16, weight, {column, add(dataOffset, halfIndex * 16)});
      columns[halfIndex].push_back(
          builder.create<vector::BitCastOp>(loc, i32x4, bytes));
    }
  }
  Value halfScales =
      builder.create<vector::FromElementsOp>(loc, half4, weightScales);
  Value floatScales = builder.create<arith::ExtFOp>(loc, f32x4, halfScales);
  builder.create<vector::StoreOp>(loc, floatScales, scales,
                                  ValueRange{group, block, zero});

  // Transpose four columns of four-byte words. Each SDOT lane then computes
  // a different output column with the same activation word.
  for (int64_t halfIndex = 0; halfIndex < kQuantBlock / 16; ++halfIndex) {
    ArrayRef<Value> c = columns[halfIndex];
    for (int64_t pair = 0; pair < 2; ++pair) {
      int64_t first = pair * 2;
      int64_t second = first + 1;
      SmallVector<int64_t, 4> mask{first, first + 4, second, second + 4};
      Value low = builder.create<vector::ShuffleOp>(loc, c[0], c[1], mask);
      Value high = builder.create<vector::ShuffleOp>(loc, c[2], c[3], mask);
      for (int64_t word = 0; word < 2; ++word) {
        int64_t start = word * 2;
        SmallVector<int64_t, 4> words{start, start + 1, start + 4, start + 5};
        Value packed = builder.create<vector::ShuffleOp>(loc, low, high, words);
        Value bytes = builder.create<vector::BitCastOp>(loc, i8x16, packed);
        int64_t offset = (halfIndex * 4 + first + word) * kGroupsPerTile * 16;
        builder.create<vector::StoreOp>(
            loc, bytes, data,
            ValueRange{tile, block, add(groupOffset, offset)});
      }
    }
  }
  builder.setInsertionPointAfter(groupLoop);
}

void PackedQ8Matmul::emitTile(Value row, Value panelStart, Value group,
                              Value blocks, int64_t rowCount, Value aData,
                              Value aScales, Value bData, Value bScales,
                              Value output, Type outputElement,
                              StringRef activation) {
  Value rowTile = builder.create<arith::DivUIOp>(loc, row, index(kRowsPerTile));
  Value firstRow =
      rowCount == kRowsPerTile
          ? zero
          : builder.create<arith::RemUIOp>(loc, row, index(kRowsPerTile));
  Value columnTile =
      builder.create<arith::DivUIOp>(loc, group, index(kGroupsPerTile));
  Value zeroF32x4 = splat(f32x4, zeroF);
  Value zeroI32x4 =
      builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(i32x4));
  SmallVector<Value> init(rowCount * kGroupsPerTile, zeroF32x4);
  auto blockLoop = loop(zero, blocks, 1, init);
  Value block = blockLoop.getInductionVar();
  SmallVector<Value> dots(rowCount * kGroupsPerTile, zeroI32x4);
  // Unroll one K=32 quantization block. LLVM selects SDOT-by-element
  // from the splat of a four-byte activation word.
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
      for (int64_t g = 0; g < kGroupsPerTile; ++g)
        weights.push_back(
            load(i8x16, bData,
                 {columnTile, block,
                  index(((half * 4 + word) * kGroupsPerTile + g) * 16)}));
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
  for (int64_t g = 0; g < kGroupsPerTile; ++g)
    weightScales.push_back(load(f32x4, bScales, {add(group, g), block, zero}));
  Value activationScales;
  if (rowCount == kRowsPerTile)
    activationScales = load(f32x4, aScales, {rowTile, block, zero});
  SmallVector<Value> accumulators;
  for (int64_t r = 0; r < rowCount; ++r) {
    Value scale =
        activationScales
            ? builder.create<vector::ExtractOp>(loc, activationScales, r)
                  .getResult()
            : builder
                  .create<memref::LoadOp>(loc, aScales,
                                          ValueRange{rowTile, block, firstRow})
                  .getResult();
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

  Value globalGroup = builder.create<arith::AddIOp>(loc, panelStart, group);
  for (int64_t r = 0; r < rowCount; ++r) {
    for (int64_t g = 0; g < kGroupsPerTile; ++g) {
      Value result = blockLoop.getResult(r * kGroupsPerTile + g);
      Value column = builder.create<arith::MulIOp>(loc, add(globalGroup, g),
                                                   index(kLanes));
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

void PackedQ8Matmul::emit(Value input, Value weight, Value output, Value rows,
                          StringRef activation) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto outputType = cast<MemRefType>(output.getType());
  int64_t channels = outputType.getDimSize(1);
  int64_t blockCount = inputType.getDimSize(1) / kQuantBlock;
  auto outputElement = outputType.getElementType();
  Value blocks = index(blockCount);
  Value tileRows =
      builder.create<arith::CeilDivUIOp>(loc, rows, index(kRowsPerTile));
  Value tail = builder.create<arith::RemUIOp>(loc, rows, index(kRowsPerTile));
  Value mainRows = builder.create<arith::SubIOp>(loc, rows, tail);
  // A packs each row tile's bytes and FP32 scales by quantization block.
  constexpr int64_t dyn = ShapedType::kDynamic;
  Value aData = builder.create<memref::AllocOp>(
      loc, MemRefType::get({dyn, blockCount, kRowsPerTile * kQuantBlock}, i8),
      ValueRange{tileRows}, builder.getI64IntegerAttr(64));
  Value aScales = builder.create<memref::AllocOp>(
      loc, MemRefType::get({dyn, blockCount, kRowsPerTile}, f32),
      ValueRange{tileRows}, builder.getI64IntegerAttr(64));
  packActivations(input, rows, blocks, aData, aScales);

  // Pack B once per task and reuse it across row panels. Each packed K block
  // interleaves the weight vectors for the two column groups of a tile.
  constexpr int64_t groupsPerPanel = kColumnsPerPanel / kLanes;
  Value groups = index(channels / kLanes);
  auto columnPanel = loop(zero, groups, groupsPerPanel, {}, /*parallel=*/true);
  Value panelStart = columnPanel.getInductionVar();
  Value remaining = builder.create<arith::SubIOp>(loc, groups, panelStart);
  Value panelGroups = min(index(groupsPerPanel), remaining);
  Value bData = builder.create<memref::AllocOp>(
      loc,
      MemRefType::get({groupsPerPanel / kGroupsPerTile, blockCount,
                       kGroupsPerTile * kLanes * kQuantBlock},
                      i8),
      ValueRange{}, builder.getI64IntegerAttr(64));
  Value bScales = builder.create<memref::AllocOp>(
      loc, MemRefType::get({groupsPerPanel, blockCount, kLanes}, f32),
      ValueRange{}, builder.getI64IntegerAttr(64));
  packWeights(weight, panelStart, panelGroups, blockCount, bData, bScales);

  auto rowPanel = loop(zero, mainRows, kRowsPerPanel);
  Value rowStart = rowPanel.getInductionVar();
  Value rowEnd = min(add(rowStart, kRowsPerPanel), mainRows);
  auto columnTile = loop(zero, panelGroups, kGroupsPerTile);
  auto rowTile = loop(rowStart, rowEnd, kRowsPerTile);
  emitTile(rowTile.getInductionVar(), panelStart, columnTile.getInductionVar(),
           blocks, kRowsPerTile, aData, aScales, bData, bScales, output,
           outputElement, activation);
  builder.setInsertionPointAfter(rowPanel);

  if (inputType.isDynamicDim(0) || inputType.getDimSize(0) % kRowsPerTile) {
    auto tailColumn = loop(zero, panelGroups, kGroupsPerTile);
    auto tailRow = loop(mainRows, rows);
    emitTile(tailRow.getInductionVar(), panelStart,
             tailColumn.getInductionVar(), blocks, 1, aData, aScales, bData,
             bScales, output, outputElement, activation);
  }
  builder.setInsertionPointAfter(columnPanel);
}

} // namespace
#endif

Value mlir::llmcp::emitPackedQ8Matmul(
    PatternRewriter &rewriter, Location loc, Value input, Value weight,
    RankedTensorType resultType, StringRef activation,
    llvm::function_ref<Value(Value)> emitSmallMatmul) {
#if defined(__aarch64__) && !defined(__AARCH64EB__) &&                         \
    defined(LLMCP_ENABLE_SDOT)
  auto inputType = cast<RankedTensorType>(input.getType());
  auto weightType = cast<MemRefType>(weight.getType());
  if (inputType.getRank() != 2 || weightType.getRank() != 2 ||
      !weightType.getElementType().isInteger(8) ||
      (!inputType.getElementType().isBF16() &&
       !inputType.getElementType().isF32()) ||
      (!resultType.getElementType().isBF16() &&
       !resultType.getElementType().isF32()))
    return Value();
  int64_t rows = inputType.getDimSize(0);
  int64_t channels = resultType.getDimSize(1);
  int64_t reduction = inputType.getDimSize(1);
  if ((!ShapedType::isDynamic(rows) && rows < kMinRows) ||
      ShapedType::isDynamic(channels) || channels < kMinColumns ||
      channels % (kGroupsPerTile * kLanes) != 0 ||
      ShapedType::isDynamic(reduction) || reduction == 0 ||
      reduction % kQuantBlock != 0)
    return Value();
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(weightType.getStridesAndOffset(strides, offset)) ||
      strides[1] != 1)
    return Value();

  Value rowCount = rewriter.create<tensor::DimOp>(loc, input, 0);
  auto outputType = MemRefType::get({ShapedType::kDynamic, channels},
                                    resultType.getElementType());
  Value output = rewriter.create<memref::AllocOp>(
      loc, outputType, ValueRange{rowCount}, rewriter.getI64IntegerAttr(128));
  auto emitPacked = [&] {
    PackedQ8Matmul(rewriter, loc)
        .emit(input, weight, output, rowCount, activation);
  };
  if (!inputType.isDynamicDim(0)) {
    emitPacked();
    return output;
  }

  // Both branches write the same output so bufferization does not need to
  // merge ownership or clone a branch-local result.
  Value minimum = rewriter.create<arith::ConstantIndexOp>(loc, kMinRows);
  Value usePacked = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::uge, rowCount, minimum);
  auto dispatch =
      rewriter.create<scf::IfOp>(loc, usePacked, /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&dispatch.getThenRegion().front());
  emitPacked();
  rewriter.setInsertionPointToStart(&dispatch.getElseRegion().front());
  emitSmallMatmul(output);
  rewriter.setInsertionPointAfter(dispatch);
  return output;
#else
  return Value();
#endif
}
