//===- ModelBuilder.cpp - Build the llmcp model graph ---------------------===//
//
//===----------------------------------------------------------------------===//

#include "llmcp/Model/ModelBuilder.h"

#include "llmcp/Dialect/LLMCP/IR/LLMCP.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::llmcp;
using llvm::StringRef;

namespace {

constexpr int64_t kDynamic = ShapedType::kDynamic;

class GraphBuilder {
public:
  GraphBuilder(MLIRContext *context, const ModelConfig &config,
               const WeightLayout &layout)
      : context(context), config(config), layout(layout), builder(context),
        loc(UnknownLoc::get(context)) {
    for (const WeightEntry &entry : layout.getEntries())
      entries[entry.name] = &entry;
  }

  llvm::Expected<OwningOpRef<ModuleOp>> build() {
    auto module = ModuleOp::create(loc);
    OpBuilder moduleBuilder(module.getBodyRegion());
    moduleBuilder.setInsertionPointToStart(module.getBody());

    const int64_t hidden = config.text.hiddenSize;
    const int64_t vocab = config.text.vocabSize;
    const int64_t numLayers = config.getNumLayers();
    const int64_t rotary = config.getRotaryDim();
    const int64_t numLinear = config.getNumLinearLayers();
    const int64_t numFull = config.getNumFullLayers();
    const int64_t headDim = config.text.headDim;
    const int64_t numKvHeads = config.text.numKeyValueHeads;
    const int64_t numHeads = config.text.numAttentionHeads;
    const int64_t numVHeads = config.text.linearNumValueHeads;
    const int64_t headKDim = config.text.linearKeyHeadDim;
    const int64_t headVDim = config.text.linearValueHeadDim;
    const int64_t kernel = config.text.linearConvKernelDim;
    const int64_t qkvDim = config.getLinearQkvDim();
    const int64_t valueDim = config.getLinearValueDim();

    Type bf16 = moduleBuilder.getBF16Type();
    Type f32 = moduleBuilder.getF32Type();
    Type i8 = moduleBuilder.getI8Type();
    Type index = moduleBuilder.getIndexType();

    auto hiddenType = RankedTensorType::get({kDynamic, hidden}, bf16);
    auto tokensType = RankedTensorType::get({kDynamic}, index);
    auto cosType = RankedTensorType::get({kDynamic, rotary}, f32);
    auto logitsType = RankedTensorType::get({kDynamic, vocab}, f32);
    auto arenaType = MemRefType::get({kDynamic}, i8);
    auto kvType =
        MemRefType::get({numFull, 2, numKvHeads, kDynamic, headDim}, bf16);
    auto deltaType =
        MemRefType::get({numLinear, numVHeads, headKDim, headVDim}, f32);
    auto convType = MemRefType::get({numLinear, kernel - 1, qkvDim}, bf16);

    buildEmbed(moduleBuilder, tokensType, arenaType, hiddenType);
    buildLmHead(moduleBuilder, hiddenType, arenaType, logitsType);
    buildForward(moduleBuilder, hiddenType, cosType, arenaType, kvType,
                 deltaType, convType, headDim, numHeads, numKvHeads, rotary);
    buildModel(moduleBuilder, tokensType, cosType, arenaType, kvType, deltaType,
               convType, hiddenType, logitsType);
    buildAotModel(moduleBuilder, rotary, vocab, arenaType, kvType, deltaType,
                  convType, logitsType);

    module->setAttr("llmcp.model_dir",
                    moduleBuilder.getStringAttr(config.modelDir));
    module->setAttr("llmcp.weight_bytes",
                    moduleBuilder.getI64IntegerAttr(layout.getTotalBytes()));
    llvm::SmallVector<Attribute> layerKinds;
    for (LayerKind kind : config.layerTypes)
      layerKinds.push_back(
          moduleBuilder.getStringAttr(stringifyLayerKind(kind)));
    module->setAttr("llmcp.layer_types",
                    moduleBuilder.getArrayAttr(layerKinds));
    module->setAttr("llmcp.num_layers",
                    moduleBuilder.getI64IntegerAttr(numLayers));

    return OwningOpRef<ModuleOp>(module);
  }

private:
  func::FuncOp createFunc(OpBuilder &moduleBuilder, StringRef name,
                          TypeRange inputs, TypeRange results,
                          OpBuilder &bodyBuilder, bool isPublic = false) {
    auto funcType = moduleBuilder.getFunctionType(inputs, results);
    auto func = moduleBuilder.create<func::FuncOp>(loc, name, funcType);
    if (isPublic)
      func.setPublic();
    else
      func.setPrivate();
    Block *entry = func.addEntryBlock();
    bodyBuilder = OpBuilder::atBlockBegin(entry);
    return func;
  }

  Value getWeight(OpBuilder &builder, Value arena, StringRef name,
                  llvm::StringMap<Value> &cache) {
    auto cached = cache.find(name);
    if (cached != cache.end())
      return cached->second;

    auto it = entries.find(name);
    assert(it != entries.end() && "weight must exist in the layout");
    const WeightEntry &entry = *it->second;

    Type elementType;
    if (entry.dtype == "BF16")
      elementType = builder.getBF16Type();
    else if (entry.dtype == "F32")
      elementType = builder.getF32Type();
    else if (entry.dtype == "F16")
      elementType = builder.getF16Type();
    else
      llvm_unreachable("unsupported weight dtype in layout");

    auto memrefType = MemRefType::get(entry.shape, elementType);
    auto offset = builder.create<arith::ConstantIndexOp>(loc, entry.offset);
    Value view = builder.create<memref::ViewOp>(loc, memrefType, arena, offset,
                                                ValueRange{});
    cache[name] = view;
    return view;
  }

  Value rmsNorm(OpBuilder &builder, Value input, Value arena, StringRef name,
                llvm::StringMap<Value> &cache) {
    Value weight = getWeight(builder, arena, name, cache);
    return builder.create<RmsNormOp>(
        loc, input.getType(), input, weight,
        builder.getF64FloatAttr(config.text.rmsNormEps),
        builder.getBoolAttr(true));
  }

  Value linear(OpBuilder &builder, Value input, Value arena, StringRef name,
               llvm::StringMap<Value> &cache) {
    Value weight = getWeight(builder, arena, name, cache);
    auto weightType = cast<ShapedType>(weight.getType());
    auto inputType = cast<ShapedType>(input.getType());
    auto resultType = RankedTensorType::get(
        {inputType.getDimSize(0), weightType.getDimSize(0)},
        inputType.getElementType());
    return builder.create<LinearOp>(loc, resultType, input, weight);
  }

  Value add(OpBuilder &builder, Value lhs, Value rhs) {
    return builder.create<AddOp>(loc, lhs.getType(), lhs, rhs);
  }

  Value mlp(OpBuilder &builder, Value input, Value arena, StringRef prefix,
            llvm::StringMap<Value> &cache) {
    Value gate =
        linear(builder, input, arena,
               (llvm::Twine(prefix) + "mlp.gate_proj.weight").str(), cache);
    Value up =
        linear(builder, input, arena,
               (llvm::Twine(prefix) + "mlp.up_proj.weight").str(), cache);
    Value act = builder.create<SiluMulOp>(loc, gate.getType(), gate, up);
    return linear(builder, act, arena,
                  (llvm::Twine(prefix) + "mlp.down_proj.weight").str(), cache);
  }

  Value buildLinearLayer(OpBuilder &builder, Value input, Value arena,
                         Value cos, Value sin, Value deltaState,
                         Value convState, int64_t layerIndex,
                         int64_t stateIndex, llvm::StringMap<Value> &cache) {
    const std::string prefix = (llvm::Twine("model.language_model.layers.") +
                                llvm::Twine(layerIndex) + ".")
                                   .str();
    const std::string base = prefix;

    Value norm =
        rmsNorm(builder, input, arena, base + "input_layernorm.weight", cache);
    Value qkv = linear(builder, norm, arena,
                       base + "linear_attn.in_proj_qkv.weight", cache);
    Value z = linear(builder, norm, arena,
                     base + "linear_attn.in_proj_z.weight", cache);
    Value a = linear(builder, norm, arena,
                     base + "linear_attn.in_proj_a.weight", cache);
    Value b = linear(builder, norm, arena,
                     base + "linear_attn.in_proj_b.weight", cache);
    Value convWeight =
        getWeight(builder, arena, base + "linear_attn.conv1d.weight", cache);
    Value convStateArg = convState;
    Value conv = builder.create<CausalConv1dOp>(
        loc, qkv.getType(), qkv, convWeight, convStateArg,
        builder.getI64IntegerAttr(stateIndex),
        builder.getI64IntegerAttr(config.text.linearConvKernelDim),
        builder.getStringAttr("silu"));

    Value aLog = getWeight(builder, arena, base + "linear_attn.A_log", cache);
    Value dtBias =
        getWeight(builder, arena, base + "linear_attn.dt_bias", cache);
    auto valueType =
        RankedTensorType::get({cast<ShapedType>(qkv.getType()).getDimSize(0),
                               config.getLinearValueDim()},
                              builder.getBF16Type());
    Value delta = builder.create<GatedDeltaNetOp>(
        loc, valueType, conv, a, b, aLog, dtBias, deltaState,
        builder.getI64IntegerAttr(stateIndex),
        builder.getI64IntegerAttr(config.text.linearNumKeyHeads),
        builder.getI64IntegerAttr(config.text.linearNumValueHeads),
        builder.getI64IntegerAttr(config.text.linearKeyHeadDim),
        builder.getI64IntegerAttr(config.text.linearValueHeadDim),
        builder.getStringAttr("prefill"));

    Value normWeight =
        getWeight(builder, arena, base + "linear_attn.norm.weight", cache);
    Value gated = builder.create<GatedRmsNormOp>(
        loc, valueType, delta, z, normWeight,
        builder.getI64IntegerAttr(config.text.linearValueHeadDim),
        builder.getF64FloatAttr(config.text.rmsNormEps));

    Value out = linear(builder, gated, arena,
                       base + "linear_attn.out_proj.weight", cache);
    Value residual = add(builder, input, out);
    Value norm2 = rmsNorm(builder, residual, arena,
                          base + "post_attention_layernorm.weight", cache);
    Value down = mlp(builder, norm2, arena, base, cache);
    return add(builder, residual, down);
  }

  Value buildFullLayer(OpBuilder &builder, Value input, Value arena, Value cos,
                       Value sin, Value kvCache, Value position,
                       Value cacheLength, int64_t layerIndex,
                       int64_t stateIndex, llvm::StringMap<Value> &cache) {
    const std::string base = (llvm::Twine("model.language_model.layers.") +
                              llvm::Twine(layerIndex) + ".")
                                 .str();

    Value norm =
        rmsNorm(builder, input, arena, base + "input_layernorm.weight", cache);
    Value qg =
        linear(builder, norm, arena, base + "self_attn.q_proj.weight", cache);
    Value k =
        linear(builder, norm, arena, base + "self_attn.k_proj.weight", cache);
    Value v =
        linear(builder, norm, arena, base + "self_attn.v_proj.weight", cache);
    Value qNorm =
        getWeight(builder, arena, base + "self_attn.q_norm.weight", cache);
    Value kNorm =
        getWeight(builder, arena, base + "self_attn.k_norm.weight", cache);

    auto outputType = RankedTensorType::get(
        {cast<ShapedType>(qg.getType()).getDimSize(0),
         config.text.numAttentionHeads * config.text.headDim},
        builder.getBF16Type());
    Value attn = builder.create<AttentionOp>(
        loc, outputType, qg, k, v, qNorm, kNorm, cos, sin, kvCache, position,
        cacheLength, builder.getI64IntegerAttr(stateIndex),
        builder.getI64IntegerAttr(config.text.numAttentionHeads),
        builder.getI64IntegerAttr(config.text.numKeyValueHeads),
        builder.getI64IntegerAttr(config.text.headDim),
        builder.getI64IntegerAttr(config.getRotaryDim()),
        builder.getStringAttr("prefill"));

    Value out =
        linear(builder, attn, arena, base + "self_attn.o_proj.weight", cache);
    Value residual = add(builder, input, out);
    Value norm2 = rmsNorm(builder, residual, arena,
                          base + "post_attention_layernorm.weight", cache);
    Value down = mlp(builder, norm2, arena, base, cache);
    return add(builder, residual, down);
  }

  void buildEmbed(OpBuilder &moduleBuilder, Type tokensType, Type arenaType,
                  Type hiddenType) {
    OpBuilder bodyBuilder(context);
    auto func = createFunc(moduleBuilder, "llmcp_embed",
                           {tokensType, arenaType}, {hiddenType}, bodyBuilder);
    llvm::StringMap<Value> cache;
    Value arena = func.getArgument(1);
    Value table = getWeight(bodyBuilder, arena,
                            "model.language_model.embed_tokens.weight", cache);
    Value result = bodyBuilder.create<EmbeddingOp>(loc, hiddenType,
                                                   func.getArgument(0), table);
    bodyBuilder.create<func::ReturnOp>(loc, result);
  }

  void buildLmHead(OpBuilder &moduleBuilder, Type hiddenType, Type arenaType,
                   Type logitsType) {
    OpBuilder bodyBuilder(context);
    auto func = createFunc(moduleBuilder, "llmcp_lm_head",
                           {hiddenType, arenaType}, {logitsType}, bodyBuilder);
    llvm::StringMap<Value> cache;
    Value arena = func.getArgument(1);
    Value table = getWeight(bodyBuilder, arena,
                            "model.language_model.embed_tokens.weight", cache);
    Value result = bodyBuilder.create<LmHeadOp>(loc, logitsType,
                                                func.getArgument(0), table);
    bodyBuilder.create<func::ReturnOp>(loc, result);
  }

  void buildForward(OpBuilder &moduleBuilder, Type hiddenType, Type cosType,
                    Type arenaType, Type kvType, Type deltaType, Type convType,
                    int64_t headDim, int64_t numHeads, int64_t numKvHeads,
                    int64_t rotary) {
    (void)headDim;
    (void)numHeads;
    (void)numKvHeads;
    (void)rotary;
    OpBuilder bodyBuilder(context);
    Type indexType = moduleBuilder.getIndexType();
    auto func = createFunc(moduleBuilder, "llmcp_forward",
                           {hiddenType, cosType, cosType, indexType, indexType,
                            arenaType, kvType, deltaType, convType},
                           {hiddenType}, bodyBuilder);
    llvm::StringMap<Value> cache;
    Value cos = func.getArgument(1);
    Value sin = func.getArgument(2);
    Value position = func.getArgument(3);
    Value cacheLength = func.getArgument(4);
    Value arena = func.getArgument(5);
    Value kvCache = func.getArgument(6);
    Value deltaState = func.getArgument(7);
    Value convState = func.getArgument(8);
    Value hidden = func.getArgument(0);

    int64_t linearIndex = 0;
    int64_t fullIndex = 0;
    for (int64_t layer = 0; layer < config.getNumLayers(); ++layer) {
      if (config.isLinearLayer(layer)) {
        hidden =
            buildLinearLayer(bodyBuilder, hidden, arena, cos, sin, deltaState,
                             convState, layer, linearIndex, cache);
        ++linearIndex;
      } else {
        hidden = buildFullLayer(bodyBuilder, hidden, arena, cos, sin, kvCache,
                                position, cacheLength, layer, fullIndex, cache);
        ++fullIndex;
      }
    }

    Value finalNorm = getWeight(bodyBuilder, arena,
                                "model.language_model.norm.weight", cache);
    Value result = bodyBuilder.create<RmsNormOp>(
        loc, hiddenType, hidden, finalNorm,
        bodyBuilder.getF64FloatAttr(config.text.rmsNormEps),
        bodyBuilder.getBoolAttr(true));
    bodyBuilder.create<func::ReturnOp>(loc, result);
  }

  void buildModel(OpBuilder &moduleBuilder, Type tokensType, Type cosType,
                  Type arenaType, Type kvType, Type deltaType, Type convType,
                  Type hiddenType, Type logitsType) {
    OpBuilder bodyBuilder(context);
    Type indexType = moduleBuilder.getIndexType();
    auto func = createFunc(moduleBuilder, "llmcp_model",
                           {tokensType, cosType, cosType, indexType, indexType,
                            arenaType, kvType, deltaType, convType},
                           {logitsType}, bodyBuilder, /*isPublic=*/true);

    Value tokens = func.getArgument(0);
    Value cos = func.getArgument(1);
    Value sin = func.getArgument(2);
    Value position = func.getArgument(3);
    Value cacheLength = func.getArgument(4);
    Value arena = func.getArgument(5);
    Value kvCache = func.getArgument(6);
    Value deltaState = func.getArgument(7);
    Value convState = func.getArgument(8);

    auto embed = bodyBuilder.create<func::CallOp>(
        loc, "llmcp_embed", TypeRange{hiddenType}, ValueRange{tokens, arena});
    Value hidden = embed.getResult(0);
    auto forward = bodyBuilder.create<func::CallOp>(
        loc, "llmcp_forward", TypeRange{hiddenType},
        ValueRange{hidden, cos, sin, position, cacheLength, arena, kvCache,
                   deltaState, convState});
    Value forwarded = forward.getResult(0);
    auto lmHead = bodyBuilder.create<func::CallOp>(
        loc, "llmcp_lm_head", TypeRange{logitsType},
        ValueRange{forwarded, arena});
    bodyBuilder.create<func::ReturnOp>(loc, lmHead.getResult(0));
  }

  void buildAotModel(OpBuilder &moduleBuilder, int64_t rotary, int64_t vocab,
                     Type arenaType, Type kvType, Type deltaType, Type convType,
                     Type logitsType) {
    Type indexType = moduleBuilder.getIndexType();
    Type f32 = moduleBuilder.getF32Type();
    auto tokensType = MemRefType::get({1}, indexType);
    auto vectorType = MemRefType::get({1, rotary}, f32);
    auto logitsMemType = MemRefType::get({1, vocab}, f32);

    OpBuilder bodyBuilder(context);
    auto func =
        createFunc(moduleBuilder, "llmcp_model_aot",
                   {tokensType, vectorType, vectorType, indexType, indexType,
                    arenaType, kvType, deltaType, convType, logitsMemType},
                   {}, bodyBuilder, /*isPublic=*/true);
    func->setAttr("llvm.emit_c_interface", UnitAttr::get(context));

    Value tokens = func.getArgument(0);
    Value cos = func.getArgument(1);
    Value sin = func.getArgument(2);
    Value position = func.getArgument(3);
    Value cacheLength = func.getArgument(4);
    Value arena = func.getArgument(5);
    Value kvCache = func.getArgument(6);
    Value deltaState = func.getArgument(7);
    Value convState = func.getArgument(8);
    Value logitsOut = func.getArgument(9);

    Value tokensTensor = bodyBuilder.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get({1}, indexType), tokens,
        /*restrict=*/true);
    Value tokensDynamic = bodyBuilder.create<tensor::CastOp>(
        loc, RankedTensorType::get({kDynamic}, indexType), tokensTensor);
    Value cosTensor = bodyBuilder.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get({1, rotary}, f32), cos,
        /*restrict=*/true);
    Value cosDynamic = bodyBuilder.create<tensor::CastOp>(
        loc, RankedTensorType::get({kDynamic, rotary}, f32), cosTensor);
    Value sinTensor = bodyBuilder.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get({1, rotary}, f32), sin,
        /*restrict=*/true);
    Value sinDynamic = bodyBuilder.create<tensor::CastOp>(
        loc, RankedTensorType::get({kDynamic, rotary}, f32), sinTensor);

    auto embed = bodyBuilder.create<func::CallOp>(
        loc, "llmcp_embed",
        TypeRange{RankedTensorType::get({kDynamic, config.text.hiddenSize},
                                        moduleBuilder.getBF16Type())},
        ValueRange{tokensDynamic, arena});
    auto forward = bodyBuilder.create<func::CallOp>(
        loc, "llmcp_forward",
        TypeRange{RankedTensorType::get({kDynamic, config.text.hiddenSize},
                                        moduleBuilder.getBF16Type())},
        ValueRange{embed.getResult(0), cosDynamic, sinDynamic, position,
                   cacheLength, arena, kvCache, deltaState, convState});
    auto lmHead = bodyBuilder.create<func::CallOp>(
        loc, "llmcp_lm_head",
        TypeRange{RankedTensorType::get({kDynamic, vocab}, f32)},
        ValueRange{forward.getResult(0), arena});
    Value logits = lmHead.getResult(0);

    Value zero = bodyBuilder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = bodyBuilder.create<arith::ConstantIndexOp>(loc, 1);
    Value vocabValue = bodyBuilder.create<arith::ConstantIndexOp>(loc, vocab);
    auto columnLoop = bodyBuilder.create<scf::ForOp>(loc, zero, vocabValue, one,
                                                     ValueRange{});
    bodyBuilder.setInsertionPointToStart(columnLoop.getBody());
    Value column = columnLoop.getInductionVar();
    Value value = bodyBuilder.create<tensor::ExtractOp>(
        loc, logits, ValueRange{zero, column});
    bodyBuilder.create<memref::StoreOp>(loc, value, logitsOut,
                                        ValueRange{zero, column});
    bodyBuilder.setInsertionPointAfter(columnLoop);
    bodyBuilder.create<func::ReturnOp>(loc);
  }

  MLIRContext *context;
  const ModelConfig &config;
  const WeightLayout &layout;
  OpBuilder builder;
  Location loc;
  llvm::StringMap<const WeightEntry *> entries;
};

} // namespace

llvm::Expected<OwningOpRef<ModuleOp>>
ModelBuilder::build(const ModelConfig &config, const TensorMap &tensors,
                    MLIRContext *context) {
  if (!context)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MLIRContext must not be null");
  if (llvm::Error error = config.verify())
    return std::move(error);

  auto layoutOrErr = WeightLayout::build(config, tensors);
  if (!layoutOrErr)
    return layoutOrErr.takeError();

  context->getOrLoadDialect<LLMCPDialect>();
  context->getOrLoadDialect<func::FuncDialect>();
  context->getOrLoadDialect<arith::ArithDialect>();
  context->getOrLoadDialect<memref::MemRefDialect>();
  context->getOrLoadDialect<tensor::TensorDialect>();
  context->getOrLoadDialect<bufferization::BufferizationDialect>();
  context->getOrLoadDialect<scf::SCFDialect>();

  GraphBuilder graphBuilder(context, config, *layoutOrErr);
  return graphBuilder.build();
}
