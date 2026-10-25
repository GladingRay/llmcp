# llmcp：Qwen3.5-4B 的 MLIR CPU 编译器

> 技术路线与架构设计（v0.1）
>
> 配套文档：[Qwen3.5-4B 模型规格与编译契约](model-spec.md)
>
> 代码仓库：`/Users/guorui/Work/llmcp`（当前处于设计阶段，`include/`、`src/`、`tests/` 为空）

---

## 0. 一页纸摘要

### 0.1 要做的事

用 MLIR 实现一个**离线编译、CPU 执行**的 Qwen3.5-4B 推理编译器：

```text
config.json + safetensors + tokenizer
        │
        ▼
  C++ ModelBuilder ──► llmcp dialect（高层语义 IR）
        │
        ▼
  llmcp-opt（降级 / 特化 / 融合 / 向量化 / 内存规划 / 并行化）
        │
        ▼
  linalg + scf + vector + memref + llvm dialect
        │
        ▼
  LLVM IR ──► AOT .dylib / .so（默认）或 ORC JIT（开发调试）
        │
        ▼
  llmcp-runtime：tokenizer、session、采样、线程池、权重 mmap
```

编译器负责：模型语义、形状特化、算子融合、tiling/向量化、量化、内存 arena、线程调度、目标代码生成。

运行时负责：BPE tokenizer、chat template、会话与状态管理、采样、流式输出、权重加载。二者通过稳定的 C ABI 交互。

### 0.2 五个关键判断

1. **这不是普通 Transformer。** 32 层里 24 层是 Gated DeltaNet 线性注意力，只有 8 层全注意力；decode 需要 KV cache + recurrent state + conv state 三类状态。编译器必须原生支持“有状态算子”，不能假设所有层都是 attention + MLP。
2. **先 decode，后 prefill；先正确，后性能。** 先用逐 token recurrent 路径打通端到端正确性，再实现 chunked prefill。否则会在最复杂的算法上同时调试语义和性能。
3. **权重不进 IR。** 9.32GB 权重放在 mmap 的单一 arena 中，编译器只生成偏移和访问代码；否则 IR/编译时间不可控。
4. **层组循环，而不是展开 32 层。** 层模式是 `[L, L, L, F] × 8`，用一个 `scf.for` 遍历 8 个 group，group 内偏移固定；IR 规模约缩小 8 倍，且地址仍可计算。
5. **BF16 存储 + F32 累加；量化必须用 int8 点积。** 仅“反量化后 f32 FMA”只能省带宽，不能省算力；在 M2 上 decode 要真正提速，需要 `sdot/udot` 或 `i8mm`。

### 0.3 范围

| 阶段 | 范围 | 目标 |
|---|---|---|
| MVP | 文本模型、BF16、batch=1、greedy/top-p 采样、AOT 动态库 + CLI | 端到端正确，decode 可接受 |
| v1 | W8/W4 量化、prefill 优化、C API、chat | 单机可用的 CPU 推理 |
| v1.5（可选） | 视觉塔、mRoPE、MTP 投机解码 | 多模态与加速 |

非目标：训练/微调、GPU 后端、分布式、多用户连续批处理、与 HuggingFace 位级一致。

### 0.4 里程碑（详细见第 12 节）

| 里程碑 | 周期（1 人） | 交付 |
|---|---:|---|
| M0 基础设施与事实基线 | 2 周 | 可构建的 `llmcp-opt`、模型解析、tokenizer、HF 参考环境、权重布局 |
| M1 全注意力层垂直切片 | 3 周 | 单层 + MLP + embedding/lm_head 的 decode，逐层数值对齐 |
| M2 混合模型 decode 端到端 | 3 周 | 32 层循环、三类状态、逐 token prefill、贪心生成对齐 |
| M3 Prefill 与性能 | 4 周 | chunked DeltaNet、flash attention、vector/thread/arena |
| M4 量化与产品化 | 3 周 | W8/W4、AOT、C API、CLI、chat |
| M5 多模态与 MTP（可选） | 3–4 周 | 视觉塔、mRoPE、投机解码 |

> 文本路径 M0–M4 约 15 周；含可选的 M5 约 18–19 周。若再加 20% buffer，建议按 18–22 周规划。两人并行可压缩到 10–12 周，但需要清晰拆分前端/运行时与 kernel/优化两条线。

---

## 1. 目标、约束与成功标准

### 1.1 目标

1. **端到端可编译**：一条命令从模型目录生成可在本机运行的共享库和 CLI，不依赖 PyTorch 运行时。
2. **语义正确**：与 HuggingFace `transformers` 的 `qwen3_5` 参考实现在约定容差内一致；贪心生成长度 64 的 token 序列高度一致。
3. **性能可用**：在 Apple M2 / 24GB 上，BF16 decode 达到数 token/s；W4 量化后达到十几 token/s；prefill 不出现不可接受的退化。
4. **可维护**：每个 op 有 verifier 和 lit 测试；每个 pass 有输入/输出 IR 测试；端到端有 golden 回归。
5. **可扩展**：后续能加入视觉塔、MTP、其他 CPU 目标（x86 AVX2/AVX512、ARM 服务器），而不推翻前端。

### 1.2 非目标

- 不做训练、微调、反向传播。
- 不做 GPU/加速器后端（但 IR 分层应允许后续接入）。
- 不做分布式推理、连续批处理、PagedAttention 的多请求调度（v1 单 session）。
- 不承诺与 PyTorch/HF 位级一致；以数值容差和 token 一致率为准。
- 不在第一阶段实现视觉塔和 MTP。
- 不把“调用外部 BLAS”作为主路径；可以作为对标/应急选项。

### 1.3 硬约束

| 约束 | 值 | 影响 |
|---|---|---|
| 开发机 | Apple M2，8 核 CPU，24GB 统一内存，macOS 26.6.2 | 目标平台；内存是硬约束 |
| LLVM/MLIR | Homebrew `llvm@20`，版本 20.1.8 | 固定版本，避免 API 漂移 |
| 语言 | C++17 为主，Python 仅用于参考实现/脚本 | 与 LLVM 20 的构建配置一致 |
| 模型 | `/Users/guorui/Work/models/qwen3.5-4b` | 配置与权重形状以实测为准 |
| 构建 | CMake + Ninja，`find_package(MLIR)` | 不把 LLVM 源码放进仓库 |

### 1.4 成功标准

**正确性**

- 单算子：与 f32 NumPy/手写参考相比，max abs error `< 1e-4`（f32 路径）。
- 单层：与 HF bf16 参考相比，SNR `> 30dB`，max abs error `< 5e-2`。
- 全模型：固定 8 组 prompt，最后一个位置的 logits max abs error `< 0.1`，top-1 一致率 100%，top-5 重叠 `> 95%`。
- 生成：贪心 64 token 与 HF 完全一致率 `>= 95%`（允许因浮点差异导致的分歧）。

**性能（M2，目标值，M3 后按实测调整）**

| 指标 | BF16 | W8 | W4 |
|---|---:|---:|---:|
| Decode（tok/s） | ≥ 3（目标 4–7） | ≥ 8 | ≥ 12 |
| Prefill 512（tok/s） | ≥ 2 | ≥ 5 | ≥ 8 |
| 常驻 RSS（4K ctx） | < 12GB | < 7GB | < 5GB |
| 编译时间（单目标） | < 5 min | < 5 min | < 5 min |

> 说明：M2 的 CPU 峰值 FP32 约 190 GFLOPS，BF16 权重 9.32GB、带宽约 100GB/s。BF16 decode 的理论带宽上限约 10.7 tok/s，实际受指令效率、线程和内存系统影响，4–7 tok/s 是合理目标。Prefill 512 约 4.3 TFLOP，CPU 上需要数十秒，量化与 int8 点积是主要改善手段。

---

## 2. 模型事实核查：为什么不能按普通 Transformer 设计

### 2.1 层结构

32 层，`layer_types` 显式给出：

```text
layer 0..31: L L L F  L L L F  ...  L L L F
             24 个 linear_attention + 8 个 full_attention
```

每层都是 pre-norm 残差结构：

```text
h = x + mixer(rms_norm_zero(x))
y = h + mlp(rms_norm_zero(h))
```

其中 `mixer` 是 Gated DeltaNet 或全注意力。

### 2.2 两类 token mixer

**Gated DeltaNet（24 层）**

- 投影出 q/k/v、门控 z、beta 的 b、decay 的 a；
- 对 qkv 做 depthwise causal conv（kernel 4）+ SiLU；
- 16 个 k head / 32 个 v head，head dim 128；
- L2 normalize q/k，q 再乘 `128^-0.5`；
- `beta = sigmoid(b)`，`g = -exp(A_log) * softplus(a + dt_bias)`；
- recurrent state `[32, 128, 128]` f32，delta rule 更新；
- 输出经 gated RMSNorm（SiLU(z)）和 out_proj。

**全注意力（8 层）**

- `q_proj` 输出 8192 维，按 head 组织为 16 组、每组 `q(256) | gate(256)`；**不是**前 4096 维全 q、后 4096 维全 gate。拆分必须按 head 交错做；
- 16 q head / 4 kv head，head dim 256；
- q/k 先做 per-head zero-centered RMSNorm，再对前 64 维做 RoPE；
- GQA repeat 4，causal softmax（f32），PV，乘 `sigmoid(gate)`；
- o_proj。

### 2.3 对编译器架构的直接影响

1. **必须有状态抽象**：KV cache 是 `[8, 4, capacity, 256]`，DeltaNet state 是 `[24, 32, 128, 128]` f32，conv state 是 `[24, 3, 8192]`。这些不能都塞进 SSA tensor；必须以 memref 参数传入，并允许 in-place 更新。
2. **必须有 recurrent 和 chunked 两种 lowering**：decode 用单步 recurrence；prefill 用 chunked 算法。两者共享同一套投影/卷积/norm 的 lowering。
3. **不能只做“attention 算子”**：DeltaNet 是若干 matmul + 三角求解 + 逐元素 decay 的组合；需要专门的高层 op 和分解 pass。
4. **roofline 不同**：decode 受权重带宽限制，prefill 受算力限制；同一套 tile/线程策略不可能同时最优，因此 decode/prefill 分别特化函数。

详细语义、张量形状与公式见 [model-spec.md](model-spec.md)。

---

## 3. 技术路线

### 3.1 候选路线对比

| 路线 | 前端 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| A. Torch-MLIR / StableHLO → linalg | Python 导出 | 生态成熟、算子覆盖广 | 依赖 PyTorch；状态/缓存表达别扭；DeltaNet 会被拆成大量底层 op，难以做整体调度；不符合 C++ 主体 | 不采用 |
| B. IREE | StableHLO/自有 input | 端到端、CPU 后端可用、有 attention 扩展 | 依赖庞大；CPU 调度细节难控；自定义 recurrent 状态和量化不灵活；学习/调试成本高 | 仅作参考 |
| C. 自建 `llmcp` 方言 + linalg/vector + LLVM | C++ ModelBuilder | 完全可控；状态、量化、融合、调度都可按模型定制；C++ 主体；IR 可读可测 | 前期工作量大；需要自己写 verifier/lowering/schedule | **采用** |
| D. 纯手写 C++ kernel，MLIR 只做 matmul | C++ | 性能上限高、实现直接 | 不是编译器；无法复用 MLIR 优化；与目标不符 | 仅作为 microkernel 逃生通道 |

选择 C 的核心理由：

- 这个模型的关键难点不是“把 matmul 降到 LLVM”，而是**混合状态、Gated DeltaNet、partial RoPE、zero-centered norm、量化布局**。这些需要高层语义算子来承载。
- 自建方言可以把“模型语义”和“目标调度”分离：语义在 `llmcp` 层稳定，调度在 linalg/vector 层可替换。
- C++ 前端不需要 PyTorch，部署链路短；Python 只做参考实现和 golden 数据。

### 3.2 分层架构

```text
┌────────────────────────────────────────────────────────────────────┐
│ 输入层                                                              │
│  config.json   safetensors + index.json   tokenizer.json/template   │
└───────────────┬──────────────────────┬─────────────────────────────┘
                │                      │
                ▼                      ▼
┌────────────────────────────────────────────────────────────────────┐
│ llmcp-model（C++）                                                  │
│  ModelConfig   WeightLayout   SafetensorsReader   ModelBuilder      │
│  - 解析配置、校验层类型                                             │
│  - 计算权重 arena 偏移（与 llmcp-pack 共享）                         │
│  - 生成 prefill/decode 两个入口的 llmcp IR                          │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ llmcp dialect（高层语义 IR）                                        │
│  tensor 语义的激活 + memref 语义的状态/权重                          │
│  linear / rms_norm / rope / causal_conv1d / gated_delta_net /       │
│  attention / embedding / lm_head / silu_mul / add                    │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼  llmcp-opt
┌────────────────────────────────────────────────────────────────────┐
│ 语义降级：llmcp → linalg + scf + arith + math                       │
│  - 每个 op 的数学分解（attention/DeltaNet/conv/norm）                │
│  - 形状校验、状态布局、mask 生成                                    │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 调度与优化                                                          │
│  fusion → tiling → vectorization → bufferization → arena → parallel │
│  - 自定义 vectorize pass（bf16→f32 contract、GEMV、softmax）         │
│  - scf.forall → runtime thread pool                                 │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ LLVM 降级与代码生成                                                 │
│  vector/scf/memref/func → LLVM dialect → LLVM IR                    │
│  AOT: TargetMachine → .o → clang -shared → .dylib                   │
│  JIT: ExecutionEngine（测试/调优）                                  │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ llmcp-runtime（C++）                                                │
│  Tokenizer / ChatTemplate / Session / KV+State / Sampler / Threads  │
│  mmap 权重 arena；dlopen 编译产物；C ABI                            │
└────────────────────────────────────────────────────────────────────┘
```

### 3.3 编译单元与入口

编译器生成 4 个逻辑入口（可以是同一个 `.dylib` 中的符号）：

| 入口 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `llmcp_embed` | token ids | hidden `[S, 2560]` | 词嵌入 gather |
| `llmcp_forward_prefill` | hidden、positions、states | hidden、更新后的 states | 32 层，chunked DeltaNet + tiled attention |
| `llmcp_forward_decode` | hidden（1 token）、position、states | hidden、更新后的 states | 32 层，recurrent DeltaNet + KV attention |
| `llmcp_lm_head` | 最后一个 hidden | logits `[248320]` f32 | tied embedding matmul |

运行时还有一个 `llmcp_session` 层面的 `prefill`/`decode` 包装，负责 tokenizer、采样和回调。

> 为什么 embed/lm_head 独立？便于单测、复用（MTP 也复用 embedding）、以及让 prefill 只为最后一个位置算 lm_head。

### 3.4 前后端边界

| 编译器负责 | 运行时负责 |
|---|---|
| 模型图构建 | tokenize/detokenize |
| 形状特化与校验 | chat template |
| 权重偏移与访问 | mmap/预取权重 |
| 算子融合/tiling/向量化 | session/状态缓冲区 |
| 量化编码的消费 | 采样（temperature/top-k/top-p） |
| 内存 arena 规划 | 线程池生命周期 |
| 机器码生成 | 动态库加载、错误处理 |

编译器**不**做：动态 batching、请求调度、tokenizer 正则、采样循环。这样编译产物是纯计算库，易测试、易复用。

### 3.5 正确性优先策略：Decode-first

推荐顺序：

1. **单层、单 token**：先实现一个 full attention 层（layer 11）和一个 linear 层（layer 0）的 decode。
2. **逐 token prefill**：用 decode 路径循环处理 prompt。速度慢（32 层 × prompt 长度），但语义与 HF 的 recurrent 路径一致，便于定位问题。
3. **端到端贪心生成**：与 HF 对比 64 token。
4. **chunked prefill**：只替换线性注意力的 prefill 实现，用同一组 golden 验证“chunked ≈ recurrent”。
5. **性能优化**：tiling、向量化、线程、量化。

这条路径的好处：在实现最复杂的 chunked 算法之前，已经有一条**可信的正确性基线**。后续任何优化都可以与它对比。

---

## 4. 仓库与模块划分

### 4.1 目录结构

```text
llmcp/
├── CMakeLists.txt
├── cmake/
│   ├── LLMCPConfig.cmake.in
│   └── LLMCPUtils.cmake
├── include/llmcp/
│   ├── Dialect/LLMCP/IR/
│   │   ├── CMakeLists.txt            # add_mlir_dialect(LLMCPOps llmcp)
│   │   ├── LLMCPDialect.td           # dialect 定义（名字/namespace/dependent dialects）
│   │   ├── LLMCPOps.td               # op 定义；add_mlir_dialect 的入口
│   │   ├── LLMCPTypes.td             # 自定义 type；首期可为空
│   │   ├── LLMCPAttrs.td             # quant scheme 等属性/枚举
│   │   ├── LLMCPDialect.h
│   │   ├── LLMCPOps.h
│   │   ├── LLMCPTypes.h
│   │   └── LLMCP.h                   # umbrella header
│   ├── Dialect/LLMCP/Transforms/
│   │   ├── Passes.td                 # pass 声明（-gen-pass-decls）
│   │   └── Passes.h
│   ├── Conversion/LLMCPToLinalg/
│   │   └── LLMCPToLinalg.h
│   ├── Conversion/LLMCPToLLVM/
│   │   └── LLMCPToLLVM.h
│   ├── Model/
│   │   ├── ModelConfig.h
│   │   ├── Safetensors.h
│   │   ├── WeightLayout.h
│   │   └── ModelBuilder.h
│   ├── Runtime/
│   │   ├── Model.h
│   │   ├── Session.h
│   │   ├── Sampler.h
│   │   ├── Tokenizer.h
│   │   └── ThreadPool.h
│   └── Support/
│       ├── Json.h
│       └── Memory.h
├── lib/                              # 由现有空 src/ 重命名
│   ├── Dialect/LLMCP/IR/
│   │   ├── CMakeLists.txt            # add_mlir_dialect_library(MLIRLLMCPDialect ...)
│   │   ├── LLMCPDialect.cpp
│   │   ├── LLMCPOps.cpp
│   │   └── LLMCPTypes.cpp
│   ├── Dialect/LLMCP/Transforms/
│   │   ├── LowerToLinalg.cpp
│   │   ├── LowerDeltaNet.cpp
│   │   ├── LowerAttention.cpp
│   │   ├── TileAndVectorize.cpp
│   │   ├── PlanArena.cpp
│   │   ├── Parallelize.cpp
│   │   └── VerifyModel.cpp
│   ├── Conversion/LLMCPToLLVM/
│   │   └── LLVMExport.cpp
│   ├── Model/
│   │   ├── ModelConfig.cpp
│   │   ├── Safetensors.cpp
│   │   ├── WeightLayout.cpp
│   │   └── ModelBuilder.cpp
│   └── Runtime/
│       ├── Model.cpp
│       ├── Session.cpp
│       ├── Sampler.cpp
│       ├── Tokenizer.cpp
│       ├── ChatTemplate.cpp
│       └── ThreadPool.cpp
├── tools/
│   ├── llmcp-opt/llmcp-opt.cpp        # 注册方言 + pass，lit 测试入口
│   ├── llmcp-compile/llmcp-compile.cpp # 模型 → .mlir → .dylib
│   ├── llmcp-inspect/llmcp-inspect.cpp # 只读 header 的配置/布局检查
│   ├── llmcp-dump/llmcp-dump.cpp      # 生成并 dump 完整 llmcp 计算图
│   ├── llmcp-pack/llmcp-pack.cpp      # safetensors → weights.bin + manifest
│   ├── llmcp-run/llmcp-run.cpp        # CLI 推理
│   └── llmcp-tokenize/                # tokenizer 对照工具（可选）
├── python/
│   ├── reference/
│   │   ├── dump_golden.py             # HF 参考 logits / 逐层激活
│   │   ├── tiny_reference.py          # NumPy 小模型参考
│   │   └── compare.py                 # 容差与 token 一致率
│   ├── tools/
│   │   ├── gen_unicode_tables.py      # 生成 Unicode 分类表
│   │   └── inspect_model.py           # 打印配置/张量/参数量
│   ├── requirements.txt               # 参考环境（torch/transformers/numpy）
│   └── requirements-test.txt          # pytest / lit
├── tests/
│   ├── lit/Dialect/LLMCP/
│   │   ├── ops.mlir
│   │   ├── verify.mlir
│   │   └── verify-rms.mlir
│   ├── lit/Conversion/*.mlir
│   ├── unittests/
│   │   ├── ModelConfigTest.cpp
│   │   ├── ModelBuilderTest.cpp
│   │   ├── SafetensorsTest.cpp
│   │   ├── TestModel.h
│   │   └── WeightLayoutTest.cpp
│   ├── tiny/                          # 小模型端到端（CI 用）
│   ├── e2e/                           # 真实模型 golden（本地）
│   └── golden/                        # 生成物，gitignore
├── docs/
│   ├── design.md
│   └── model-spec.md
├── scripts/
│   ├── build.sh
│   ├── bench.sh
│   └── fetch_reference.sh
├── third_party/
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── utf8proc/                # submodule, v2.11.0
│   └── googletest/              # submodule, v1.18.0
└── AGENTS.md
```

> 现有仓库中的空 `src/` 在 M0 重命名为 `lib/`（与 MLIR 的目录习惯一致）；如果希望保留 `src/`，只需要替换 CMake 中的路径，不影响架构。

**TableGen 源文件与生成物**：`include/llmcp/Dialect/LLMCP/IR/` 下的 `.td` 是方言的第一等源文件，不能只写 `.h/.cpp`。`LLMCPOps.td` 是 `add_mlir_dialect(LLMCPOps llmcp)` 的目标文件，include `LLMCPDialect.td`、`LLMCPTypes.td`、`LLMCPAttrs.td`。生成物全部落在 build 目录，不提交、不手改：

```text
build/include/llmcp/Dialect/LLMCP/IR/
├── LLMCPOps.h.inc
├── LLMCPOps.cpp.inc
├── LLMCPOpsTypes.h.inc
├── LLMCPOpsTypes.cpp.inc
├── LLMCPOpsDialect.h.inc
└── LLMCPOpsDialect.cpp.inc
```

| TableGen 文件 | 作用 |
|---|---|
| `LLMCPDialect.td` | `Dialect` 定义：name、cppNamespace、dependent dialects |
| `LLMCPOps.td` | op 定义：operands/results/attributes/traits/assemblyFormat/verifier；也是 `mlir-tblgen` 的入口 |
| `LLMCPTypes.td` | `!llmcp.*` 自定义类型；首期可以为空 |
| `LLMCPAttrs.td` | `#llmcp.quant<...>` 等属性与枚举 |
| `Transforms/Passes.td` | pass 声明，配合 `-gen-pass-decls` 生成注册代码 |

> 只列 `.h/.cpp` 会漏掉“IR schema 从哪来”这一层；`.td` 才是 op/type/dialect 的单一真源，`.h/.cpp` 只是 include 生成物并补充 verifier/parser/printer 等手写逻辑。

### 4.2 构建目标

| 目标 | 类型 | 依赖 | 说明 |
|---|---|---|---|
| `MLIRLLMCPOpsIncGen` | TableGen | `LLMCP*.td` | 生成 op/type/dialect 的 `.inc` |
| `MLIRLLMCPDialect` | 库 | MLIRIR, MLIRSupport | 方言 + ops + verifier |
| `MLIRLLMCPTransforms` | 库 | Dialect, MLIRLinalgDialect/Transforms, MLIRSCF, MLIRVector | 语义降级与优化 pass |
| `MLIRLLMCPModel` | 库 | Dialect, LLVMSupport | 配置解析、权重布局、IR builder |
| `MLIRLLMCPRuntime` | 库 | LLVMSupport, Threads | tokenizer/session/sampler |
| `llmcp-opt` | 工具 | 上述 | `mlir-opt` 的专用替代 |
| `llmcp-compile` | 工具 | 上述 + MLIRExecutionEngine/TargetLLVMIR | 端到端编译 |
| `llmcp-inspect` | 工具 | Model | 只读 header，打印配置/层类型/权重布局 |
| `llmcp-dump` | 工具 | Model, Dialect | 生成完整计算图并写 `.mlir` |
| `llmcp-pack` | 工具 | Model | 权重打包/量化 |
| `llmcp-run` | 工具 | Runtime + dlopen | CLI |

### 4.3 依赖清单与用途

**Python 环境固定使用 `/Users/guorui/Work/models/venv314`**（Python 3.14.8，pip 26.2.2），不要为 `llmcp` 另建 venv。该环境已安装 `torch 2.14.1`（macOS arm64 cp314 wheel）、`transformers 5.20.2`（PyPI 版已包含 `qwen3_5`）、`numpy 2.5.4`、`safetensors 0.8.0`、`tokenizers 0.23.3`、`huggingface_hub 2.2.0`、`accelerate 1.15.0`、`jinja2 3.1.6`。参考实现只用于生成 golden，不进入编译产物。

#### 4.3.1 C++ 构建与测试

| 依赖 | 本机版本 | 用途 | 必需性 | 状态 |
|---|---|---|---|---|
| LLVM/MLIR (`llvm@20`) | 20.1.8 | 方言、TableGen、Linalg/Vector/Bufferization、LLVM IR、JIT；同时提供 `FileCheck`、`llvm-mca`、`clang-tidy` | 必需 | 已安装 |
| CMake | 4.1.1 | `find_package(MLIR)`、构建与测试 | 必需 | 已安装 |
| Ninja | 1.13.2 | 构建后端 | 必需 | 已安装 |
| Homebrew clang++ | 20.1.8 | 与 MLIR 同一工具链的 C++17 编译器 | 必需 | 已安装 |
| utf8proc | 2.11.0 | tokenizer 的 NFC 归一化与 Unicode 分类（`\p{L}`/`\p{M}`/`\p{N}`） | 必需 | 已 vendor（submodule v2.11.0） |
| GoogleTest | 1.18.0 | op verifier、layout、sampler、tokenizer 的 C++ 单测 | 测试必需 | 已 vendor（submodule v1.18.0） |
| ccache | 4.14.2 | 编译缓存；MLIR/TableGen 重编译提速 | 可选（推荐） | **未安装** |
| lit | 23.1.4 | `llmcp-opt` 的 IR 文件测试驱动 | 测试必需 | 已安装（venv314） |
| `FileCheck` / `llvm-mca` / `clang-tidy` / `llvm-cov` | 20.1.8 | IR 断言 / 机器码分析 / 静态检查 / 覆盖率 | 开发测试 | 随 LLVM 安装，但不在 PATH |

#### 4.3.2 Python 参考与测试

| 依赖 | 本机版本 | 用途 | 必需性 | 状态 |
|---|---|---|---|---|
| Python | 3.14.8 | 运行参考实现与工具脚本 | 开发必需 | venv314 |
| PyTorch | 2.14.1 | 在 CPU 上以 bf16 跑 HF 参考，导出 logits/逐层激活 | 参考必需 | 已安装 |
| Transformers | 5.20.2 | `qwen3_5` 参考实现（`Qwen3_5ForConditionalGeneration`、tokenizer） | 参考必需 | 已安装 |
| NumPy | 2.5.4 | 数值对比、tiny 模型参考、量化误差统计 | 参考必需 | 已安装 |
| safetensors | 0.8.0 | Python 侧读取/校验权重；HF 依赖 | 参考/校验 | 已安装 |
| tokenizers | 0.23.3 | HF tokenizer 的 Rust 后端；仅参考 | 参考 | 已安装 |
| huggingface_hub | 2.2.0 | 模型加载与缓存；HF 依赖 | 参考 | 已安装 |
| accelerate | 1.15.0 | `low_cpu_mem_usage` / device map，降低加载峰值内存 | 参考 | 已安装 |
| jinja2 | 3.1.6 | HF chat template 渲染；参考 | 参考 | 已安装 |
| pytest | 9.1.1 | 组织 e2e golden 对比、量化精度报告 | 可选 | 已安装（venv314） |
| Pillow / torchvision | 最新版 | M5 视觉塔的图像预处理参考 | 可选 | 未安装 |

#### 4.3.3 明确不引入

- **PCRE2 / ICU / RE2**：tokenizer 的预分词正则用 utf8proc + 手写扫描器实现，避免引入重量级正则库。
- **nlohmann/json / simdjson / RapidJSON**：用 `llvm/Support/JSON.h` 解析 config、manifest、tokenizer.json。
- **CLI11 / fmt / spdlog**：用 `llvm::cl`、`llvm::formatv`、`llvm::errs`，保持与 MLIR 工具风格一致。
- **Eigen / oneDNN / MKL / OpenBLAS**：主路径是 MLIR 生成 kernel；macOS 如需对标，用系统 `Accelerate.framework`（无需安装），不默认链接。
- **OpenMP / TBB / Taskflow**：用 C++17 `std::thread` + 自研线程池。本机虽已装 `libomp 21.1.0`，但首期不链接，避免与 LLVM 20 运行时混淆。
- **protobuf / flatbuffers / pybind11 / tokenizers-cpp**：无需要；权重格式自己解析，ABI 用 C，tokenizer 自己实现。
- **Git LFS / valgrind / perf**：golden 不入库；内存检查用 ASan，性能分析用 Instruments/`sample`/`llvm-mca`。

#### 4.3.4 安装命令

```bash
# 拉取 vendored 源码（utf8proc / googletest）
git submodule update --init --recursive

# 只缺（可选的）ccache
brew install ccache

# 让 llvm@20 的 clang-tidy/FileCheck/llvm-mca/llvm-cov 进入 PATH
export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"

# C 和 C++ 都用 Homebrew clang 20，避免 AppleClang 21 + macOS 27 SDK 的链接问题
export CC=/opt/homebrew/opt/llvm@20/bin/clang
export CXX=/opt/homebrew/opt/llvm@20/bin/clang++

# 如果装了 ccache，在 CMake 配置时启用
# -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

# Python：使用已有的 venv314，不要新建 venv
V=/Users/guorui/Work/models/venv314
"$V/bin/python" -m pip install -r python/requirements-test.txt

# 可选：M5 视觉参考
# "$V/bin/python" -m pip install pillow torchvision
```

> `lit 23` 已废弃 `lit.formats.ShTest(execute_external=True)`。测试配置统一写 `lit.formats.ShTest(execute_external=False)`，使用 lit 内置 shell；`%FileCheck`、管道和常用命令都可正常工作。已用 `%FileCheck %s --input-file=%s` 做过 smoke test。

`utf8proc` 和 `googletest` 已通过 git submodule 固定到 `third_party/utf8proc/` 与 `third_party/googletest/`，根工程只需：

```cmake
add_subdirectory(third_party)
target_link_libraries(MLIRLLMCPRuntime PRIVATE utf8proc)
target_link_libraries(llmcp_unittests PRIVATE GTest::gtest_main)
```

`third_party/CMakeLists.txt` 会关闭两个项目的 install/testing 目标，并保留 `utf8proc`、`GTest::gtest`、`GTest::gtest_main` 三个 target。子模块的版本、commit、升级流程见 [third_party/README.md](../third_party/README.md)。

如果临时要用 Homebrew 的 utf8proc 版本，它只提供 `libutf8proc.pc`（在 `/opt/homebrew/opt/utf8proc/lib/pkgconfig`），没有 CMake config；此时才需要 `find_path`/`find_library` 或设置 `PKG_CONFIG_PATH`。正式构建统一用 vendored 版本。

#### 4.3.5 许可证提示

- LLVM/MLIR、lit、clang-tools：Apache-2.0 WITH LLVM-exception。
- utf8proc：MIT；GoogleTest：BSD-3-Clause；vendored 源码保留各自的 LICENSE 文件。
- ccache：GPL-3.0-or-later，仅作为构建期可执行工具，不链接进产物。
- PyTorch：BSD-3-Clause；Transformers：Apache-2.0；NumPy：BSD-3-Clause；safetensors：Apache-2.0。

### 4.4 工具链验证（已在本机完成）

```text
$ /opt/homebrew/opt/llvm@20/bin/llvm-config --version
20.1.8

$ ls /opt/homebrew/opt/llvm@20/lib/cmake/mlir
MLIRConfig.cmake  MLIRConfigVersion.cmake  MLIRTargets.cmake ...

$ /opt/homebrew/opt/llvm@20/bin/mlir-opt --version
Homebrew LLVM version 20.1.8

$ /opt/homebrew/opt/llvm@20/bin/mlir-runner --help
OVERVIEW: MLIR CPU execution driver
```

用 `find_package(MLIR REQUIRED CONFIG)` 构建了一个最小 C++ 程序（解析并打印 `func.func`），配置、编译、运行均通过。结论：

- Homebrew 的 MLIR 头文件/静态库/CMake 包完整可用，不需要自建 LLVM。
- `MLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir`，`LLVM_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/llvm`。
- Homebrew 同时提供静态库和 `libMLIR.dylib`；链接时出现 duplicate library 警告是正常的，可忽略或统一选择一种。
- **注意**：Homebrew 的 `mlir-opt --list-passes` 中没有注册 `-linalg-vectorize`/`-linalg-tile`。`linalg::vectorize`、`tileLinalgOp` API 在头文件中可用，Transform dialect 的 `transform.structured.vectorize` 也可用（已用 bf16→f32 matmul 验证）。因此我们**不依赖 stock pass**，而是：
  - 自研 `llmcp-opt` 注册自己的 pass；或
  - 用 Transform dialect 脚本做实验，再把稳定 schedule 固化成 C++ pass。
- LLVM 20 的 JIT 工具名是 `mlir-runner`（不是旧版 `mlir-cpu-runner`）。
- `add_mlir_dialect` 依赖 `tablegen()` 函数，根 CMake 必须显式 `include(TableGen)`；否则报 `Unknown CMake command "tablegen"`。
- 本机 SDK 的 `libz.1.tbd` 在 Homebrew clang 20 下链接会报 `unknown architecture`，因此 CMake 固定使用 `/opt/homebrew/opt/zlib` 的 zlib；`ZLIB::ZLIB` 的 SDK include 也会把 `<math.h>` 排到 libc++ 前面，构建时清掉其 interface include。
- **C 和 C++ 编译器都要指向 Homebrew clang 20**。如果只指定 `CMAKE_CXX_COMPILER`，CMake 会为 C 语言选择 `/usr/bin/cc`（AppleClang 21）；在当前 macOS 27 SDK 下，AppleClang 21 链接 `libSystem.tbd` 会报 `unknown architecture arm64e.x1-macos`，导致 `project(... LANGUAGES C CXX)` 直接失败。实测以下配置可以正常构建 vendored utf8proc + googletest：

```bash
export CC=/opt/homebrew/opt/llvm@20/bin/clang
export CXX=/opt/homebrew/opt/llvm@20/bin/clang++
```

> 这是本机 SDK/CLT 组合的问题，不是 utf8proc 或 googletest 的问题；固定 Homebrew clang 20 同时也能保证和 MLIR 的 ABI 一致。

### 4.5 CMake 骨架

```cmake
cmake_minimum_required(VERSION 3.20)
project(llmcp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)  # generated *.inc
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(third_party)

# TableGen runs from the include side; the library is built under lib/.
add_subdirectory(include/llmcp/Dialect/LLMCP/IR)

add_subdirectory(lib)
add_subdirectory(tools)

option(LLMCP_BUILD_TESTS "Build llmcp tests" ON)
if(LLMCP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
```

配置时请同时指定 C/C++ 编译器：

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/llvm
```

方言定义（ODS 片段，展示 TableGen 的文件拆分与风格）：

```tablegen
// include/llmcp/Dialect/LLMCP/IR/LLMCPDialect.td
#ifndef LLMCP_DIALECT
#define LLMCP_DIALECT

include "mlir/IR/OpBase.td"

def LLMCP_Dialect : Dialect {
  let name = "llmcp";
  let cppNamespace = "::mlir::llmcp";
  let summary = "High-level dialect for hybrid LLM inference graphs";
  let dependentDialects = [
    "arith::ArithDialect",
    "math::MathDialect",
    "tensor::TensorDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
  ];
}

#endif // LLMCP_DIALECT
```

```tablegen
// include/llmcp/Dialect/LLMCP/IR/LLMCPOps.td
#ifndef LLMCP_OPS
#define LLMCP_OPS

include "llmcp/Dialect/LLMCP/IR/LLMCPDialect.td"
include "llmcp/Dialect/LLMCP/IR/LLMCPTypes.td"
include "llmcp/Dialect/LLMCP/IR/LLMCPAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class LLMCP_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLMCP_Dialect, mnemonic, traits>;

def LLMCP_LinearOp : LLMCP_Op<"linear", [Pure]> {
  let summary = "Y = X * W^T (+ bias), accumulate in f32";
  let arguments = (ins AnyTensor:$input,
                       AnyMemRef:$weight,
                       Optional<AnyTensor>:$bias,
                       DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
                       OptionalAttr<LLMCP_QuantSchemeAttr>:$quant);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

#endif // LLMCP_OPS
```

对应两个 CMake 文件（TableGen 在 `include` 侧运行，库定义在 `lib` 侧）：

```cmake
# include/llmcp/Dialect/LLMCP/IR/CMakeLists.txt
add_mlir_dialect(LLMCPOps llmcp)
```

```cmake
# lib/Dialect/LLMCP/IR/CMakeLists.txt
add_mlir_dialect_library(MLIRLLMCPDialect
  LLMCPDialect.cpp
  LLMCPOps.cpp
  LLMCPTypes.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llmcp/Dialect/LLMCP/IR

  DEPENDS
    MLIRLLMCPOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRSideEffectInterfaces
    MLIRArithDialect
    MLIRMathDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRSCFDialect)
```

> `add_mlir_dialect(LLMCPOps llmcp)` 会生成 `LLMCPOps.h.inc`、`LLMCPOps.cpp.inc`、`LLMCPOpsTypes.*.inc`、`LLMCPOpsDialect.*.inc` 以及 `MLIRLLMCPOpsIncGen` 目标。`LLMCPOps.cpp` 只 include 这些生成物并补充手写 verifier/parser/printer，不手写 op 的样板代码。

> 说明：`llmcp` 的激活是 tensor，权重/状态是 memref。降级时用 `bufferization.to_tensor` 把权重视图转成 tensor，再喂给 `linalg.matmul`；状态更新保留 memref/scf。详见 7.2。

### 4.6 代码规范

- 遵循 LLVM/MLIR 风格：`clang-format`（LLVM style）、`clang-tidy`、`-Wall -Wextra`。
- 命名：类型 `CamelCase`，函数/变量 `camelCase`，文件 `CamelCase.cpp`，pass 名 `llmcp-lower-to-linalg`。
- 源码注释用英文；文档用中文（与 `AGENTS.md` 一致）。本文档中的伪代码为讲解方便可能带中文注释，落地源码统一英文。
- 每个 op 必须有 `summary`、verifier、至少一个 lit 测试。
- 每个 pass 必须有 `--help` 描述、legality 说明、IR 前后测试。
- 提交信息用英文；功能完成可自动 commit。

---

## 5. 前端：模型导入与 IR 构建

### 5.1 数据模型

```cpp
struct LinearAttnConfig {
  int64_t hiddenSize;          // 2560
  int64_t numKeyHeads;         // 16
  int64_t numValueHeads;       // 32
  int64_t keyHeadDim;          // 128
  int64_t valueHeadDim;        // 128
  int64_t convKernel;          // 4
};

struct FullAttnConfig {
  int64_t hiddenSize;          // 2560
  int64_t numHeads;            // 16
  int64_t numKvHeads;          // 4
  int64_t headDim;             // 256
  int64_t rotaryDim;           // 64
  double  ropeTheta;           // 1e7
};

struct LayerDesc {
  enum Kind { LinearAttention, FullAttention } kind;
  LinearAttnConfig linear;
  FullAttnConfig  full;
  int64_t mlpIntermediate;     // 9216
  double  rmsEps;              // 1e-6
};

struct ModelConfig {
  std::string modelDir;
  int64_t hiddenSize, vocabSize, numLayers;
  std::vector<LayerDesc> layers;
  bool tieWordEmbeddings;
  // vision/mtp metadata, not used in v1
};
```

读取与校验：

- 用 `llvm::json::parse` 读 `config.json`，取 `text_config`。
- 校验 `layer_types.size() == num_hidden_layers`，且 `linear_attn`/`self_attn` 权重与实际 tensor 名称匹配。
- 校验 `num_heads * head_dim == q_proj.out / 2`（16×256=4096），`q_proj.out == 16 × 2 × 256 = 8192`；q/gate 在每 head 内交错，不是全局前后两半。
- 校验 `key_dim = 16×128 = 2048`，`value_dim = 32×128 = 4096`，`qkv.out = 8192`。
- 任何不一致都在构建 IR 前报错，而不是生成错误 IR。

### 5.2 safetensors 读取

- 解析 8 字节小端 header 长度 + JSON header；每个 tensor 有 `dtype`、`shape`、`data_offsets`。
- 两个分片按 `model.safetensors.index.json` 的 `weight_map` 定位；但 **打包/编译只依赖 `WeightLayout` 的确定性顺序**，不依赖文件内顺序。
- 读取采用 `mmap` + `pread` 只读；不把 9.3GB 读入堆。
- 校验：文件大小、offset 边界、dtype、shape、tensor 数量；可选 sha256（默认关闭，`--verify-weights` 开启）。
- 支持“只读 header”模式：编译器不需要数据，只需要 shapes；打包工具才读数据。

### 5.3 权重 arena 与布局

**核心决策：所有权重放在一个 mmap 的 byte arena 中，编译器使用编译期偏移。**

布局由 `WeightLayout` 统一计算，`llmcp-pack` 和 `llmcp-compile` 共享同一份代码，避免两边偏移不一致。

```text
offset 0:      header (magic, version, model hash, dtype, layout hash)
offset 64:     embedding                  [248320 × 2560] bf16
offset ...:    group 0 (layers 0..3)      3 × linear + 1 × full
offset ...:    group 1 (layers 4..7)
...
offset ...:    group 7 (layers 28..31)
offset ...:    final norm
```

每个 group 的层集合相同，因此 **group stride 固定**。group 内每层、每个张量的偏移固定。这样：

- 展开 32 层会生成 32 份权重地址常量；用 `scf.for` 遍历 group 时，只需 `group_base + const_offset`。
- 新增/删除层（换模型）时，布局代码自动适配；编译器不硬编码 offset。
- 量化时，group 内多出 scale/zero 张量，布局代码统一处理。

对齐：每个 tensor 起始地址 64B 对齐；量化 group scale 也 64B 对齐。arena 总量写入 manifest，运行时据此 mmap/分配。

**q/gate 的反交错**：全注意力 `q_proj.weight [8192, 2560]` 的行是 head-major 的 `[q_h(256), gate_h(256)]` 交错。`llmcp-pack` 应把它反交错成两个逻辑矩阵 `q_all [4096, 2560]` 和 `gate_all [4096, 2560]`，前端发射两个 `llmcp.linear`。这样每次 matmul 的输出都是连续的，避免 kernel 以 512 行为步长跨行读取。数学上与 HF 等价。若选择保留交错布局，则必须由 attention op 内部按 head 拆分，不能 flat 前后切。

manifest 示例：

```json
{
  "version": 1,
  "model": "qwen3.5-4b",
  "dtype": "bf16",
  "total_bytes": 9319737920,
  "layout_hash": "…",
  "tensors": [
    {"name": "embed", "offset": 64, "shape": [248320, 2560], "dtype": "bf16"},
    {"name": "g0.l0.qkv_w", "offset": 1271398464, "shape": [8192, 2560], "dtype": "bf16"}
  ]
}
```

> 运行时不解析每个 tensor 的 offset；编译产物里已经是常量。manifest 用于工具链、校验和调试。

### 5.4 IR 构建：完整图 dump 与层组循环

`ModelBuilder` 当前生成 4 个函数，`llmcp-dump` 可以一次性把整个模块写出来：

```mlir
// Embedding lookup.
func.func @llmcp_embed(%tokens: tensor<?xindex>, %arena: memref<?xi8>)
    -> tensor<?x2560xbf16>

// Unrolled decoder layers (current dump is the prefill graph).
func.func @llmcp_forward(
    %hidden: tensor<?x2560xbf16>,
    %cos: tensor<?x64xf32>, %sin: tensor<?x64xf32>,
    %arena: memref<?xi8>,
    %kv: memref<8x4x?x256xbf16>,
    %delta_state: memref<24x32x128x128xf32>,
    %conv_state: memref<24x3x8192xbf16>) -> tensor<?x2560xbf16>

// Tied lm_head.
func.func @llmcp_lm_head(%hidden: tensor<?x2560xbf16>, %arena: memref<?xi8>)
    -> tensor<?x248320xf32>

// End-to-end wrapper: embed -> forward -> lm_head.
func.func @llmcp_model(...) -> tensor<?x248320xf32>
```

当前 `llmcp_forward` **展开 32 层**，便于 dump、diff 和单层测试；每个权重都是 `memref.view %arena[%offset][]` 的静态视图。`layer_index` 在 delta/conv 上是线性层 slot（0..23），在 attention 上是 full 层 slot（0..7）。

> 这是“可读/可测”的 dump 形式。M3 做代码尺寸优化时，可以切换为下面的层组循环；decode 图也会复用同一套 layer builder。

层组循环（后续 codegen，伪 IR）：

```mlir
%c0 = arith.constant 0 : index
%c8 = arith.constant 8 : index
%c1 = arith.constant 1 : index
%h = scf.for %g = %c0 to %c8 step %c1 iter_args(%arg = %hidden) -> (tensor<?x2560xbf16>) {
  // group_base = g * GROUP_STRIDE (compile-time constant)
  %base = arith.muli %g, %c_group_stride : index
  // 3 linear layers at offsets 0, 1, 2; 1 full layer at offset 3
  %h1 = call @llmcp.linear_layer(%arg, %weights, %base, %delta_state, %conv_state, ...)
  %h2 = call @llmcp.linear_layer(%h1, %weights, %base, ...)
  %h3 = call @llmcp.linear_layer(%h2, %weights, %base, ...)
  %h4 = call @llmcp.full_layer(%h3, %weights, %base, %kv, ...)
  scf.yield %h4 : tensor<?x2560xbf16>
}
```

两种实现方式：

1. **内联**：把 group 的 4 层直接写在 `scf.for` 体内（IR 约 4 层规模），上文即此。
2. **函数调用**：outline 成 `@llmcp.linear_layer` / `@llmcp.full_layer`，循环体只调用。便于单独测试和 pass 调试，但函数边界会阻碍部分跨层融合（本模型跨层融合收益很小）。

建议：**先在 `scf.for` 内内联**，方便 linalg 融合；稳定后再考虑 outline。

如果 `layer_types` 不是规则的 `[L,L,L,F]`，回退到逐层展开，并给出警告；当前模型是规则的，走快路径。

### 5.5 形状特化与 bucket

- **Decode**：序列长度恒为 1，全部静态形状，最有利于向量化。
- **Prefill**：序列长度动态。两条路：
  - 动态 `tensor<?x...>` + masked vectorization；
  - 静态 bucket：128 / 512 / 2048 / 4096，padding 到 bucket，分别编译入口。
- 建议首期 decode 静态 + prefill 动态；M3 根据性能决定是否加 bucket。若加 bucket，编译产物按 `prefill_512`、`prefill_2048` 导出多个符号，运行时按 prompt 长度选择并 padding。
- `pos` 是运行时 scalar；KV cache 的 capacity 是运行时参数，IR 中为 `?`。
- `max_seq` 不做成编译期常量，避免为 262144 生成巨大代码；由运行时限制（默认 8K/32K，可配置）。

### 5.6 为什么前端用 C++ 而不是 Python 导出

- 不引入 PyTorch 运行时依赖，部署链路短。
- 线性注意力/状态/量化需要自定义 op，Torch-MLIR 导出会先把它们拆散，难以还原语义。
- C++ 可以直接复用 `WeightLayout`、`ModelConfig`、safetensors 解析，和 `llmcp-pack` 共享代码。
- Python 仍然用于参考实现和 golden，不参与编译主链路。

---

## 6. `llmcp` 方言设计

### 6.1 设计原则

1. **只承载模型语义，不承载调度**。tile size、线程数、是否量化属于 lowering 的决策，不进入高层 op 的属性（量化 scheme 除外，因为它改变数学）。
2. **激活用 tensor，状态/权重用 memref**。激活是 SSA 数据流，便于 linalg 融合；状态是长期存在的可变缓冲区，用 memref 显式表达 in-place 更新。
3. **算子粒度适中**。`gated_delta_net` 是一个复合语义 op（含 L2 norm、beta/g、recurrence），但投影/卷积/norm/out_proj 是独立 op；这样既能表达核心算法，又能复用通用优化。
4. **每个 op 可独立测试**。给定随机输入和参考实现，能单独验证。
5. **不重复造 linalg**。纯逐元素、纯 matmul 不新增 op；`llmcp` 只保留有特殊语义或需要特殊 lower 的算子。

### 6.2 类型与属性

首期不引入自定义 type，降低实现成本：

- 激活：`tensor<...xbf16>`、`tensor<...xf32>`
- 权重：`memref<...xbf16>`、`memref<...xf32>`、量化时 `memref<...xi8>`
- 状态：`memref<...xf32>`、`memref<...xbf16>`
- 属性：
  - `#llmcp.quant<w8, group=0>`、`#llmcp.quant<w4, group=64>`（`AttrDef`，参数为 scheme + groupSize）
  - `eps`、`rotary_dim`、`num_heads`、`num_kv_heads`、`chunk_size`、`mode`（`"decode"`/`"prefill"`）等

未来如果状态语义变复杂，再引入 `!llmcp.kv_cache` / `!llmcp.recurrent_state`，并提供 materialize pass；首期用 memref + verifier 足够。

### 6.3 算子清单

| Op | 语义 | 关键属性 | 降级目标 |
|---|---|---|---|
| `llmcp.linear` | `Y = X W^T (+b)`，f32 累加 | `transpose_b`, `quant` | `linalg.matmul` + cast；量化时 `linalg.generic`/custom |
| `llmcp.rms_norm` | zero-centered RMSNorm | `eps`, `zero_centered` | `linalg.reduce` + `linalg.generic` |
| `llmcp.gated_rms_norm` | 128 维 gated RMSNorm | `eps`, `activation="silu"` | 同上 + SiLU |
| `llmcp.rope` | partial RoPE / mRoPE | `rotary_dim`, `interleaved` | `linalg.generic` + cos/sin 表 |
| `llmcp.causal_conv1d` | depthwise causal conv + state | `kernel=4`, `activation="silu"` | `scf.for` + `linalg.generic`；decode 特化 |
| `llmcp.gated_delta_net` | DeltaNet recurrence/chunk | `mode`, `chunk_size`, head 配置 | decode: linalg reductions；prefill: matmul + solve |
| `llmcp.attention` | GQA causal attention + KV cache + gate | `mode`, `causal`, `scale`, `kv_slot` | prefill: tiled online softmax；decode: cache scan |
| `llmcp.embedding` | 词嵌入 gather | — | `tensor.extract` / `linalg.generic` |
| `llmcp.lm_head` | tied embedding matmul（只算最后位置） | `chunk_size` | matmul + 可选 argmax |
| `llmcp.silu_mul` | `silu(gate) * up` | — | `linalg.generic` |
| `llmcp.add` | 残差加 | — | `linalg.add` |
| `llmcp.qmatmul`（后续） | 量化 matmul | `scheme` | custom vector kernel |

> 拆分/拼接（q/gate、q/k/v）用 `tensor.extract_slice`、`tensor.expand_shape`、`tensor.collapse_shape`，不新增 op。

### 6.4 关键算子语义与校验

**`llmcp.linear`**

- 输入 rank 2（`[M, K]`）；权重 `[N, K]`；输出 `[M, N]`。
- `M` 可以是 1（decode）或动态（prefill）；`K/N` 静态。
- 输出 bf16；内部 f32 累加。
- verifier：`K` 匹配、`N` 匹配、dtype 匹配、量化 scheme 与权重类型匹配。

**`llmcp.rms_norm`**

- 输入 `[..., D]`；weight `[D]` f32 或 bf16。
- `zero_centered=true` 时用 `(1 + w)`。
- 内部 f32；输出与输入同 dtype。
- verifier：`D` 匹配、`eps > 0`。

**`llmcp.gated_rms_norm`**

- 输入 `[M, 128]`，gate `[M, 128]`，weight `[128]`。
- 语义见 model-spec 6.4。
- verifier：最后一维 128、gate 同形状。

**`llmcp.rope`**

- q `[S, Hq, D]`，k `[S, Hk, D]`，cos/sin `[S, rotary_dim]`。
- `rotary_dim=64`、`D=256`；只作用于前 64 维。
- verifier：`rotary_dim` 偶数、`<= D`，cos/sin 形状匹配。

**`llmcp.causal_conv1d`**

- 输入 `[S, C]`，weight `[4, C]`（转置后），state `[3, C]`。
- `S=1` 时只读 state + 当前输入，输出 `[1, C]`，并写回 state；`S>1` 时做因果卷积并更新 state 为最后 3 个输入。
- verifier：`C=8192`、kernel 4、state 维度匹配。

**`llmcp.gated_delta_net`**

- q `[S, 16, 128]`，k 同，v `[S, 32, 128]`，a/b `[S, 32]`，A_log `[32]` f32，dt_bias `[32]`，state `[32, 128, 128]` f32。
- `mode="decode"`：`S` 必须为 1；走 recurrent。
- `mode="prefill"`：走 chunked；`chunk_size` 默认 64。
- 内部完成 L2 norm、q scale、beta、g、head 映射（`k_head = v_head/2`）、state 更新。
- verifier：head 数/维度、state 形状、mode 与 S 的一致性。

**`llmcp.attention`**

- q `[S, 16, 256]`，k/v `[S, 4, 256]`，gate 可选 `[S, 16, 256]`，cache `[8, 4, capacity, 256]` bf16。
- `mode="decode"`：S=1，写 cache 的 `pos`，读 `[0, pos]`。
- `mode="prefill"`：写 `[pos, pos+S)`，causal mask。
- 输出 `[S, 16, 256]` bf16。
- verifier：GQA 整除、head_dim 256、cache slot 合法、capacity 足够。

**`llmcp.embedding` / `llmcp.lm_head`**

- embedding：tokens `[S]`、table `[248320, 2560]` → `[S, 2560]`。
- lm_head：hidden `[1, 2560]`、table `[248320, 2560]` → logits `[1, 248320]` f32；内部分块，避免 1MB 以上临时 buffer（可选）。
- 首期 lm_head 输出 f32；采样在 runtime。

### 6.5 示例 IR

Decode 的线性注意力层（省略部分 SSA 名称）：

```mlir
// Weight slices are views into the arena; offsets are compile-time constants.
%w_in   = memref.subview %arena[...] : memref<?xi8> to memref<2560xbf16, strided<[1], offset: ?>>
%w_qkv  = memref.subview %arena[...] : memref<?xi8> to memref<8192x2560xbf16, strided<[2560, 1], offset: ?>>
%w_z    = memref.subview %arena[...] : memref<?xi8> to memref<4096x2560xbf16, strided<[2560, 1], offset: ?>>
// ...

%h = llmcp.rms_norm %x, %w_in {eps = 1.0e-6, zero_centered = true}
    : (tensor<1x2560xbf16>, memref<2560xbf16>) -> tensor<1x2560xbf16>

%qkv = llmcp.linear %h, %w_qkv
    : (tensor<1x2560xbf16>, memref<8192x2560xbf16>) -> tensor<1x8192xbf16>
%z   = llmcp.linear %h, %w_z
    : (tensor<1x2560xbf16>, memref<4096x2560xbf16>) -> tensor<1x4096xbf16>
%b   = llmcp.linear %h, %w_b
    : (tensor<1x2560xbf16>, memref<32x2560xbf16>) -> tensor<1x32xbf16>
%a   = llmcp.linear %h, %w_a
    : (tensor<1x2560xbf16>, memref<32x2560xbf16>) -> tensor<1x32xbf16>

%conv = llmcp.causal_conv1d %qkv, %w_conv, %conv_state
    {kernel = 4 : i32, activation = "silu"}
    : (tensor<1x8192xbf16>, memref<4x8192xbf16>, memref<3x8192xbf16>)
      -> tensor<1x8192xbf16>

%q, %k, %v = "llmcp.split_qkv"(%conv) : (tensor<1x8192xbf16>)
    -> (tensor<1x16x128xbf16>, tensor<1x16x128xbf16>, tensor<1x32x128xbf16>)

%o = llmcp.gated_delta_net %q, %k, %v, %a, %b, %w_alog, %w_dtbias, %delta_state
    {mode = "decode", num_k_heads = 16 : i32, num_v_heads = 32 : i32}
    : (tensor<1x16x128xbf16>, tensor<1x16x128xbf16>, tensor<1x32x128xbf16>,
       tensor<1x32xbf16>, tensor<1x32xbf16>, memref<32xf32>, memref<32xbf16>,
       memref<32x128x128xf32>) -> tensor<1x32x128xbf16>

%g = llmcp.gated_rms_norm %o, %z, %w_norm {eps = 1.0e-6}
    : (tensor<1x32x128xbf16>, tensor<1x32x128xbf16>, memref<128xf32>)
      -> tensor<1x32x128xbf16>

%y = llmcp.linear %g_flat, %w_out
    : (tensor<1x4096xbf16>, memref<2560x4096xbf16>) -> tensor<1x2560xbf16>
%res = llmcp.add %x, %y : (tensor<1x2560xbf16>, tensor<1x2560xbf16>) -> tensor<1x2560xbf16>
```

> 上面的 `llmcp.split_qkv` 在正式设计中不需要新 op，用 `tensor.extract_slice` 即可；这里只是示意。权重视图的实际写法会在实现时用 `memref.reinterpret_cast`/`memref.subview` + `bufferization.to_tensor` 完成。

### 6.6 为什么不用 StableHLO/Torch-MLIR

- **状态表达**：StableHLO 是纯函数式、无状态语义；KV cache/recurrent state 需要额外的 buffer 读写约定，容易退化成“大量 gather/scatter”。
- **DeltaNet**：没有对应的高层 op；导出后会变成几十个 matmul/elementwise，编译器很难识别“这是一个 delta rule”，也就无法做 chunked 特化。
- **量化**：Torch-MLIR 的量化类型/流程与自定义 group-wise W4 不完全匹配，改造成本高。
- **调试**：自建方言的 IR 更接近模型结构，出问题时可以直接对照论文/HF 代码。

IREE 的 `iree_linalg_ext.attention` 等扩展值得借鉴，但不作为依赖。

---

## 7. 降级与优化流水线

### 7.1 阶段划分

```text
[1] llmcp-lower-to-linalg      # 语义分解，消除 llmcp op
[2] canonicalize / cse
[3] llmcp-fuse-elementwise     # silu_mul、residual、norm 邻接融合
[4] llmcp-tile-and-vectorize   # 自定义 tiling + vectorization
[5] one-shot-bufferize          # tensor → memref
[6] llmcp-plan-arena            # alloc → workspace view
[7] llmcp-parallelize           # scf.forall → runtime thread pool
[8] llmcp-lower-to-llvm         # 标准 dialect → LLVM dialect
[9] translate + emit            # LLVM IR → .o/.dylib
```

每个阶段都可以单独运行，便于 `lit` 测试和 bisect。

### 7.2 逐算子降级

#### `llmcp.linear`

首期（未量化）：

```mlir
// 1) weight view -> tensor (no copy)
%w_t = bufferization.to_tensor %w restrict : memref<8192x2560xbf16> -> tensor<8192x2560xbf16>
// 2) matmul with f32 accumulator
%init = tensor.empty(%m) : tensor<?x8192xf32>
%mm = linalg.matmul ins(%x, %w_t : tensor<?x2560xbf16>, tensor<8192x2560xbf16>)
                   outs(%init : tensor<?x8192xf32>) -> tensor<?x8192xf32>
// 3) cast to bf16
%y = linalg.generic { ... arith.truncf ... } ins(%mm) outs(%empty_bf16) ...
```

- MLIR 20 的 `linalg.matmul` 允许 bf16 输入 + f32 输出（已用 `mlir-opt` 验证 verifier 通过），但**不允许 tensor 与 memref 混用**（会报 “expected to have pure tensor or buffer semantics”）。因此权重必须先用 `bufferization.to_tensor` 转成 tensor 视图。
- `bufferization.to_tensor` 必须是零拷贝别名。bufferization 之后要检查它已被消除；如果被物化成 copy，整段 kernel 改走 buffer 语义（memref 版 linalg）或显式 `memref.view`，**禁止对 9.3GB 权重产生拷贝**。
- `linalg::vectorize` 对 mixed matmul 会生成 `arith.extf` + `arith.mulf` + `vector.multi_reduction`（已实测），**不会**直接生成 mixed `vector.contract`。因此自定义 vectorize pass 需要把 `extf` 后的 f32 tile 组成 `vector.contract`，或接受 multi-reduction 并依靠 LLVM 融合 FMA。
- 量化时，`llmcp.linear` 直接降级为 `llmcp.qmatmul` 的分解（dequant + dot），见 10 节。

#### `llmcp.rms_norm`

```text
%sq   = linalg.generic { sum = sum(x*x) } ... -> tensor<?xf32>   // reduction
%mean = arith.divf %sq, 2560
%var  = arith.addf %mean, eps
%rstd = math.rsqrt %var
%y    = linalg.generic { y = x_f32 * rstd * (1 + w) } -> bf16
```

- 用 `linalg.reduce` 或 `linalg.generic` + `iterator_types=["reduction"]`。
- 归约顺序固定（单线程时），便于 golden 对比；多线程只并行不同行，行内归约顺序不变。
- f32 计算，输出 bf16。

#### `llmcp.gated_rms_norm`

与上面相同，额外读 gate，计算 `silu(gate) = gate * sigmoid(gate)`，最后乘。gate 在 f32 中算。

#### `llmcp.rope`

```text
%cos = ... // [S, 64] f32, 由 runtime 表或 IR 内 sin/cos 生成
%q_rot = tensor.extract_slice %q[..., 0:64]
%q_pass = tensor.extract_slice %q[..., 64:256]
%q_rot' = q_rot * cos + rotate_half(q_rot) * sin
%q' = tensor.insert_slice %q_rot' into %q_pass
```

- `rotate_half` 在 64 维内做前后 32 维交换取负。
- 纯文本时 positions 是 `[S]`；mRoPE 时 positions 是 `[3, S]`（T/H/W），cos/sin 表按交错规则生成。
- cos/sin 表由运行时传入；decode 只生成一行，prefill 生成 `[S, 64]`。不在 IR 中为 262144 个位置预计算。

#### `llmcp.causal_conv1d`

Decode（S=1）：

```text
out[c] = silu(w[0,c]*state[0,c] + w[1,c]*state[1,c] + w[2,c]*state[2,c] + w[3,c]*x[c])
state = shift(state, x)
```

- 对 c 做向量化（8192 个通道）。
- state 布局 `[3, C]`，4 个 tap 都是连续向量，适合 NEON。
- 权重打包成 `[4, C]`（转置），避免 stride。

Prefill（S>1）：

```text
for t in 0..S:
  out[t] = silu(sum_{k=0..3} w[k] * x[t-k])   # t-k < 0 时用 state 或 0
state = x[S-3:S]
```

- 首期可以用 `scf.for` + `linalg.generic`；向量化沿 C 维。
- 边界处理：左侧补 3 个零/旧 state；用 `tensor.pad` 或显式 `scf.if`。

#### `llmcp.gated_delta_net`

Decode（S=1）分解为两个阶段（减少 state 读写）：

```text
# 阶段 1：读旧 S，算 kv_mem
%kv_mem = linalg.generic reduction over K:
            sum_k S[h,k,v] * k[h,k]  -> [32, 128]
# 阶段 2：更新 S，同时算 o（读旧 S 一次，写新 S）
%S_new[h,k,v] = S[h,k,v] * exp(g[h]) + k[h,k] * delta[h,v]
%o[h,v]       = sum_k S_new[h,k,v] * q[h,k]
```

- 阶段 1/2 都按 `(h, v-block)` 并行；每个线程独占一个 head/block，无跨线程归约。
- `delta = (v - kv_mem) * beta`；`g`、`beta` 在 op 内部计算（或提前用 f32 算好作为输入）。
- q/k 的 L2 norm 和 q scale 在 op 内部完成；q/k repeat 用 `k_head = v_head/2` 的索引，不物化 repeat。

Prefill（chunked）：

```text
for chunk in 0..ceil(S/64):
  # 1) cum_decay / pair_decay
  # 2) ut_system、intra_attn
  # 3) solve_lower_unit（前向代入）
  # 4) v_new、o、S 更新
```

- 1) 用 `linalg.generic` + `math.exp`；pair_decay 的上三角置 0。
- 2) 用 `linalg.batch_matmul`（32 个 head）+ elementwise。
- 3) 用 `scf.for` 对 64 行前向代入；每行是一个 64×128 的小 matvec。
- 4) 用 `linalg.batch_matmul` + elementwise。
- chunk 64 时，临时 buffer 约 32×64×64×4B = 512KB，可放 arena。

> 实现顺序：先 decode recurrent；prefill 先循环调用 recurrent；最后替换为 chunked。chunked 的正确性用“与 recurrent 对比”来保证，而不是只对 HF。

#### `llmcp.attention`

Prefill（tiled flash）：

```text
for q_block in 0..S step BQ:
  for kv_block in 0..(q_block_end) step BK:
    scores = q_block @ k_block^T * scale + mask
    online_softmax_update(acc, m, l, scores)
    acc += p @ v_block
  out[q_block] = acc / l
```

- 每个 head 独立；16 个 head 可并行。
- `BQ=32/64`、`BK=64/128`；scores 用 f32。
- causal mask：只处理 `kv_pos <= q_pos` 的块；块内上三角置 `-inf`。
- gate 在输出后乘 `sigmoid(gate)`；可以在最后融合，避免单独 pass。
- KV cache 写入：在 attention 前把 k/v 写到 `cache[layer, kv_head, pos, :]`；或 attention op 内部完成。建议 op 内部完成，减少接口。

Decode：

```text
for h in 0..16:
  scores[t] = dot(q[h], cache_k[h/4, t]) * scale   # t = 0..pos
  m = max(scores); p = exp(scores - m); l = sum(p)
  out[h] = sum_t p[t] * cache_v[h/4, t] / l
out *= sigmoid(gate)
```

- 16 个 head 并行；每个 head 扫 KV cache。
- 长上下文时，KV 读取是主要成本；KV cache 连续布局很重要。
- 可以用两遍（先 max/sum，再加权求和）或 online 一遍；decode 时两遍更简单，KV 读两次；后续可 fused。

#### `llmcp.embedding` / `llmcp.lm_head`

- embedding：`tensor.extract`/gather；prefill 一次取 S 行。
- lm_head：只对最后一个 hidden 做 `[1, 2560] × [2560, 248320]`；输出 f32。可以分 vocab block 计算，减少峰值 buffer；首期直接输出 1MB logits 也可接受。

### 7.3 Tiling、向量化与融合

**硬件参数（Apple M2）**

- NEON 128-bit：f32 4 lanes，bf16/f16 8 lanes。
- 有 `dotprod`、`i8mm`、`fullfp16`、`bf16` 目标特性（用 `clang -mcpu=apple-m2 -dM` 验证）；无 SVE/SME。
- 4 个性能核 + 4 个能效核；L2 共享，内存带宽约 100GB/s。

**Matmul/GEMV tiling**

| 场景 | BM | BN | BK | 说明 |
|---|---:|---:|---:|---|
| Decode GEMV（M=1） | 1 | 64 | 256 | 每个线程处理若干 N 行；K 连续 |
| Prefill GEMM | 8/16 | 32/64 | 256 | 寄存器分块；M 方向并行 |
| lm_head (M=1) | 1 | 128 | 256 | 词表大，N 方向分块 + argmax 融合（后续） |

- K 维 256 对齐：2560/256=10、9216/256=36、4096/256=16、8192/256=32。
- bf16 权重加载 `vector<8xbf16>`，`arith.extf` 到 `vector<8xf32>`，再 FMA；避免 bf16 累加。
- 对 M=1，避免 3-D vector（MLIR 的默认 vectorizer 会生成 `vector<1xNxK>`），改用 2-D reduction 的专用 schedule。
- 对 prefill，先做 `linalg::vectorize`，再把 `multi_reduction(mulf(extf, extf))` 重写成 `vector.contract` f32；这样 codegen 更好。

**融合**

- `silu(gate_proj) * up_proj`：一个 `linalg.generic`，避免中间 9216 向量。
- 残差加 + 下一个 RMSNorm：可以在 norm 内先读两个输入相加，省一次写回；收益小，按需。
- q/k norm + RoPE：per-head 小算子，融合成 head 内循环。
- attention 输出 gate：在 attention 的最后一次写回时乘 `sigmoid(gate)`。
- `lm_head` + argmax/top-k：greedy 可以流式求 argmax，避免写 248320 个 logits；但 top-p 需要完整 logits，首期不做。

**不融合**

- 不要把 4 个 projection 融成一个巨大 matmul：它们的 N 不同、语义不同；分开更利于 tiling 和调试。权重读取是主要成本，融合收益有限。
- 不要把整个 DeltaNet 融成一个 giant generic：会失去 matmul 的 tiling 机会。

### 7.4 Bufferization 与 arena

```text
one-shot-bufferize { bufferize-function-boundaries = true }
buffer-deallocation
hoist-static-allocs
llmcp-plan-arena
```

`llmcp-plan-arena`：

1. 收集所有 `memref.alloc`（静态形状、非 escaping）。
2. 计算 live range（基于 SSA 使用链；loop 内分配视为跨迭代复用）。
3. 线性扫描分配 offset，64B 对齐；生成 `memref.view %workspace[%offset][]`。
4. 记录 workspace 峰值大小，写入模块属性 `llmcp.workspace_bytes`。
5. 删除 alloc/dealloc；如果存在逃逸 alloc，报错（不允许在 hot path 上）。

运行时按 `workspace_bytes` 分配一次；prefill/decode 复用。对于动态 seq，workspace 大小按 bucket/最大 seq 计算；也可以让 runtime 传更大的 workspace。

> 目标：稳态 forward 中 **0 次 malloc/free**。测试用 `--llmcp-verify-no-alloc` 检查。

### 7.5 并行化

IR 层用 `scf.forall` 表达 tile 级并行：

```mlir
scf.forall (%n) in (16) {
  // compute output rows for this n-tile
}
```

`llmcp-parallelize`：

- 把 `scf.forall` 的 body outline 成 `func.func`，替换为 `llmcp_rt_parallel_for(begin, end, grain, ctx, fn)`。
- 运行时线程池：持久线程、静态分区、每次调用一个 barrier；grain size 可配（例如 32 行/任务）。
- 或者先不并行，M3 再加；但接口和 pass 位置要预留。

并行粒度：

| 算子 | 并行维度 | 备注 |
|---|---|---|
| linear（decode） | N（输出行） | 每线程独立权重行，无归约 |
| linear（prefill） | M/N tile | 避免同一输出 tile 跨线程 |
| attention（decode） | head（16） | 16 个任务，8 核可接受 |
| attention（prefill） | q block × head | 任务数足够 |
| DeltaNet（decode） | v head（32） | 每 head 独立 state |
| DeltaNet（prefill） | head × chunk？ | chunk 间有依赖，只能 head 内并行；或按 head 并行 |
| RMSNorm | 行 | 行内归约不跨线程 |
| conv1d | 通道 block | 每通道独立 |

> 嵌套并行要禁止：一个 kernel 内只允许一层 `scf.forall`；外层调用不再嵌套线程池。

线程数：默认 `min(8, hw_concurrency)`；decode 可能 4 个性能核更好，prefill 用 8 个。提供 `--threads` 和运行时选项，benchmark 决定默认值。

### 7.6 LLVM 降级与代码生成

标准 pass 序列（MLIR 20 名称）：

```text
--lower-affine
--convert-scf-to-cf
--convert-vector-to-llvm
--convert-math-to-llvm          # 或 --convert-math-to-libm（首期更稳）
--convert-arith-to-llvm
--convert-index-to-llvm
--finalize-memref-to-llvm
--convert-func-to-llvm='use-bare-ptr-memref-call-conv=1'
--reconcile-unrealized-casts
```

然后：

```text
mlir-translate --mlir-to-llvmir model.mlir -o model.ll
clang++ -O3 -mcpu=apple-m2 -shared -o libqwen35.dylib model.ll llmcp_runtime.o
```

或 in-process：`translateModuleToLLVMIR` + `TargetMachine::addPassesToEmitFile` 直接生成 `.o`。

- `use-bare-ptr-memref-call-conv=1` 把 memref 参数降成裸指针，C ABI 稳定；动态 shape 需要额外传 sizes（`memref<?x...>` 的 descriptor 不适合直接暴露）。
- 目标 CPU：`-mcpu=apple-m2`；跨平台时用 `--target=generic-arm64`/`x86-64-v3`。
- 浮点：不要在全局加 `-ffast-math`；在需要融合的 `arith.mulf/addf` 上设 `fastmath<contract>`，或设置 LLVM `AllowFPOpFusion=Fast`。这会影响数值，需要 golden 验证。
- `math.exp`：首期用 libm（`--convert-math-to-libm`），正确但慢；M3 用多项式近似（自定义 `llmcp.fast_exp`）优化 softmax/SiLU。
- `memref` 动态 stride 会增加地址计算；decode 全静态，prefill 用 static bucket 可减少开销。

### 7.7 精度策略

- BF16 存储，F32 累加；state f32；softmax/norm/rope f32。
- 不依赖 mixed-precision `vector.contract`：MLIR 20 的 `convert-vector-to-llvm` 不降级 bf16×bf16→f32 的 contract（已实测）。要么先 `extf` 到 f32 再做 f32 contract，要么接受 `extf + mul + multi_reduction`。
- 量化后：激活仍 bf16/f32，权重 int8/int4；点积用 int32 累加，最后乘 scale。
- 数值验证按 [model-spec.md](model-spec.md) 第 12 节的分层容差。

### 7.8 Pass pipeline 示例

```bash
llmcp-compile \
  --model /Users/guorui/Work/models/qwen3.5-4b \
  --weights /Users/guorui/Work/models/qwen3.5-4b/weights.bin \
  --emit=shared --out=build/libqwen35.dylib \
  --target=apple-m2 --precision=bf16 \
  --prefill=dynamic --threads=8 \
  --mlir-print-ir-after=llmcp-tile-and-vectorize \
  --verify-each
```

内部 pass pipeline（C++ `PassManager`）：

```cpp
pm.addPass(llmcp::createLowerToLinalgPass({/*mode=*/Mode::Both}));
pm.addPass(createCanonicalizerPass());
pm.addPass(createCSEPass());
pm.addPass(llmcp::createFuseElementwisePass());
pm.addPass(llmcp::createTileAndVectorizePass({/*target=*/Target::AppleM2}));
pm.addPass(createOneShotBufferizePass());
pm.addPass(llmcp::createPlanArenaPass());
pm.addPass(llmcp::createParallelizePass());
pm.addPass(llmcp::createLowerToLLVMPass());
```

### 7.9 调试与可观测性

- `--mlir-print-ir-after-all` / `--mlir-print-ir-after=<pass>`；每个阶段可 dump。
- `llmcp-opt --llmcp-lower-to-linalg --debug-only=llmcp`（`LLVM_DEBUG`）。
- 生成 `.ll` 和 `.s`（`llc -filetype=asm`），用 `llvm-mca` 分析热点 kernel。
- 运行时 `LLMCP_DUMP_TENSOR=layer0.qkv:/tmp/x.bin` 导出中间张量，与 golden 对比。
- 每个 kernel 打点（可选）：`llmcp_rt_profile_begin/end`，输出每层耗时。
- 崩溃定位：AOT 保留 DWARF（`-g`）；JIT 用 `--debug`。

---

## 8. 关键内核设计

### 8.1 Decode GEMV

Decode 的绝大多数时间在 GEMV（M=1）。策略：

- 一个线程处理一段输出行（例如 32/64 行），每行 `K` 个权重连续读取。
- 每行：`acc = sum_k extf(w[k]) * extf(x[k])`，f32 FMA；K 为 2560/9216/4096/8192，可按 4/8 展开。
- 权重流式读取，硬件预取；每个线程的读取地址连续，避免多线程互相踩缓存行。
- 对于 bf16，加载 8 个 bf16 = 16B，扩成 8 个 f32（32B），FMA 8 次；计算/加载比 1:1，接近带宽上限。
- lm_head 的 N=248320，按行分块给线程；只算 logits，不做 softmax。
- 线程数：decode 可能 4 个 P 核最优（E 核拖慢 barrier）；实测决定。

### 8.2 DeltaNet decode

每个 v head 独立；state 2MiB/layer，24 层共 48MiB。每 token 读+写约 50MiB，占 decode 带宽的 0.5%（相对 9.3GB 权重），可接受。

优化点：

- 阶段 1（kv_mem）与阶段 2（update + o）分开，避免重复读 state 三次。
- 阶段 1 只读 state 和 k；阶段 2 读 state、写 state、读 q。
- 每个 head 的 K=128、V=128，向量化 V 维；K 维归约。
- `exp(g)` 每 head 一个标量，提前算好 32 个 f32。
- `beta`、`g` 的投影（32 维）可以和其他 projection 融合，或单独小 GEMV。

### 8.3 DeltaNet prefill（chunked）

- chunk 64；每 chunk 内 64×64 的三角求解是串行瓶颈，但只有 64 行，每行 64 维；可以用 4 个 warp 式并行（按列分块）。
- 32 个 v head 可并行；k head 复用。
- `ut_system`、`intra_attn` 是 64×64，矩阵小，适合寄存器/共享内存；不要调用通用 GEMM。
- 跨 chunk 的 state 更新是串行链；prefill 的并行度主要在 head 维（32）。
- 长 prompt 可以分段（例如每 2048 token）处理，段间保留 state；这与 chunk 循环天然兼容。

### 8.4 Full attention

Prefill：

- flash-attention 风格，q block 32/64，kv block 64/128。
- 16 个 head 并行；GQA 的 k/v 只读 4 份。
- 每个 q block 的 online softmax：维护 `m`、`l`、`acc`（f32）。
- 输出乘 gate 后写回；o_proj 是 4096→2560 GEMV/GEMM。

Decode：

- 每个 head 扫 KV cache；16 个 head 并行。
- 两遍 softmax（max/sum，再加权）或 online；KV 读两次 vs 一次，权衡。
- 长上下文时 KV 读取是瓶颈；8 层 × 32KB/token = 32KB/token；32K ctx 约 1GB，读一遍 10ms（100GB/s），可接受但会显著增加 decode 时间。
- KV cache 可以先 bf16；后续 int8 量化减半。

### 8.5 Conv1d + state

- 权重转置为 `[4, 8192]`；state `[3, 8192]`。HF 的 cache 存 4 个历史输入，但当前 token 只用最近 3 个 + 当前值；llmcp 存 3 个以节省状态内存，输出等价。
- Decode：4 个连续向量乘加 + SiLU；写回 state（shift）。
- Prefill：对每个 t，读 `x[t-3..t]`；边界用 state 或 0。向量化通道维。
- 这是低算术强度算子（4 MAC/channel），主要成本是读写 8192×2B；每 token 约 64KB，24 层 1.5MB，可忽略。

### 8.6 RMSNorm

- 每行 2560 或 256 或 128；f32 平方和。
- 向量化：8 个 f32 累加，最后水平归约；固定归约树。
- 2560 行在 decode 只有 1 行，32 层 × 2 个 norm = 64 次，约 160K 元素，忽略不计。
- Prefill 时行数 = seq，可并行行；每行仍很小。

### 8.7 RoPE / mRoPE

- 文本：cos/sin 表 `[S, 64]`，f32；q/k 的 16/4 个 head 应用。
- 每 token 每个 head 64 维，16+4 个 head = 1280 维，约 2.5KB 读写；可忽略。
- mRoPE：positions `[3, S]`；按 `mrope_section` 重排频率；只影响 cos/sin 表生成，不影响 apply kernel。
- 位置表可以由 runtime 生成（C++ `sin/cos`）或编译进 IR；建议 runtime 生成，避免 IR 里出现 262144 常量。

### 8.8 Embedding / LM head

- Embedding：248320×2560 bf16 = 1.27GB；每 token 只读 5KB，随机访问。mmap + page cache；首 token 可能 page fault，warmup 预取。
- LM head：每 token 读 1.27GB（tied embedding），是 decode 带宽的大头之一；W4 后降到约 0.32GB。
- Greedy：可以只维护 top-1，不必写全部 logits；但为了 top-p/temperature，首期写 1MB logits，成本可忽略（相对 1.27GB 权重读取）。
- 词表 248320 不是 2 的幂；tile 时处理尾部（248320 = 128 × 1940，可被 128 整除；好）。

### 8.9 量化内核

- W8：int8 权重 + per-channel scale；`dot` 用 `sdot`（4×i8 MAC）或先 `sext` 到 i32 再乘。
- W4：group 64/128，两个 nibble/byte；解包后用 `sdot`；scale 在 group 结束时乘。
- 激活量化（W8A8）首期不做；weight-only 先落地。
- 反量化到 f32 再 FMA 只能省带宽；要省算力必须用整数点积。M2 支持 `dotprod`/`i8mm`，这是 W4/W8 的性能关键。

### 8.10 视觉与 MTP（后续）

- 视觉塔：patch embed（Conv3D 等价）、24 个 ViT block、2D RoPE、merge MLP；输出 2560 维，替换 `<|image_pad|>` 的 embedding。
- mRoPE：3D position ids，`mrope_section` 交错。
- MTP：`fc([embed; hidden])` → 1 个全注意力层 → norm → lm_head；用于 draft token。
- 这两块不阻塞文本 MVP；接口上预留 `llmcp.embedding` 的替换点和 `llmcp_forward_*` 的额外输入。

---

## 9. 运行时设计

### 9.1 C ABI

```c
typedef struct llmcp_model   llmcp_model;
typedef struct llmcp_session llmcp_session;

typedef struct {
  const char* weights_path;
  int32_t     num_threads;
  int32_t     max_seq_len;
  bool        verify_weights;
} llmcp_model_options;

typedef struct {
  int32_t max_seq_len;
  float   temperature;
  int32_t top_k;
  float   top_p;
  float   repetition_penalty;
  uint64_t seed;
} llmcp_session_options;

llmcp_model*   llmcp_model_load(const char* model_dir, const llmcp_model_options*);
void           llmcp_model_free(llmcp_model*);

llmcp_session* llmcp_session_create(llmcp_model*, const llmcp_session_options*);
void           llmcp_session_free(llmcp_session*);
void           llmcp_session_reset(llmcp_session*);

int llmcp_prefill(llmcp_session*, const int32_t* tokens, size_t n);
int llmcp_decode(llmcp_session*, int32_t token, float* logits, size_t vocab);
int llmcp_sample(llmcp_session*, const float* logits, int32_t* out_token);

int llmcp_generate(llmcp_session*, const int32_t* prompt, size_t n_prompt,
                   int32_t max_new_tokens,
                   void (*on_token)(int32_t, void*), void* user_data);
```

- 所有函数返回 `int` 错误码；不跨 ABI 抛异常。
- `llmcp_prefill` 内部调用 `llmcp_embed` + `llmcp_forward_prefill` + `llmcp_lm_head`（只算最后一个位置），并把 logits 存到 session。
- `llmcp_decode` 调用 `llmcp_embed` + `llmcp_forward_decode` + `llmcp_lm_head`。
- 词表大小从 config 读取，调用方不需要知道 248320。

### 9.2 会话与内存

```cpp
struct Session {
  // states
  std::vector<bf16> kv;              // [8, 4, max_seq, 256]
  std::vector<float> delta_state;    // [24, 32, 128, 128]
  std::vector<bf16> conv_state;      // [24, 3, 8192]
  std::vector<uint8_t> workspace;    // arena
  // bookkeeping
  int64_t pos = 0;
  std::vector<float> logits;         // [vocab]
  Sampler sampler;
};
```

- KV/state/workspace 在 session 创建时一次性分配（对齐 64B）。
- `max_seq_len` 默认 4096 或 8192；用户可调，受内存限制。
- `reset` 清零 state/KV（或只重置 pos；KV 中未使用部分无需清零，但 DeltaNet state 必须清零）。
- 权重 arena 在 model 级共享，只读；多 session 共享。

### 9.3 权重加载

- `mmap(MAP_PRIVATE, PROT_READ)` 权重 blob；不 `memcpy` 到堆。
- `madvise(MADV_WILLNEED)` 预取前几层（避免首 token 大量 page fault）；不要一次预取 9.3GB（会挤占 page cache）。
- 可选 `mlock` 热点层（前几层/lm_head），但 24GB 内存下要谨慎。
- 校验 header magic/version/layout hash；`--verify-weights` 时逐 tensor sha256（慢，仅调试）。

### 9.4 调度与采样

- 单 session、单请求；prefill 一次，之后逐 token decode。
- 采样：greedy、temperature、top-k、top-p、repetition/frequency/presence penalty；复杂度 O(vocab)，1MB logits，耗时可忽略。
- 停止：`<|im_end|>` 248046、`<|endoftext|>` 248044、max_new_tokens、用户自定义 stop 字符串（detokenize 后匹配）。
- 流式：每 token 回调；增量 detokenizer 处理 UTF-8 多字节边界（不完整的字节缓冲到下一次）。
- 思考模式：默认 `enable_thinking=true`，prompt 以 `<|im_start|>assistant\n<think>\n` 结尾；关闭时追加 `<think>\n\n</think>\n\n`。`</think>` 只是文本，不是停止 token。

### 9.5 Tokenizer 与 Chat

**BPE 实现要点**：

- `tokenizer.json` 的 vocab 248044 + added tokens 33；BPE merges 247587。
- Byte-level：先把 UTF-8 字节映射到 GPT-2 byte-to-unicode，再做 BPE，最后 decoder 还原字节。
- 预分词正则（Qwen 版）：
  ```text
  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
   ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  ```
- 用 utf8proc 做 NFC + Unicode 分类；手写扫描器实现上述分支，不依赖 PCRE2。
- Added tokens（`<|im_start|>` 等）在 BPE 之前做最长匹配；`<think>` 不是 special，但也是 added token，需原子匹配。
- 增量 detokenize：维护 pending bytes，遇到不完整 UTF-8 序列时等待下一个 token。

**Chat template（首期最小实现）**：

```text
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\n{user}<|im_end|>\n
<|im_start|>assistant\n<think>\n
```

- 支持多轮；工具调用/视觉 token 后续。
- 不在 C++ 里实现完整 Jinja；提供 `llmcp_chat_format(messages, enable_thinking)` 的最小模板，复杂模板由上层预处理成 prompt 字符串再 tokenize。
- 特殊 token 直接用 id 拼接，避免 tokenizer 把 `<|im_start|>` 拆开。

### 9.6 线程池

```cpp
class ThreadPool {
 public:
  explicit ThreadPool(int numThreads);
  void parallelFor(int64_t begin, int64_t end, int64_t grain,
                   llvm::function_ref<void(int64_t, int64_t)> fn);
};
```

- 持久线程，条件变量唤醒；每次调用一个 barrier。
- 静态分区：`tasks = ceil((end-begin)/grain)`，按线程轮转；不做 work stealing（简单、确定）。
- 线程亲和性：macOS 不暴露硬亲和；用 `QOS_CLASS_USER_INTERACTIVE` 提升 P 核调度（可选）。
- 嵌套调用检测：如果已在池内，顺序执行，避免死锁。

### 9.7 错误与确定性

- 错误码：`LLMCP_OK`、`INVALID_ARGUMENT`、`OUT_OF_MEMORY`、`WEIGHT_MISMATCH`、`BACKEND_ERROR`。
- 权重/配置不匹配在加载时失败，不进入 forward。
- 线程数变化会改变归约顺序；golden 测试固定 `--threads=1` 或固定分区。
- decode 的 state 更新必须幂等可重置；异常中断后 `reset` 能恢复。

---

## 10. 量化方案

### 10.1 路线

| 阶段 | 方案 | 目标 | 风险 |
|---|---|---|---|
| Q0 | BF16 | 正确性基线 | 内存 9.3GB |
| Q1 | W8 per-channel（int8） | 带宽减半，精度近无损 | 需要 int8 dot 才有算力收益 |
| Q2 | W4 group=64/128（int4） | 带宽 1/4，decode 提速 | 解包开销、精度 |
| Q3（可选） | W8A8 / KV int8 | 进一步提升 | 激活量化误差、校准 |

首期做 weight-only；W8A8 需要校准和额外误差控制，放到 Q3。

### 10.2 格式

```text
W8:
  qweight: int8[N, K]         // per-channel scale
  scales:  f16[N] 或 bf16[N]

W4 (group g):
  qweight: uint8[N, K/2]      // 两个 nibble 一个 byte，低 nibble 在前
  scales:  f16[N, K/g]
  zero:    f16[N, K/g]        // 可选，对称量化可省略
```

- 对称量化：`q = clamp(round(w/s), -8, 7)`（W4）或 `-128..127`（W8）；`w' = q*s`。
- group 内 scale 用 `max|w| / qmax`；group 64 是精度/开销的折中。
- 打包布局：`[N/BN, K/BK, BN, BK]` 可提升 prefill 的局部性；首期 row-major，M3 再评估。
- embedding/lm_head 对精度敏感：先 W8，W4 需单独评估；`A_log`、norm、`dt_bias` 不量化。

### 10.3 量化算子降级

```mlir
%y = llmcp.qmatmul %x, %wq, %scales {scheme = #llmcp.quant<w4, group=64>}
    : (tensor<1x2560xbf16>, memref<9216x1280xi8>, memref<9216x40xf16>)
      -> tensor<1x9216xbf16>
```

降级两条路：

1. **通用**：`linalg.generic` 在 K 归约中解包、乘 scale、f32 累加。可移植，但解包在标量/向量层可能低效。
2. **专用 microkernel**：`llmcp.microkernel` 调用 `extern "C"` 的 `llmcp_gemv_w4`（C/NEON intrinsics 编译），由编译器生成调用。性能上限高，但引入手写 kernel。

建议：Q1/Q2 先走通用路径保证正确；性能不达标时再引入 microkernel，并把 microkernel 作为 `llmcp` 的一个合法 lowering 目标（不是绕过编译器）。

### 10.4 精度验证

- 每层输出 SNR、max abs error。
- 全模型 logits 的 KL divergence、top-1/top-5 一致率。
- 小样本困惑度（WikiText/中文样本）相对误差 < 1%。
- 生成 128 token 的人工检查（重复、乱码、语言混杂）。
- 量化校准数据：从通用语料取 128 条，长度 512；weight-only 不需要激活校准，W8A8 才需要。

### 10.5 性能预期

| 方案 | 权重读取 | 计算 | 预期 decode（M2） |
|---|---:|---|---:|
| BF16 | 9.32GB | f32 FMA | 4–7 tok/s |
| W8（dequant→f32） | 4.8GB | f32 FMA | 6–9 tok/s |
| W8 + sdot | 4.8GB | int8 dot | 10–15 tok/s |
| W4 + sdot | 2.6GB | int8 dot | 12–20 tok/s |

> 这些是工程估算，不是承诺；M3 用实测替换。

---

## 11. 测试与验证

### 11.1 参考实现

- Python：`/Users/guorui/Work/models/venv314`（Python 3.14.8，`torch 2.14.1` + `transformers 5.20.2`；PyPI 版已含 `qwen3_5`）。
- 参考配置：`attn_implementation="eager"`、`torch_dtype=torch.bfloat16`、`device="cpu"`、`use_cache=True`。
- 导出：
  - prompt token ids；
  - 每层输入/输出（选定位置）；
  - q/k/v、attention 输出、DeltaNet state、conv state；
  - 最后位置 logits；
  - 贪心 64 token。
- 参考数据格式：`.npy` + JSON metadata；不入 git；脚本可复现。
- 小模型参考：`python/reference/tiny_reference.py` 用 NumPy 实现同构小模型（2 层、hidden 64），供 CI 使用，不依赖 torch。

### 11.2 测试分层

| 层级 | 内容 | 工具 | 触发 |
|---|---|---|---|
| L0 | 方言 parse/print/verify | `llmcp-opt` + lit | 每次提交 |
| L1 | 单算子数值（随机输入） | C++ gtest / Python | 每次提交 |
| L2 | 单层数值（layer 0/11） | Python golden | 每日/本地 |
| L3 | 全模型 logits | Python golden | 本地/夜间 |
| L4 | 生成质量/性能 | CLI + benchmark | 夜间/发版 |
| L5 | 回归（tiny model） | CTest | 每次提交 |

### 11.3 容差

见 model-spec 第 12 节。关键原则：

- 单算子用 f32 参考，阈值紧；全模型用 bf16 参考，阈值宽。
- 不比较中间层的所有元素，选首/中/末位置，降低数据量。
- 生成对比用 token id，不比较文本；tokenizer 单独测。
- 浮点非确定性：固定线程数/归约顺序后再比较。

### 11.4 Golden 流程

```text
python/reference/dump_golden.py \
  --model /Users/guorui/Work/models/qwen3.5-4b \
  --prompts tests/e2e/prompts.json \
  --out tests/golden/ \
  --layers 0,3,11,31 \
  --positions first,mid,last
```

对比：

```text
python/reference/compare.py \
  --golden tests/golden/ \
  --actual build/out/ \
  --report build/report.json
```

报告包含每层 SNR、max abs、top-1/top-5、token 一致率；CI 只跑 tiny 模型，真实模型在本地/夜间。

### 11.5 CI

- macOS arm64：Homebrew LLVM 20；构建 + lit + tiny e2e。
- Linux x86_64：LLVM 20（apt/源码预编译）；构建 + lit + tiny e2e；可选 AVX2。
- 不在 CI 下载 9.3GB 模型；tiny 模型权重由脚本生成（固定 seed）。
- 缓存 CMake/ccache；构建时间控制在 10 分钟内。
- 代码格式：`clang-format --dry-run --Werror`；`clang-tidy` 可选。

### 11.6 性能基准

- 固定 prompt 长度 16/128/512/2048；输出 128 token。
- 指标：TTFT、prefill tok/s、decode tok/s、RSS、编译时间、二进制大小。
- 工具：`scripts/bench.sh` 调用 `llmcp-run --bench`，输出 JSON；与上次结果对比。
- 机器状态：插电、关闭后台、固定线程数；记录温度/频率（可选）。
- 基准不通过不阻塞功能提交，但进入趋势图。

---

## 12. 里程碑与排期

### 12.1 总表

| 阶段 | 周期 | 目标 | 关键交付 | 退出标准 |
|---|---:|---|---|---|
| M0 基础设施 | 2w | 能构建、能解析、能参考 | CMake/CI、`llmcp-opt` 骨架、方言骨架、ModelConfig/safetensors、WeightLayout、tokenizer、HF golden 脚本 | `llmcp-opt` 可 parse/print；tokenizer 与 HF 20 条字符串一致；tiny manifest 生成 |
| M1 全注意力切片 | 3w | 单层数值正确 | full layer（layer 11）+ MLP + embed/lm_head 的 llmcp IR、lowering、decode AOT、逐层对比 | layer 11 输出 SNR > 30dB；q/k/v/attn/mlp 中间量与 HF 对齐；tiny 同构模型 logits top-1 一致 |
| M2 混合 decode | 3w | 端到端正确 | DeltaNet decode、conv state、gated norm、32 层 group loop、KV cache、逐 token prefill、AOT C ABI | 全模型 logits max abs < 0.1；64 token 贪心一致率 ≥ 95%；token-by-token prefill 可跑 |
| M3 Prefill/性能 | 4w | 可用速度 | chunked DeltaNet、flash attention、tile/vector、线程池、arena、bucket | BF16 decode ≥ 3 tok/s；prefill 512 ≥ 2 tok/s；编译 < 5 min；RSS < 12GB |
| M4 量化/产品化 | 3w | 可交付 | W8/W4 pack、qmatmul、CLI、C API、chat、benchmark | W4 decode ≥ 12 tok/s；生成质量通过；API 稳定 |
| M5 多模态/MTP | 3–4w | 扩展 | 视觉塔、mRoPE、图像预处理、MTP draft | 图像问答可用；mRoPE 与 HF 对齐 |

### 12.2 关键路径

```text
M0 权重布局/参考数据
  → M1 full layer 数值
    → M2 DeltaNet decode 数值
      → M2 端到端
        → M3 chunked prefill 数值
          → M3 性能
            → M4 量化
              → M5 多模态
```

最长的杆是 **DeltaNet 的数值正确性**，不是 MLIR 基础设施。建议在 M0 就把 HF 的逐层 golden 和 tiny 参考搭好，M1/M2 只做“对齐”，不重复造参考。

### 12.3 每阶段任务清单

**M0**

> 当前进度（2026-10-25）：CMake/`llmcp-opt`、`llmcp` 方言、`ModelConfig`/`Safetensors`/`WeightLayout`、`ModelBuilder`/`llmcp-dump`、`llmcp-inspect`、lit 与 C++ 单测已完成并通过；tokenizer、HF golden、tiny 模型尚未开始。

- [ ] CMake + `find_package(MLIR)` + `llmcp-opt` + lit 配置。
- [ ] `LLMCPDialect`/`LLMCPOps` 骨架（先 2 个 op：linear、rms_norm）。
- [ ] `ModelConfig` 解析与校验；`WeightLayout`（bf16）；`llmcp-pack` 生成 manifest。
- [ ] tokenizer：BPE + byte-level + added tokens + NFC；与 HF 对照。
- [ ] 确认 `venv314` 的依赖（`pytest 9.1.1`、`lit 23.1.4` 已装）、HF 参考脚本、8 组 prompt、golden 格式。
- [ ] tiny 模型（2 层、hidden 64）与 NumPy 参考。
- [ ] 环境验证记录、CI 骨架。

**M1**

- [ ] `llmcp.linear`、`rms_norm`、`rope`、`attention`、`silu_mul`、`add` 的 lowering。
- [ ] full layer（layer 11）+ MLP 的 IR builder。
- [ ] decode AOT，`llmcp_lm_head`，C ABI 雏形。
- [ ] 单层 golden 对比（q/k/v、attention 输出、mlp 输出）。
- [ ] `llmcp-opt` lit 测试。

**M2**

- [ ] `llmcp.causal_conv1d`、`gated_delta_net`（decode）。
- [ ] gated RMSNorm、A_log/dt_bias 处理。
- [ ] 32 层 group loop；三类状态；KV cache 布局。
- [ ] 逐 token prefill；端到端 golden；贪心生成。
- [ ] `llmcp.attention` 的 decode KV scan。

**M3**

- [ ] chunked DeltaNet（含三角求解）；与 recurrent 对比。
- [ ] flash attention prefill；causal mask；online softmax。
- [ ] 自定义 tile/vectorize pass；GEMV 专用 schedule。
- [ ] `scf.forall` → 线程池；arena 规划；bucket。
- [ ] benchmark 脚本与 roofline 报告。

**M4**

- [ ] W8/W4 pack；qmatmul lowering；int8 dot（microkernel 或 vector）。
- [ ] CLI（`llmcp-run`）、C API、chat template、采样。
- [ ] 量化精度报告；性能对比。
- [ ] 文档/示例/错误处理。

**M5**

- [ ] 视觉塔（patch embed、ViT block、2D RoPE、merger）。
- [ ] mRoPE 3D positions；图像 token 替换。
- [ ] MTP fc + layer + draft 采样。
- [ ] 多模态 golden。

### 12.4 人员假设

- 1 名熟悉 MLIR 的编译器工程师：文本路径 15 周，含 M5 约 18–19 周；加 buffer 按 18–22 周规划。
- 2 人：一人负责方言/lowering/vectorize，一人负责模型语义/runtime/测试；可并行，10–12 周。
- 如果没有 MLIR 经验，M0–M1 增加 2–3 周学习成本；建议先做 toy dialect + linalg 教程。

---

## 13. 风险与对策

| 风险 | 概率 | 影响 | 触发信号 | 对策 |
|---|---|---|---|---|
| DeltaNet 语义/数值错误 | 高 | 高 | 单层 SNR 低、state 漂移 | 逐算子 golden；先 recurrent；f32；与 HF 对比 state；tiny 模型 |
| chunked prefill 复杂/不稳定 | 高 | 高 | chunked 与 recurrent 不一致 | 先 token 循环；独立 op；属性测试；分块求解器单测 |
| CPU GEMV/GEMM codegen 差 | 高 | 高 | decode < 2 tok/s | 自定义 vectorize、GEMV schedule、microkernel；外部 BLAS 对标 |
| 量化后精度下降 | 中 | 高 | logits/KL 超阈值 | W8 先；group 64；embedding 单独策略；校准 |
| 内存不足/抖动 | 中 | 中 | RSS 超 20GB、swap | mmap、W4、限制 context、arena、释放临时 buffer |
| 编译时间/IR 爆炸 | 中 | 中 | 编译 > 10 min | group loop、bucket 数量控制、pass 并行、避免展开 32 层 |
| MLIR 版本/工具链差异 | 中 | 低 | pass 不存在/API 变化 | 固定 20.1.8；自研 pass；CI 锁定版本 |
| Homebrew MLIR 缺 pass | 中 | 低 | `mlir-opt` 无 linalg-vectorize | 已验证；用自研 `llmcp-opt`/Transform dialect |
| Tokenizer Unicode 边界 | 中 | 中 | 与 HF token 不一致 | utf8proc + 手写扫描器；大量边界用例 |
| 视觉/MTP 范围蔓延 | 高 | 中 | 文本 MVP 延期 | 明确 M5 可选；接口预留但不实现 |
| 性能预期不现实 | 高 | 中 | 目标 vs 实测差距大 | 先 roofline、后 benchmark；分阶段目标；公开实测 |
| 长上下文 KV 带宽 | 中 | 中 | 32K decode 明显变慢 | 限制默认 context；KV int8；GQA 已减少 KV |
| BF16 目标特性假设错误 | 低 | 中 | illegal instruction/慢 | f32 路径；运行时 CPU 特性检查；microbench |

---

## 14. 决策记录（ADR）

### ADR-1：自建 `llmcp` 方言，而不是 StableHLO/Torch-MLIR

- **背景**：模型含 Gated DeltaNet、三类状态、partial RoPE、zero-centered norm、group-wise 量化。
- **决策**：定义高层 `llmcp` 方言，承载模型语义；再降级到 linalg/vector/LLVM。
- **后果**：需要自己写 op/verifier/lowering；但状态、量化、融合和调度完全可控。IREE/Torch-MLIR 仅作参考。

### ADR-2：C++ ModelBuilder，而不是 Python 导出

- **背景**：部署不想依赖 PyTorch；需要与权重打包共享布局代码。
- **决策**：C++ 读取 config/safetensors，直接构建 MLIR；Python 只做参考实现。
- **后果**：前端代码量增加；但编译链路短、可测试、无 Python 运行时依赖。

### ADR-3：激活 tensor 语义 + 状态 memref 语义

- **背景**：激活是纯数据流，状态是长期可变缓冲区。
- **决策**：`llmcp` 激活用 tensor，权重/状态用 memref；降级时用 `bufferization.to_tensor` 桥接权重，状态用 scf/memref 原地更新。
- **后果**：混合语义需要清晰的 bufferization 边界；但避免把 state 强行 SSA 化，也避免全 buffer 语义失去 linalg 融合。

### ADR-4：单一权重 arena + 编译期偏移

- **背景**：9.3GB 权重不能进 IR；per-tensor global 会产生大量符号和重定位。
- **决策**：`llmcp-pack` 生成一个 blob；`WeightLayout` 计算偏移；编译产物中偏移是常量；运行时 mmap。
- **后果**：换模型需重新编译（形状/布局变化）；但 IR 干净、加载快、便于量化。

### ADR-5：层组循环，而不是展开 32 层

- **背景**：32 层展开会导致 IR 巨大、编译慢；层模式规则。
- **决策**：`scf.for` 遍历 8 个 group；group 内 3 linear + 1 full；group stride 固定。
- **后果**：权重偏移是动态 base + 常量；失去少量常量折叠；换来 8× IR 缩减。需要时可用 `--unroll-layers` 展开做性能实验。

### ADR-6：Decode-first，prefill 后置

- **背景**：chunked DeltaNet 是最复杂的部分；端到端正确性应尽早建立。
- **决策**：先 recurrent decode + 逐 token prefill；再实现 chunked prefill。
- **后果**：早期 prefill 很慢，但正确性基线可靠；chunked 可与 recurrent 对比验证。

### ADR-7：BF16 存储 + F32 累加，不依赖 mixed contract

- **背景**：MLIR 20 的默认 `convert-vector-to-llvm` pipeline 不降级 bf16/f16→f32 的 `vector.contract`（已实测）。
- **决策**：权重 bf16，tile 内 `extf` 到 f32，再做 f32 contract/FMA；state/norm/softmax 用 f32。
- **后果**：寄存器压力/指令数增加；但正确性和 codegen 稳定。后续可用 int8 点积或 bf16 原生指令优化。

### ADR-8：AOT 默认，JIT 仅用于测试

- **背景**：部署需要稳定 ABI 和可调试性；JIT 便于快速迭代。
- **决策**：`llmcp-compile` 生成 `.dylib`/`.so`；`llmcp-run` dlopen；`mlir-runner`/ExecutionEngine 用于单测。
- **后果**：AOT 编译需要目标 CPU 信息；跨机器需重新编译或做多版本。

### ADR-9：自研线程池，不引入 OpenMP

- **背景**：macOS 的 libomp 依赖、线程调度和嵌套控制；kernel 粒度固定。
- **决策**：`scf.forall` 降级到 `llmcp_rt_parallel_for`，运行时持久线程池。
- **后果**：需要自己实现 outline/ABI；但依赖少、可控、确定性强。后续可选 OpenMP。

### ADR-10：主路径 MLIR kernel，外部 BLAS 仅作对标

- **背景**：Accelerate/AMX 的 GEMM 远快于初版 MLIR codegen，尤其在 prefill。
- **决策**：默认不调用 BLAS；提供 `--enable-external-gemm` 作为对标/应急；性能报告同时给出两条曲线。
- **后果**：初版 prefill 可能慢；但编译器目标不被外部库替代。若长期差距过大，再把外部 GEMM 作为正式 lowering 选项。

### ADR-11：连续 KV cache 起步，paged 后续

- **背景**：单 session、batch=1，连续布局最简单、对 attention 最友好。
- **决策**：`[layer, kv_head, pos, head_dim]` 连续；容量固定。
- **后果**：多 session/变长场景会浪费内存；v1 不做连续批处理，后续再引入 paged。

### ADR-12：tied embedding，lm_head 只算最后位置

- **背景**：词表 248320，全位置 logits 内存/计算巨大；生成只需要最后一个位置。
- **决策**：embedding 与 lm_head 共享权重；prefill 只对最后 hidden 算 lm_head。
- **后果**：困惑度评估需要额外循环；生成路径显著省算力。

---

## 15. 附录

### A. 环境验证记录

```text
OS:        macOS 26.6.2 (25G83)
CPU:       Apple M2, arm64, 8 cores
Memory:    24GB
LLVM/MLIR: Homebrew llvm@20, 20.1.8
Tools:     mlir-opt, mlir-translate, mlir-runner, mlir-tblgen, mlir-pdll, llvm-config
CMake:     4.1.1
Ninja:     1.13.2
```

已验证：

1. `find_package(MLIR REQUIRED CONFIG)` + C++17 最小程序可编译运行。
2. `linalg.matmul` 接受 bf16 输入 + f32 输出（verifier 通过）。
3. `transform.structured.vectorize` 可将该 matmul 向量化为 `extf + mulf + multi_reduction`。
4. mixed `vector.contract`（bf16/f16→f32）在默认 pipeline 下**不会**被 `convert-vector-to-llvm` 降级；f32→f32 会。
5. Homebrew `mlir-opt --list-passes` 未注册 `-linalg-vectorize`/`-linalg-tile`；`linalg::vectorize` API 存在。
6. `mlir-runner` 是 LLVM 20 的 CPU 执行驱动（不是 `mlir-cpu-runner`）。
7. `--use-bare-ptr-memref-call-conv`、`one-shot-bufferize`、`hoist-static-allocs`、`buffer-deallocation` 等 pass 存在。

### B. 关键路径速查

| 文件/目录 | 作用 |
|---|---|
| `include/llmcp/Dialect/LLMCP/IR/LLMCPOps.td` | 方言定义 |
| `lib/Dialect/LLMCP/Transforms/LowerToLinalg.cpp` | 语义降级入口 |
| `lib/Model/WeightLayout.cpp` | 权重偏移的唯一真源 |
| `lib/Model/ModelBuilder.cpp` | 生成 prefill/decode IR |
| `lib/Dialect/LLMCP/Transforms/TileAndVectorize.cpp` | 性能关键 pass |
| `lib/Runtime/Session.cpp` | 状态/采样/调度 |
| `tests/lit/` | IR 回归 |
| `python/reference/dump_golden.py` | 数值基准 |

### C. 术语

| 术语 | 含义 |
|---|---|
| Gated DeltaNet | 带门控的 delta rule 线性注意力 |
| recurrent state | DeltaNet 的 `[Hv, K, V]` 状态矩阵 |
| chunked prefill | 将序列分块，用矩阵化 UT transform 计算 DeltaNet |
| GQA | grouped-query attention，16 q head / 4 kv head |
| zero-centered RMSNorm | `x_norm * (1 + w)`，w 初始为 0 |
| mRoPE | 多模态 3D RoPE（T/H/W） |
| arena | 一次性分配、按偏移切分的权重/工作区内存 |
| bucket | 预编译的固定序列长度 |
| AOT | ahead-of-time 编译为共享库 |
| ORC JIT | LLVM 的即时编译执行引擎 |

### D. 参考链接

- [Qwen3.5 文本建模代码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Qwen3.5 配置代码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/configuration_qwen3_5.py)
- [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Vector](https://mlir.llvm.org/docs/Dialects/Vector/)
- [MLIR Transform](https://mlir.llvm.org/docs/Dialects/Transform/)
- [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [MLIR Standalone Example](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone)
- [IREE LinalgExt（参考）](https://github.com/iree-org/iree/tree/main/compiler/src/iree/compiler/Dialect/LinalgExt)

---

## 16. 下一步（本周可执行）

1. 搭 CMake + `llmcp-opt` + `LLMCPDialect`/`LLMCPOps` 骨架，跑通 `llmcp.linear` 的 parse/print。
2. 写 `ModelConfig` + safetensors header 解析，打印 32 层类型与每层参数量，和本文档对齐。
3. 用 `venv314` 跑 HF 参考，导出 1 组 prompt 的逐层激活和 logits。
4. 写 `WeightLayout` 与 `llmcp-pack` 的 bf16 版本，生成 manifest；校验张量总字节数与 checkpoint 的 9,319,737,856 一致（blob 另加 64B header）。
5. 实现 tokenizer 的最小闭环（BPE + special tokens），与 HF 对比 20 条字符串。
6. 用 tiny 模型（2 层、hidden 64）跑通“IR → linalg → LLVM → 可执行”的全链路，作为 CI 的 smoke test。
