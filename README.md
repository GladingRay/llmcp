# llmcp

基于 MLIR 的 Qwen3.5-4B CPU 推理编译器。

## 目标

把 Qwen3.5-4B 的文本推理图编译成 CPU 可执行的共享库：

```text
config + safetensors ──► llmcp dialect ──► linalg/vector/LLVM ──► .dylib
                                                                    │
                                              llmcp-runtime（tokenizer/session/采样）
```

- 后端：CPU，Apple M2 / arm64 优先；
- 工具链：MLIR 20（Homebrew `llvm@20`）；
- 模型：默认 `/Users/guorui/Work/models/qwen3.5-4b`，可用 `-DLLMCP_MODEL_DIR=...` 或环境变量 `LLMCP_MODEL_DIR` 覆盖；
- 主体语言：C++17；Python 仅用于参考实现、测试和脚本。

## 环境

- LLVM/MLIR 20.1.8（Homebrew `llvm@20`）
- Python：`/Users/guorui/Work/models/venv314`（Python 3.14.8，`torch 2.14.1`，`transformers 5.20.2`）
- 第三方源码：git submodule 固定版本（`googletest v1.18.0`、`tokenizers-cpp`），见 [third_party/README.md](third_party/README.md)
- 依赖清单、用途与安装命令见 [设计文档 4.3 节](docs/design.md#43-依赖清单与用途)

## 文档

- [技术路线与架构设计](docs/design.md)：分层架构、方言、pass pipeline、内核、里程碑、风险、ADR
- [模型规格与编译契约](docs/model-spec.md)：Qwen3.5-4B 的形状、公式、状态布局、数值语义
- [单文件权重格式](docs/packed-weights.md)：权重与 manifest 合并、版本头部与迁移方式
- [真实权重数值对齐](docs/real-alignment.md)：单 token、AOT、Q8 对齐结果
- [AOT 缓冲区生命周期](docs/aot-buffer-lifetimes.md)：中间分配释放与回归验证
- [Unicode 流式输出](docs/streaming-unicode.md)：跨 token 字符缓冲与结束刷新
- [FlashAttention-2](docs/flash-attention.md)：full attention 的 online softmax + key tiling
- [算子并行化](docs/parallelization.md)：OpenMP、RMSNorm 合并、elementwise 融合、benchmark
- [INT8 量化](docs/int8-quantization.md)：Q8_0/Q8_1、llmcp-pack、sdot、真实模型接入
- [Linear/Activation 融合](docs/linear-activation-fusion.md)：SwiGLU、GatedRMSNorm、DeltaNet 的 activation epilogue
- [matmul benchmark](docs/matmul-benchmark.md)：模型 matmul shape 与单算子性能
- [CacheLineAlignedVector](docs/cache-line-aligned-vector.md)：128 字节对齐的 tensor 存储
- [CI/CD](docs/ci-cd.md)：GitHub Actions 构建、测试与 release
- [历史性能记录](docs/performance.md)：早期 benchmark；最新数据见本文“性能”节

## 当前状态

当前已经可以端到端运行真实 Qwen3.5-4B 文本模型：

```text
safetensors
  -> llmcp dialect
  -> linalg / scf / vector / LLVM
  -> AOT dylib
  -> llmcp-run：tokenizer / prefill / greedy decode / 流式输出
```

已实现：

- `llmcp` 方言 11 类算子：
  `linear`、`rms_norm`、`add`、`silu_mul`、`mul`、`embedding`、`lm_head`、
  `causal_conv1d`、`gated_delta_net`、`attention`、`gated_rms_norm`；
- `ModelConfig` / `Safetensors` / `WeightLayout`：
  真实模型 32 层、738 tensor、文本 arena 8.41GB；
- `ModelBuilder`：
  embedding、24 层 Gated DeltaNet、8 层 full attention、MLP、final norm、tied lm_head；
- `llmcp-dump`：
  导出完整 prefill 图，支持 `--verify-module` 和 `--print-stats`；
- `llmcp-lower-to-linalg`：
  全部 11 类算子 lowering；
  Linear/LmHead 在支持 BF16 的 arm64 上使用 `bfdot`，否则回退 vector/f32；
  Q8 使用 `sdot`，需要 ARM DotProd；不支持时禁用 Q8；
  attention 使用 FlashAttention-2 风格 online softmax + key tiling；
- `llmcp-pack`：
  单文件 bf16 arena（内嵌 JSON manifest）；
  `--quant q8_0` 生成 Q8_0 arena，按四列 SDOT lane 预打包；
- `llmcp-compile`：
  完整图经过 linalg、elementwise 融合、OpenMP 并行化、
  bf16/bfdot（不支持时 vector/f32）或 Q8/sdot（需要 DotProd），
  AOT 到 LLVM IR / object / shared library；
  `--quant q8_0` 编译真实 Q8 模型；
- `llmcp-run`：
  mmap arena、tokenizers-cpp、chat template、流式输出、
  多 token prefill + greedy decode、`--aot` 和 Q8 AOT；
- `llmcp-benchmark` / `llmcp-matmul-benchmark` / `llmcp-microbench`；
- `CacheLineAlignedVector`：128 字节对齐 tensor 存储。

当前结果：

```text
单 token bf16 对齐：
  max_abs  1.875e-01
  mean_abs 2.476e-02
  top1     11 / 11
  top5     5/5

4 token greedy：
  tokens: 0 271 760 2614

Q8 AOT：
  next 11
  tokens: 0 271 760 2614
  Q8 vs bf16 logits: max_abs 0.509, mean_abs 0.083
  top1/top5 一致

Q8 权重：
  bf16 arena 8.41 GB
  Q8 arena   4.47 GB
  Q8_0 使用四列 lane 打包：每个 block 存 [4×f16 scales][8×16B words]
  embedding/lm_head 共用同一份 Q8_0 表
```

当前限制：

- 只支持文本模型，视觉塔和 MTP 尚未纳入；
- 长序列 AOT decode 还没有系统性对齐；
- 采样目前只有 greedy，temperature/top-k/top-p 未实现；
- Q8 量化 linear matmul 与 tied embedding/lm_head；KV cache、DeltaNet state
  仍为 bf16/f32；
- 没有通用 tiling pass，性能仍主要依赖 OpenMP、bfdot/sdot 和循环顺序；
- 所有性能数字受本机 page cache / 内存压力影响，趋势比绝对值更可靠。

## 性能

以下数据为 **2026-11-19** 在 Apple M2（24 GiB 内存、8 个逻辑核）上使用
`b4b001e` 的实测。
macOS 26.6.2，Release 构建，LLVM 20.1.8，AOT 使用 `-O2 -mcpu=native`，bfdot/sdot 均启用。
测试串行执行；短序列每种精度先完整预热一次，再用新进程测量三次，表中为中位数。
未主动清除系统 page cache，因此这些是预热后的结果，不代表冷启动性能。
原始输出和命令见
[本次测量记录](docs/benchmarks/2026-11-19-q8-embedding.json)。
[2026-11-17 的测量记录](docs/benchmarks/2026-11-17.json) 和
[2026-11-08 的测量记录](docs/benchmarks/2026-11-08.json) 保留作历史参考。

全模型短序列使用 `OMP_NUM_THREADS=8`、`prefill=32 / decode=32`，prompt 为重复的
合成 token `9419`。计时包含 forward 内的运行时开销，排除权重文件加载、动态库加载
和 tokenizer；decode 吞吐统计 32 次自回归 forward。

| 模型 | prefill | prefill tok/s | decode | decode tok/s |
|---|---:|---:|---:|---:|
| bf16 + bfdot | 1.40 s | 22.87 | 5.55 s | 5.76 |
| Q8_0 + sdot | 0.42 s | 76.92 | 2.81 s | 11.39 |

三次测量的吞吐范围：

- bf16 的 prefill 为 22.56–23.03 tok/s，decode 为 5.75–5.77 tok/s。
- Q8_0 的 prefill 为 76.32–77.49 tok/s，decode 为 11.35–11.39 tok/s。

2026-11-19 的连续改动把 Q8 decode 从 8.18 提升到 11.39 tok/s（+39%），
Q8 prefill 从 54.31 提升到 76.92 tok/s。除第 9 节的 GEMV 内核外，还包括：

- decode 的 RMSNorm、逐元素算子和 DeltaNet value tile 改用串行/向量特化 lowering；
- tied embedding/lm_head 改为 Q8_0，行查表时反量化；
- Q8 权重在 `llmcp-pack` 阶段预打包成四列 SDOT lane 布局，prefill GEMM 不再
  每次调用重新打包，decode GEMV 直接复用同一份权重。

GEMV 与打包布局的测量见
[GEMV 与整模型测量记录](docs/matmul-benchmark.md#9-q8-sdot-decode-gemv)。

prefill 和 decode 使用不同的 AOT 入口：

```text
llmcp_model_prefill  -> llmcp_forward_prefill
llmcp_model_decode   -> llmcp_forward_decode
```

两者共享权重和状态布局，但调度和 lowering 实现不同：

生成用的 prefill/decode 入口只对最后一行 hidden state 执行 lm_head，
输出固定为 `[1, vocab_size]`。1024 token、248320 词表时，输出 logits
从约 970 MiB 降为约 0.95 MiB；词表投影的行数从 1024 降为 1。
张量返回入口 `llmcp_model` 仍提供所有位置的 logits，供逐位置数值检查使用。
此输出契约使用 AOT ABI v2，运行时通过 `llmcp_model_abi_version` 校验；
旧动态库需要重新执行 `llmcp-compile --emit=shared` 编译。

- 独立入口：
  - [LowerPrefillGemm.cpp](lib/Dialect/LLMCP/Transforms/LowerPrefillGemm.cpp)
  - [LowerDecodeGemm.cpp](lib/Dialect/LLMCP/Transforms/LowerDecodeGemm.cpp)
- 共享 kernel 和 loop scheduler：
  - [MatmulLoweringCommon.cpp](lib/Dialect/LLMCP/Transforms/MatmulLoweringCommon.cpp)

- prefill：
  - attention 按 query position 并行，head 串行；
  - linear 使用 GEMM：column 并行，row loop 保留；
  - bf16 使用 4-row 寄存器 tile + cache block；
  - 非 BF16 Apple Silicon 自动回退到 vector/f32 路径；
  - `LLMCP_ENABLE_BFDOT` / `LLMCP_ENABLE_SDOT` 可显式控制；
  - Q8 直接读预打包的四列 lane，使用 4-row × 8-column SDOT 寄存器 tile，
    只打包每次变化的激活面板。
    [打包内核](lib/Dialect/LLMCP/Transforms/Q8Matmul.cpp)
  - activation epilogue 在 matmul 输出 store 前完成。
- decode：
  - attention 按 head 并行，query loop 只有 1 次；
  - linear 使用 GEMV：column 并行，去掉 row loop。
  - Q8 权重是预打包的四列 lane 布局；一条 `SDOT` 的四个 lane 同时产生四个
    输出列（每 lane 累加 4 个 K 元素），不再需要 `ADDP` 合并列。
    激活向量量化串行执行，小投影和尾列使用串行循环。
    [decode GEMV 实测](docs/matmul-benchmark.md#9-q8-sdot-decode-gemv)
  - tied embedding/lm_head 使用同一份 Q8_0 表；embedding 查表按行所在 lane
    读取 f16 scale 并反量化为 bf16。

prefill 的 Q8 GEMM 直接读取同一份预打包权重，只打包每次变化的激活面板。

对应的 arena：

```text
bf16: 8.41 GB
Q8:   4.47 GB
```

长上下文 Q8（8 线程，`prefill=1024 / decode=1024`，短序列测试之后单次测量）：

```text
prefill: 11.35 s  (90.23 tok/s)
decode:  98.03 s  (10.45 tok/s)
total:   109.38 s
```

单算子 matmul（`llmcp-matmul-benchmark`）每组运行三次，每次内部预热一次。
1024 方阵计时 80 次迭代，其余形状计时 20 次。下表为吞吐中位数，括号内为
三次范围。两种精度均按 `2*M*N*K/time` 统计吞吐。Q8 计时包含激活量化、
权重打包、缩放累加和输出拷贝。编译与动态库加载在计时前完成。

| M × N × K | 线程 | bf16 GFLOPS（范围） | Q8 等效 GOPS（范围） |
|---|---:|---:|---:|
| 256 × 256 × 256 | 4 | 61.4（59.2–68.6） | 182.6（150.5–251.6） |
| 16 × 8192 × 2560 | 4 | 152.7（150.2–162.9） | 424.7（421.7–459.2） |
| 1024 × 1024 × 1024 | 8 | 208.1（205.6–210.6） | 1072.8（982.4–1075.9） |

1024³ Q8 GEMM 与同线程数的 SDOT 峰值对比。GEMM 基线为 `c6158b7`，指令峰值
使用 `--sdot-threads N --seconds 0.5` 测量。各列使用三次运行的中位数。
[测量方法与指令混合对照](docs/matmul-benchmark.md#8-q8_0-sdot-的-1024-方阵)

| 线程 | 基线 GOPS | 打包内核 GOPS | SDOT 峰值 GOPS | 峰值占比 |
|---:|---:|---:|---:|---:|
| 1 | 131.6 | 288.3 | 434.1 | 66.4% |
| 4 | 453.9 | 921.6 | 1609.9 | 57.2% |
| 8 | 491.6 | 1072.8 | 2171.9 | 49.4% |

CPU 指令与内存带宽（`llmcp-microbench`）：三次独立运行的中位数和范围。
compute 单线程，每项预热后计时至少 1 秒；带宽使用 8 线程、256 MiB 缓冲区单次遍历，
包含线程创建和回收开销，copy 按读写合计流量计算。硬件特性为
`FP16=1 DotProd=1 BF16=1 I8MM=1`。

| 项目 | 中位数 | 三次范围 |
|---|---:|---:|
| neon_bf16_dot | 55.39 GFLOPS | 55.32–55.46 GFLOPS |
| neon_i8_dot | 438.96 GOPS | 424.59–443.17 GOPS |
| neon_i8_mma | 272.66 GOPS | 254.73–276.22 GOPS |
| read | 48.79 GB/s | 34.23–52.81 GB/s |
| write | 62.54 GB/s | 53.26–63.77 GB/s |
| copy | 68.40 GB/s | 59.46–68.94 GB/s |

当前 `neon_i8_dot` 基准使用 16 条独立累加链，每轮展开到 64 条寄存器间 SDOT，
每条指令计 16 次乘加，即 32 次整数运算。
`--sdot-threads N` 仅测 SDOT，N 个线程预热后同步开始，用 `32 × 总指令数`
除以共同的墙钟时间计算吞吐量。测量 Q8 GEMM 峰值占比时，使用相同线程数并重复采样。

以下为单次运行示例：短序列每种精度先预热一次，再重复三次；matmul 和微基准各
重复三次，长上下文只运行一次。所有实际命令均保存在本次测量记录中。

```bash
OMP_NUM_THREADS=8 ./build/bin/llmcp-benchmark \
  --weights build/weights.bin \
  --aot build/libqwen35_aot.dylib \
  --prefill-tokens 32 --decode-tokens 32

OMP_NUM_THREADS=8 ./build/bin/llmcp-benchmark \
  --weights build/weights-q8.bin \
  --aot build/libqwen35_aot_q8.dylib \
  --prefill-tokens 32 --decode-tokens 32

OMP_NUM_THREADS=8 ./build/bin/llmcp-benchmark \
  --weights build/weights-q8.bin \
  --aot build/libqwen35_aot_q8.dylib \
  --prefill-tokens 1024 --decode-tokens 1024

./build/bin/llmcp-matmul-benchmark \
  --m 16 --n 8192 --k 2560 --iters 20 --threads 4 --q8

./build/bin/llmcp-matmul-benchmark \
  --m 1024 --n 1024 --k 1024 --iters 80 --threads 8 --q8

./build/bin/llmcp-microbench --seconds 1 --buffer-mb 256 --threads 8
./build/bin/llmcp-microbench --sdot-threads 8 --seconds 0.5
```

注意：

- 该机器在反复加载 8 GB 权重后会出现明显 page cache / 内存压力，
  绝对值可能波动；本次保留全部样本并报告中位数和范围；
- 短序列为 32/32，长序列为 1024/1024；decode 计时包括 argmax 和运行时开销；
- 尚未做 CUDA/Metal 等 GPU 后端对比；
- prefill/decode 的性能差异来自独立入口、attention 调度以及 GEMM/GEMV 分叉。

## 构建与测试

```bash
# 0. 拉取 vendored 第三方源码
git submodule update --init third_party/googletest third_party/tokenizers-cpp

# 1. 构建（C/C++ 都用 Homebrew clang 20）
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/llvm \
  -DLLMCP_MODEL_DIR=/Users/guorui/Work/models/qwen3.5-4b
cmake --build build -j 8

# 2. 全量测试：默认 16 个 CTest 目标（golden 需要已生成的 arena/dylib）
ctest --test-dir build --output-on-failure

# 真实模型路径可用 CMake 变量或环境变量覆盖：
#   cmake -S . -B build -DLLMCP_MODEL_DIR=/path/to/qwen3.5-4b
#   LLMCP_MODEL_DIR=/path/to/qwen3.5-4b cmake -S . -B build
# 模型目录不存在时，依赖真实模型的测试会自动跳过。

# 3. 检查真实模型配置与权重布局（只读 header）
./build/bin/llmcp-inspect --model /Users/guorui/Work/models/qwen3.5-4b

# 4. IR / lowering
./build/bin/llmcp-opt tests/lit/Dialect/LLMCP/ops.mlir
./build/bin/llmcp-opt tests/lit/Conversion/lower-linear.mlir --llmcp-lower-to-linalg

# 5. dump 完整 prefill 图
./build/bin/llmcp-dump --model /Users/guorui/Work/models/qwen3.5-4b \
  --verify-module --print-stats -o build/qwen35_graph.mlir

# 6. numeric（11 类算子，含 Q8 linear/lm_head）
ctest --test-dir build -R llmcp-numeric --output-on-failure

# 7. 输出 token golden test
#    需要 build/weights*.bin（内嵌 manifest）和 AOT dylib 已生成
ctest --test-dir build -R llmcp-golden-tokens --output-on-failure
```

`--print-stats` 当前输出：

```text
functions: 8
linear: 496, rms_norm: 130, delta: 48, attention: 16, conv: 48,
gated_norm: 48, embedding: 1, lm_head: 1, silu_mul: 64, add: 128,
weight_views: 852
```

`llmcp-inspect` 当前输出：

```text
layers: 32 (24 linear, 8 full)
tensors: 738
checkpoint bytes: 9319737856
arena bytes: 8411510336 (header 64, group stride 892513344, groups 8)
```

## 常用命令

```bash
# 打包 bf16 权重
./build/bin/llmcp-pack --model /Users/guorui/Work/models/qwen3.5-4b \
  --out build/weights.bin

# 只校验，不写 arena
./build/bin/llmcp-pack --model /Users/guorui/Work/models/qwen3.5-4b --dry-run

# 打包 Q8_0 权重
./build/bin/llmcp-pack --model /Users/guorui/Work/models/qwen3.5-4b \
  --quant q8_0 --out build/weights-q8.bin

# AOT 编译 bf16 / Q8
./build/bin/llmcp-compile --model /Users/guorui/Work/models/qwen3.5-4b \
  --emit=shared --out build/libqwen35_aot.dylib
./build/bin/llmcp-compile --model /Users/guorui/Work/models/qwen3.5-4b \
  --quant q8_0 --emit=shared --out build/libqwen35_aot_q8.dylib

# 手写路径：单 token golden / 多 token greedy
./build/bin/llmcp-run --weights build/weights.bin \
  --text Hello --golden build/golden_logits.bin
./build/bin/llmcp-run --weights build/weights.bin \
  --text 'Hello world' --max-new-tokens 2

# AOT bf16 / Q8
./build/bin/llmcp-run --weights build/weights.bin \
  --text Hello --aot build/libqwen35_aot.dylib
./build/bin/llmcp-run --weights build/weights-q8.bin \
  --text Hello \
  --aot build/libqwen35_aot_q8.dylib

# Chat template（默认 thinking；可加 --no-thinking / --system）
./build/bin/llmcp-run --weights build/weights.bin \
  --text Hello --chat --max-new-tokens 8 \
  --aot build/libqwen35_aot.dylib

# AOT 性能测试
./build/bin/llmcp-benchmark --weights build/weights.bin \
  --aot build/libqwen35_aot.dylib \
  --prefill-tokens 32 --decode-tokens 32

# 单算子 matmul benchmark；--q8 走 Q8/sdot
./build/bin/llmcp-matmul-benchmark --m 256 --n 256 --k 256 --iters 10 --threads 4
./build/bin/llmcp-matmul-benchmark --m 16 --n 8192 --k 2560 --iters 20 --threads 4 --q8

# CPU 指令与带宽 microbench
./build/bin/llmcp-microbench --seconds 1 --buffer-mb 256 --threads 8

# SDOT 峰值 microbench，四个计算线程
./build/bin/llmcp-microbench --seconds 1 --sdot-threads 4
```

## License

MIT License，见 [LICENSE](LICENSE)。
