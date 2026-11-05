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
- [真实权重数值对齐](docs/real-alignment.md)：单 token、AOT、Q8 对齐结果
- [FlashAttention-2](docs/flash-attention.md)：full attention 的 online softmax + key tiling
- [算子并行化](docs/parallelization.md)：OpenMP、RMSNorm 合并、elementwise 融合、benchmark
- [INT8 量化](docs/int8-quantization.md)：Q8_0/Q8_1、llmcp-pack、sdot、真实模型接入
- [matmul benchmark](docs/matmul-benchmark.md)：模型 matmul shape 与单算子性能
- [CacheLineAlignedVector](docs/cache-line-aligned-vector.md)：128 字节对齐的 tensor 存储
- [历史性能记录](docs/performance.md)：2026-10-26 的原始 benchmark

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

- `llmcp` 方言 10 类算子：
  `linear`、`rms_norm`、`add`、`silu_mul`、`embedding`、`lm_head`、
  `causal_conv1d`、`gated_delta_net`、`attention`、`gated_rms_norm`；
- `ModelConfig` / `Safetensors` / `WeightLayout`：
  真实模型 32 层、738 tensor、文本 arena 8.41GB；
- `ModelBuilder`：
  embedding、24 层 Gated DeltaNet、8 层 full attention、MLP、final norm、tied lm_head；
- `llmcp-dump`：
  导出完整 prefill 图，支持 `--verify-module` 和 `--print-stats`；
- `llmcp-lower-to-linalg`：
  全部 10 类算子 lowering；
  Linear/LmHead 在 arm64 上使用 `bfdot`；
  attention 使用 FlashAttention-2 风格 online softmax + key tiling；
- `llmcp-pack`：
  bf16 arena + JSON manifest；
  `--quant q8_0` 生成 Q8_0 arena；
- `llmcp-compile`：
  完整图经过 linalg、elementwise 融合、OpenMP 并行化、
  bf16/bfdot 或 Q8/sdot，AOT 到 LLVM IR / object / shared library；
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
  Q8 vs bf16 logits: max_abs 0.509, mean_abs 0.087
  top1/top5 一致

Q8 权重：
  bf16 arena 8.41 GB
  Q8 arena   5.07 GB
  （embedding/lm_head 仍为 bf16）
```

当前限制：

- 只支持文本模型，视觉塔和 MTP 尚未纳入；
- 长序列 AOT decode 还没有系统性对齐；
- 采样目前只有 greedy，temperature/top-k/top-p 未实现；
- Q8 只量化 linear matmul，embedding/lm_head、KV cache、DeltaNet state 仍为 bf16/f32；
- 没有通用 tiling pass，性能仍主要依赖 OpenMP、bfdot/sdot 和循环顺序；
- 所有性能数字受本机 page cache / 内存压力影响，趋势比绝对值更可靠。

## 性能

以下数据为 2026-11-02 在本机 Apple M2 上的实测，Release 构建，
`OMP_NUM_THREADS=8`，AOT dylib，`prefill=32 / decode=32`：

| 模型 | prefill | prefill tok/s | decode | decode tok/s |
|---|---:|---:|---:|---:|
| bf16 + bfdot | 7.50 s | 4.27 | 6.43 s | 4.98 |
| Q8_0 + sdot | 5.65 s | 5.67 | 4.77 s | 6.71 |

prefill 和 decode 使用不同的 AOT 入口：

```text
llmcp_model_prefill  -> llmcp_forward_prefill
llmcp_model_decode   -> llmcp_forward_decode
```

两者共享权重和状态布局，但调度和 lowering 实现不同：

- 独立 lowering 文件：
  - [LowerPrefillGemm.cpp](lib/Dialect/LLMCP/Transforms/LowerPrefillGemm.cpp)
  - [LowerDecodeGemm.cpp](lib/Dialect/LLMCP/Transforms/LowerDecodeGemm.cpp)

- prefill：
  - attention 按 query position 并行，head 串行；
  - linear 使用 GEMM：column 并行，row loop 保留。
- decode：
  - attention 按 head 并行，query loop 只有 1 次；
  - linear 使用 GEMV：column 并行，去掉 row loop。

对应的 arena：

```text
bf16: 8.41 GB
Q8:   5.07 GB
```

单算子 matmul（4 线程，`llmcp-matmul-benchmark`）：

```text
256 x 256 x 256:
  bf16: 26.9 GFLOPS
  q8:   53.6 GFLOPS

M=16, N=8192, K=2560:
  bf16: 59.5 GFLOPS
  q8:   120.8 GFLOPS
```

CPU 指令与内存带宽（`llmcp-microbench`）：

```text
Apple M2, 8 logical CPUs
FP16=1 DotProd=1 BF16=1 I8MM=1

neon_bf16_dot   55.3 GFLOPS
neon_i8_dot    286.2 GOPS

read            36.7 GB/s
write           63.3 GB/s
copy            56.9 GB/s
```

复现命令：

```bash
OMP_NUM_THREADS=8 ./build/bin/llmcp-benchmark \
  --weights build/weights.bin --manifest build/weights.json \
  --aot build/libqwen35_aot.dylib \
  --prefill-tokens 32 --decode-tokens 32

OMP_NUM_THREADS=8 ./build/bin/llmcp-benchmark \
  --weights build/weights-q8.bin --manifest build/weights-q8.json \
  --aot build/libqwen35_aot_q8.dylib \
  --prefill-tokens 32 --decode-tokens 32

./build/bin/llmcp-matmul-benchmark \
  --m 16 --n 8192 --k 2560 --iters 5 --threads 4 --q8

./build/bin/llmcp-microbench --seconds 1 --buffer-mb 256
```

注意：

- 该机器在反复加载 8 GB 权重后会出现明显 page cache / 内存压力，
  绝对值可能波动；上表用于同一时段的 bf16/Q8 对比；
- prefill 包含 32 个 prompt token，decode 统计 32 个生成 token；
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

# 2. 全量测试：默认 14 个 CTest 目标
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

# 6. numeric（10 类算子，含 Q8 linear/lm_head）
ctest --test-dir build -R llmcp-numeric --output-on-failure
```

`--print-stats` 当前输出：

```text
functions: 7
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
  --out build/weights.bin --manifest build/weights.json

# 只校验，不写 arena
./build/bin/llmcp-pack --model /Users/guorui/Work/models/qwen3.5-4b --dry-run

# 打包 Q8_0 权重
./build/bin/llmcp-pack --model /Users/guorui/Work/models/qwen3.5-4b \
  --quant q8_0 --out build/weights-q8.bin --manifest build/weights-q8.json

# AOT 编译 bf16 / Q8
./build/bin/llmcp-compile --model /Users/guorui/Work/models/qwen3.5-4b \
  --emit=shared --out build/libqwen35_aot.dylib
./build/bin/llmcp-compile --model /Users/guorui/Work/models/qwen3.5-4b \
  --quant q8_0 --emit=shared --out build/libqwen35_aot_q8.dylib

# 手写路径：单 token golden / 多 token greedy
./build/bin/llmcp-run --weights build/weights.bin \
  --manifest build/weights.json --text Hello --golden build/golden_logits.bin
./build/bin/llmcp-run --weights build/weights.bin \
  --manifest build/weights.json --text 'Hello world' --max-new-tokens 2

# AOT bf16 / Q8
./build/bin/llmcp-run --weights build/weights.bin \
  --manifest build/weights.json --text Hello --aot build/libqwen35_aot.dylib
./build/bin/llmcp-run --weights build/weights-q8.bin \
  --manifest build/weights-q8.json --text Hello \
  --aot build/libqwen35_aot_q8.dylib

# Chat template（默认 thinking；可加 --no-thinking / --system）
./build/bin/llmcp-run --weights build/weights.bin \
  --manifest build/weights.json --text Hello --chat --max-new-tokens 8 \
  --aot build/libqwen35_aot.dylib

# AOT 性能测试
./build/bin/llmcp-benchmark --weights build/weights.bin \
  --manifest build/weights.json --aot build/libqwen35_aot.dylib \
  --prefill-tokens 32 --decode-tokens 32

# 单算子 matmul benchmark；--q8 走 Q8/sdot
./build/bin/llmcp-matmul-benchmark --m 256 --n 256 --k 256 --iters 10 --threads 4
./build/bin/llmcp-matmul-benchmark --m 16 --n 8192 --k 2560 --iters 5 --threads 4 --q8

# CPU 指令与带宽 microbench
./build/bin/llmcp-microbench --seconds 1 --buffer-mb 256
```

## License

MIT License，见 [LICENSE](LICENSE)。
