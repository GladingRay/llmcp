# llmcp

基于 MLIR 的 Qwen3.5-4B 大模型 CPU 推理编译器。

## 目标

把 Qwen3.5-4B 的文本推理图编译成 CPU 可执行的共享库：

```text
config + safetensors ──► llmcp dialect ──► linalg/vector/LLVM ──► .dylib
                                                                    │
                                              llmcp-runtime（tokenizer/session/采样）
```

- 后端：CPU（Apple M2 / arm64 优先，后续 x86_64）
- 工具链：MLIR 20（Homebrew `llvm@20`）
- 模型：`/Users/guorui/Work/models/qwen3.5-4b`
- 主体语言：C++17；Python 仅用于参考实现和脚本

## 环境

- LLVM/MLIR 20.1.8（Homebrew `llvm@20`）
- Python：`/Users/guorui/Work/models/venv314`（Python 3.14.8，`torch 2.14.1`，`transformers 5.20.2`）
- 第三方源码：git submodule 固定版本（`utf8proc v2.11.0`、`googletest v1.18.0`），见 [third_party/README.md](third_party/README.md)
- 依赖清单、用途与安装命令见 [设计文档 4.3 节](docs/design.md#43-依赖清单与用途)

## 文档

- [技术路线与架构设计](docs/design.md)：分层架构、方言设计、pass pipeline、内核、运行时、里程碑、风险、ADR
- [模型规格与编译契约](docs/model-spec.md)：Qwen3.5-4B 的精确形状、公式、状态布局、数值语义

## 当前状态

M0 基础设施已落地：

- CMake + MLIR 20 构建，`llmcp-opt` 可解析/打印/校验 IR
- `llmcp` 方言：`linear`/`rms_norm`/`add`/`silu_mul`/`embedding`/`lm_head`/`causal_conv1d`/`gated_delta_net`/`attention`/`gated_rms_norm`，含 TableGen 定义与 C++ verifier
- `ModelConfig`：解析并校验 `config.json`（层类型、GQA、RoPE、mRoPE、线性注意力头数等）
- `Safetensors`：只读 header，不加载 tensor 数据
- `WeightLayout`：文本权重 arena 的确定性偏移、64B 对齐、group stride
- `llmcp-inspect`：对真实 Qwen3.5-4B 输出 32 层 / 738 tensor / 8.41GB 文本布局
- `ModelBuilder`：生成完整 prefill 计算图，包含 embedding、32 层、final norm、lm_head，以及 4 个入口函数
- `llmcp-dump`：把完整计算图 dump 成 `.mlir`，支持 `--verify-module` 和 `--print-stats`
- `llmcp-lower-to-linalg`：已覆盖 `linear`、`rms_norm`、`add`、`silu_mul`、`embedding`、`gated_rms_norm`、`lm_head`、`causal_conv1d`；`attention`/`gated_delta_net` 仍待分解
- 测试：10 个 lit 用例 + 14 个 gtest 用例 + 8 类已 lower op 的数值测试（对 PyTorch 参考）+ 真实模型 dump/lowering 集成测试，全部通过

仍在 M0/M1 之间：tokenizer、HF golden 导出、`llmcp → linalg` lowering 尚未开始。

## 构建与测试

```bash
# 0. 拉取 vendored 第三方源码
git submodule update --init --recursive

# 1. 构建（C/C++ 都用 Homebrew clang 20）
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/llvm
cmake --build build -j 8

# 2. 测试
ctest --test-dir build --output-on-failure

# 3. 检查真实模型配置与权重布局（只读 header）
./build/bin/llmcp-inspect --model /Users/guorui/Work/models/qwen3.5-4b

# 4. 解析/打印 llmcp IR，并测试 lowering
./build/bin/llmcp-opt tests/lit/Dialect/LLMCP/ops.mlir
./build/bin/llmcp-opt tests/lit/Conversion/lower-linear.mlir --llmcp-lower-to-linalg

# 5. dump 完整计算图
./build/bin/llmcp-dump --model /Users/guorui/Work/models/qwen3.5-4b \
  --verify-module --print-stats -o build/qwen35_graph.mlir

# 6. 数值测试（linear / rms_norm / add / silu_mul / embedding /
#    gated_rms_norm / lm_head / causal_conv1d，对 PyTorch 参考）
ctest --test-dir build -R llmcp-numeric --output-on-failure
```

当前 dump 的是 **prefill 图**（32 层展开）。`--print-stats` 输出：

```text
functions: 4
linear: 248, rms_norm: 65, delta: 24, attention: 8, conv: 24,
gated_norm: 24, embedding: 1, lm_head: 1, silu_mul: 32, add: 64,
weight_views: 427
```

`llmcp-inspect` 当前输出：

```text
layers: 32 (24 linear, 8 full)
tensors: 738
checkpoint bytes: 9319737856
arena bytes: 8411510336 (header 64, group stride 892513344, groups 8)
```

文本 arena 为 8.41GB；视觉塔和 MTP 尚未纳入文本编译布局。

## 后续（规划中）

```bash
# 打包权重
llmcp-pack --model /Users/guorui/Work/models/qwen3.5-4b --out build/weights.bin

# 编译模型
llmcp-compile --model /Users/guorui/Work/models/qwen3.5-4b \
  --weights build/weights.bin --emit=shared --out build/libqwen35.dylib

# 运行
llmcp-run --lib build/libqwen35.dylib --prompt "你好" --max-new-tokens 64
```
