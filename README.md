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
- `llmcp` 方言：`llmcp.linear`、`llmcp.rms_norm`，含 TableGen 定义与 C++ verifier
- `ModelConfig`：解析并校验 `config.json`（层类型、GQA、RoPE、mRoPE、线性注意力头数等）
- `Safetensors`：只读 header，不加载 tensor 数据
- `WeightLayout`：文本权重 arena 的确定性偏移、64B 对齐、group stride
- `llmcp-inspect`：对真实 Qwen3.5-4B 输出 32 层 / 738 tensor / 8.41GB 文本布局
- 测试：3 个 lit 用例 + 3 组 C++ 单测（ModelConfig / Safetensors / WeightLayout）全部通过

仍在 M0/M1 之间：tokenizer、HF golden 导出、单层 lowering 尚未开始。

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

# 4. 解析/打印 llmcp IR
./build/bin/llmcp-opt tests/lit/Dialect/LLMCP/ops.mlir
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
