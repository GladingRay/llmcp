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

设计阶段。仓库中的 `include/`、`src/`、`tests/` 为空；下一步从 M0（构建骨架 + 模型解析 + 参考数据）开始。

## 快速开始（规划）

```bash
# 0. 拉取 vendored 第三方源码
git submodule update --init --recursive

# 1. 构建（C/C++ 都用 Homebrew clang 20）
cmake -G Ninja -S . -B build \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/llvm
ninja -C build

# 2. 打包权重（规划中的工具）
llmcp-pack --model /Users/guorui/Work/models/qwen3.5-4b --out build/weights.bin

# 3. 编译模型
llmcp-compile --model /Users/guorui/Work/models/qwen3.5-4b \
  --weights build/weights.bin --emit=shared --out build/libqwen35.dylib

# 4. 运行
llmcp-run --lib build/libqwen35.dylib --prompt "你好" --max-new-tokens 64
```
