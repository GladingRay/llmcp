# GitHub CI/CD 方案

## 1. 目标

为 `llmcp` 建立：

```text
push / PR:
  macOS arm64 构建
  + CTest
  + artifact 日志

tag:
  构建工具链
  + macOS arm64 release 包
```

## 2. 关键约束

### 必须使用 Apple Silicon

项目的主路径依赖：

```text
bfdot
sdot
-mcpu=apple-m2
OpenMP
```

因此 CI 使用：

```yaml
runs-on: macos-15
```

并检查：

```bash
test "$(uname -m)" = arm64
```

GitHub 的 `macos-14` 镜像已于 2026-11-02 退役；当前使用 `macos-15`（arm64 标签）。
如果 `macos-15` 不是 arm64，改成自托管 runner：

```yaml
runs-on: [self-hosted, macOS, ARM64]
```

### 模型不进入 CI

4B 权重和 arena 不提交到仓库，因此 CI：

- 不跑真实模型测试；
- 不跑 golden token；
- 不打包权重；
- 通过 `LLMCP_MODEL_DIR=/tmp/llmcp-ci-no-model` 自动跳过真实模型 CTest。

CI 仍然会运行：

```text
llmcp-lit
llmcp-numeric
llmcp-microbench
llmcp-matmul-benchmark
llmcp-matmul-benchmark-q8
llmcp-unittests
```

## 3. CI 流程

文件：

```text
.github/workflows/ci.yml
```

步骤：

```text
checkout
  -> brew install llvm@20 cmake ninja libomp
  -> setup-python
  -> pip install torch numpy lit
  -> rust toolchain
  -> cmake configure
  -> cmake --build
  -> ctest
  -> upload ctest logs
```

关键 CMake 参数：

```bash
-DCMAKE_C_COMPILER=$LLVM_PREFIX/bin/clang
-DCMAKE_CXX_COMPILER=$LLVM_PREFIX/bin/clang++
-DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir
-DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm
-DLLMCP_BUILD_TESTS=ON
-DLLMCP_CARGO_OFFLINE=OFF
-DLLMCP_MODEL_DIR=/tmp/llmcp-ci-no-model
```

`LLMCP_CARGO_OFFLINE=OFF` 是必需的：CI 上没有本机 cargo cache，需要联网构建
`tokenizers-cpp` 的 Rust 依赖。

## 4. CD 流程

文件：

```text
.github/workflows/release.yml
```

触发：

```text
tag v*
或 workflow_dispatch
```

产物：

```text
llmcp-macos-arm64-<tag>.tar.gz
```

内容：

```text
build/bin/llmcp-*
```

不包含：

- 模型权重；
- AOT dylib；
- 用户生成的 arena。

AOT 仍然由用户在本机执行：

```bash
llmcp-pack ...
llmcp-compile ...
```

## 5. 真实模型回归

默认 CI 不下载 4B 模型。

如果需要真实模型回归，建议使用：

```text
self-hosted Apple Silicon runner
+ 预先放置 /Users/guorui/Work/models/qwen3.5-4b
+ 或者在 workflow 中从对象存储下载
```

注意：

- 不要把模型放进 Git；
- 不要把 arena 放进 Git；
- 可以通过 GitHub Secrets 保存对象存储凭证；
- 真实模型测试应作为夜间/手动 workflow，而不是每个 PR 的必跑项。

## 6. 缓存建议

适合缓存：

```text
Homebrew 下载缓存
cargo registry
ccache
pip cache
```

不建议直接缓存 `build/`，因为 MLIR 版本、编译选项和模型配置变化会导致
不可预期的 stale object。

可以后续增加：

```yaml
- uses: hendrikmuhs/ccache-action@v1
```

并在 CMake 中启用 `ccache`。

## 7. 分支保护

在 GitHub 仓库 Settings -> Branches 中保护 `main`：

```text
Require status checks to pass before merging
  build-and-test
Require branches to be up to date
Require conversation resolution
```

## 8. 当前工作流的限制

- `macos-15` 是当前 GitHub 的 arm64 标签；若账号/仓库不支持，使用自托管 runner；
- 不在 CI 中执行 4B 模型；
- 不在 CI 中生成 AOT dylib；
- 没有跨平台 x86_64 CI；x86 路径需要另外的条件测试；
- 首次运行会安装 Homebrew/PyTorch，耗时较长；后续可加 cache。


## 11. 加速指令开关

### 11.1 BF16 dot

部分 Apple Silicon runner 没有：

```text
hw.optional.arm.FEAT_BF16
```

如果仍然生成：

```text
llvm.aarch64.neon.bfdot
```

LLVM 会报：

```text
fatal error: error in backend: Cannot select: intrinsic %llvm.aarch64.neon.bfdot
```

configure 时同时检测硬件和编译器后端：

```cmake
sysctl -n hw.optional.arm.FEAT_BF16
check_cxx_source_compiles(... -mcpu=native ...)
```

并控制：

```cmake
LLMCP_ENABLE_BFDOT
```

手动关闭：

```bash
cmake -S . -B build -G Ninja -DLLMCP_ENABLE_BFDOT=OFF
```

BF16 不可用时：

```text
使用 vector/f32 fallback
跳过 bfdot lit 测试
numeric/unit/matmul 测试继续运行
```

### 11.2 Q8 sdot

Q8 的加速指令同样需要检测：

```text
hw.optional.arm.FEAT_DotProd
```

configure 时检测：

```cmake
sysctl -n hw.optional.arm.FEAT_DotProd
check_cxx_source_compiles(... -mcpu=native ...)
```

并控制：

```cmake
LLMCP_ENABLE_SDOT
```

手动关闭：

```bash
cmake -S . -B build -G Ninja -DLLMCP_ENABLE_SDOT=OFF
```

sdot 不可用时：

```text
emitQ8Matmul 使用 scalar fallback
emitPrefillQ8Blocked 使用 scalar fallback
microbench 不注册 neon_i8_dot
numeric Q8 测试继续验证正确性
```

当前 `LLMCP_ENABLE_BFDOT` 与 `LLMCP_ENABLE_SDOT` 相互独立：

```text
BF16=0, DotProd=1:
  bf16 走 fallback
  Q8 继续使用 sdot

BF16=0, DotProd=0:
  bf16 走 fallback
  Q8 走 scalar fallback
```

`microbench` 使用 `-mcpu=native`，避免在旧 CPU 上编译只在新 CPU 上可执行的指令。


### 11.3 生成代码的编译选项

`llmcp-compile`、`llmcp-matmul-benchmark`、numeric 测试在编译生成的 LLVM IR 时，
会根据启用的特性显式使用：

```text
-march=armv8.2-a+dotprod
-march=armv8.6-a+bf16
-march=armv8.6-a+bf16+dotprod
```

而不是只依赖 `-mcpu=native`，避免编译器无法从 native CPU 推导出 dotprod/bf16
feature 时出现：

```text
Cannot select: AArch64ISD::SDOT
Cannot select: intrinsic %llvm.aarch64.neon.bfdot
```

'
