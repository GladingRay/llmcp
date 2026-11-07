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


## 11. 非 BF16 runner 的处理

并非所有 Apple Silicon runner 都支持 BF16 dot product。例如部分 GitHub M1 runner
没有 `hw.optional.arm.FEAT_BF16`，如果仍然生成：

```text
llvm.aarch64.neon.bfdot
```

LLVM 会报：

```text
fatal error: error in backend: Cannot select: intrinsic %llvm.aarch64.neon.bfdot
```

根仓库在 configure 时检测：

```cmake
sysctl -n hw.optional.arm.FEAT_BF16
```

并控制：

```cmake
LLMCP_ENABLE_BFDOT
```

行为：

```text
BF16=1:
  生成 bfdot 路径
  运行 bfdot lit 测试

BF16=0:
  使用 vector/f32 fallback
  跳过 bfdot lit 测试
  numeric/unit/matmul 测试仍然运行
```

手动关闭：

```bash
cmake -S . -B build -G Ninja -DLLMCP_ENABLE_BFDOT=OFF
```

手动开启只应在确认 runner 和编译器都支持 BF16 时使用：

```bash
cmake -S . -B build -G Ninja -DLLMCP_ENABLE_BFDOT=ON
```

`microbench` 也改为 `-mcpu=native`，避免在 M1 上编译仅 M2 可执行的 BF16
指令。
