# 算子并行化

## 1. 方案

当前使用 OpenMP 作为 CPU 并行后端：

```text
llmcp.linear / custom ops
  -> linalg / scf
  -> scf.parallel
  -> omp.parallel + omp.wsloop
  -> LLVM IR
  -> libomp
```

新增 pass：

```text
llmcp-parallelize
```

它把标记为 `llmcp.parallel` 的 `scf.for` 转成 `scf.parallel`。

## 2. 各算子并行方式

| 算子 | 并行方式 |
|---|---|
| `linear` | bfdot 输出行循环 |
| `lm_head` | 输出行循环 |
| `add` | linalg 的 parallel 维度 |
| `silu_mul` | linalg 的 parallel 维度 |
| `embedding` | linalg 的 parallel 维度 |
| `rms_norm` | linalg 的 parallel 维度 |
| `gated_rms_norm` | linalg 的 parallel 维度 |
| `causal_conv1d` | time 维循环 |
| `gated_delta_net` | 每个 value head 并行 |
| `attention` | Q head / KV head / query head 并行 |

时间维、递归状态更新和 attention score 的依赖关系仍然串行。

## 3. 依赖处理

attention 和 delta 的 scratch buffer 改成按 head 分离：

```text
attention:
  scores    [numHeads, cacheLength]
  normTemp [numHeads, headDim]

delta:
  qn    [numVHeads, headK]
  kn    [numVHeads, headV]
  delta [numVHeads, headV]
```

这样 head 循环之间没有共享 scratch 写冲突。

## 4. 线程数

OpenMP 默认使用全部逻辑核，可以用：

```bash
OMP_NUM_THREADS=8 ./build/bin/llmcp-run ...
```

## 5. Pipeline

`llmcp-compile` 现在执行：

```text
--llmcp-lower-to-linalg
--llmcp-parallelize
--convert-linalg-to-parallel-loops
--convert-scf-to-openmp
--convert-openmp-to-llvm
...
```

AOT dylib 链接：

```text
libomp
libmlir_runner_utils
libmlir_c_runner_utils
```

## 6. 验证

- `llmcp-parallelize` lit test：检查 `scf.parallel` / `scf.reduce`
- numeric：全部通过
- AOT：单 token `next 11`，greedy `0 271 760 2614`
