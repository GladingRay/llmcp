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
| `linear` | bfdot 输出列（N）循环 |
| `lm_head` | 输出列（N）循环 |
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

## 6. 并行维度选择

`linear` / `lm_head` 并行 **N 维（输出列）**，而不是 M 维（sequence）：

```text
prefill: M 小、N 大（2560 / 8192 / 248320）
decode:  M=1、N 仍然很大
```

按 N 并行可以覆盖 decode 和 prefill，并且避免 M=1 时没有并行度。

## 7. 验证

- `llmcp-parallelize` lit test：检查 `scf.parallel` / `scf.reduce`
- numeric：全部通过
- AOT：单 token `next 11`，greedy `0 271 760 2614`

## 8. 性能

AOT dylib，`4 prefill / 2 decode`：

```text
OMP_NUM_THREADS=1:
  prefill: 2.84 s
  decode:  1.00 s

OMP_NUM_THREADS=4:
  prefill: 0.79 s
  decode:  0.35 s

OMP_NUM_THREADS=8:
  prefill: 0.86 s
  decode:  0.41 s
```

按 N 维并行后，decode 也有明显收益，不再出现按 M 行并行时 M=1 无并行度的问题。
