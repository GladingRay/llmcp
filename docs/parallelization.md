# 算子并行化

> 状态（2026-11-01）：OpenMP 并行、N 维 matmul 并行、RMSNorm 合并、
> elementwise 融合均已落地；Q8 路径使用 sdot。本文的性能数字来自早期
> 单机测量，绝对值会受 page cache / 内存压力影响。

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
| `rms_norm` | 每个 row 一个 parallel region，内部完成 reduction/rstd/scale |
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
--linalg-fuse-elementwise-ops
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

## 8. elementwise 融合

在 bufferization 之前加入：

```text
--linalg-fuse-elementwise-ops
```

融合范围：

- add；
- silu_mul；
- bf16/f32 cast；
- gated RMSNorm 内部的 elementwise 链；
- 其他 linalg elementwise producer-consumer。

当前完整图中 `linalg.generic` 数量：

```text
融合前: 217
融合后: 169
```

减少的是可以安全融合的 elementwise 链，reduction/matmul/自定义 scf 循环不受影响。

## 9. matmul microbenchmark

`llmcp-matmul-benchmark` 已接入同一套并行化 pipeline：

```bash
./build/bin/llmcp-matmul-benchmark \
  --m 256 --n 256 --k 256 --iters 10 --threads 4
```

实测：

```text
256x256x256:
  --threads 4:   41.9 GFLOPS
  --threads 1:   14.7 GFLOPS
  --no-parallel: 11.7 GFLOPS

M=16, N=8192, K=2560:
  --threads 4:   58.0 GFLOPS
  --no-parallel: 20.8 GFLOPS
```

## 10. 性能

AOT dylib，`4 prefill / 2 decode`：

```text
OMP_NUM_THREADS=1:
  prefill: 6.21 s
  decode:  1.58 s

OMP_NUM_THREADS=4:
  prefill: 0.82 s
  decode:  0.43 s

OMP_NUM_THREADS=8:
  prefill: 0.80 s
  decode:  0.44 s
```

按 N 维并行后，decode 也有明显收益，不再出现按 M 行并行时 M=1 无并行度的问题。


### elementwise 融合后

8 线程，32-token decode：

```text
prefill: 约 0.37 s
decode:  约 6.4-6.6 s
```

融合前同条件约 7.0-7.3 s，约有 5-10% 改善。


## 11. Q8 路径

Q8 matmul 也接入同一套并行化 pipeline：

```text
activation 预量化 row loop -> llmcp.parallel
Q8 dot 的 N 维 loop     -> llmcp.parallel
```

Q8 使用：

```text
llvm.aarch64.neon.sdot
```

而不是 bf16 的 `bfdot`。

Apple M2，4 线程，全 1 输入：

```text
M=16, N=8192, K=2560:
  bf16/bfdot: 约 55.9 GFLOPS
  q8/sdot:    约 189.5 GFLOPS
```

完整模型 Q8 结果和精度见 [int8-quantization.md](int8-quantization.md)。
