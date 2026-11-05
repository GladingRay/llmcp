# Qwen3.5-4B 中的 matmul shape 与 AOT 性能

> 注：下表是未启用 OpenMP 的单线程基线。当前工具默认并行，
> 可用 `--threads N` / `--no-parallel` 对比。最新结果见
> [parallelization.md](parallelization.md)。

## 1. 统计口径

这里统计的是完整图中由 `llmcp.linear` / `llmcp.lm_head` 表示的矩阵乘：

```text
C[M, N] = A[M, K] * B[N, K]^T
```

- `M`：当前序列长度。prefill 时为 token 数，decode 时为 1；
- `N`：输出特征数；
- `K`：输入特征数；
- A：`[M, K]`，行主序；
- B：`[N, K]`，行主序；
- `K` 维连续；
- 所有模型权重都走 `transpose_b = false`，即 `A * B^T`。

每个 forward 图（prefill/decode）中有 **248 个 `llmcp.linear`**，加 **1 个 `llmcp.lm_head`**；AOT 模块包含两个 forward 副本。

## 2. Shape 与出现次数

| 次数 | N | K | 作用 |
|---:|---:|---:|---|
| 32 | 8192 | 2560 | 24 个 linear layer 的 fused QKV；8 个 full layer 的 q_proj（含 output gate） |
| 24 | 4096 | 2560 | linear layer 的 `in_proj_z` |
| 48 | 32 | 2560 | 24 个 `in_proj_a` + 24 个 `in_proj_b` |
| 64 | 9216 | 2560 | 32 层 MLP 的 gate/up，每层两个 |
| 32 | 2560 | 9216 | 32 层 MLP 的 down |
| 32 | 2560 | 4096 | 24 个 linear `out_proj` + 8 个 full `o_proj` |
| 16 | 1024 | 2560 | 8 个 full layer 的 k_proj / v_proj |
| 1 | 248320 | 2560 | lm_head |

线性层数量：

```text
32 + 24 + 48 + 64 + 32 + 32 + 16 = 248 linear
1 lm_head
```

按 `2 * M * N * K` 计算，M=1 时这些 matmul 合计：

```text
约 8.41 GFLOP / token
```

## 3. B 的 layout

所有 linear/lm_head 的 B 都是：

```text
[N, K] row-major
```

例如：

```text
gate_proj: [9216, 2560]
down_proj: [2560, 9216]
o_proj:    [2560, 4096]
lm_head:   [248320, 2560]
```

因此 K 维连续，和 `matmul_transpose_b` 的：

```text
C[m, n] = sum_k A[m, k] * B[n, k]
```

一致。

## 4. Benchmark 方法

使用：

```bash
./build/bin/llmcp-matmul-benchmark \
  --m <M> --n <N> --k <K> --iters 5
```

每次运行会：

```text
llmcp.linear
  -> linalg
  -> vector / scf
  -> LLVM IR
  -> dylib
  -> dlopen
  -> 计时
```

当前 AOT 代码在 arm64 上使用 `bfdot`。

测量值：

```text
GFLOPS = 2 * M * N * K * iters / seconds / 1e9
```

输入为全 1 bf16，包含一次 warmup 和输出 copy 开销。

## 5. 实测结果

机器：Apple M2，8 逻辑核，LLVM 20，单线程 AOT 标量调用。

| N | K | M=1 GFLOPS | M=1 秒 | M=16 GFLOPS | M=16 秒 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 2560 | 9.4039 | 0.0223 | 20.1976 | 0.1661 |
| 4096 | 2560 | 8.0034 | 0.0131 | 16.9068 | 0.0992 |
| 32 | 2560 | 4.7092 | 0.0002 | 5.6592 | 0.0023 |
| 2560 | 4096 | 8.4968 | 0.0123 | 16.5128 | 0.1016 |
| 9216 | 2560 | 8.7670 | 0.0269 | 20.4339 | 0.1847 |
| 2560 | 9216 | 8.3190 | 0.0284 | 18.5472 | 0.2035 |
| 1024 | 2560 | 5.6778 | 0.0046 | 9.4743 | 0.0443 |
| 248320 | 2560 | 20.7022 | 0.3071 | 20.4212 | 4.9807 |

按上表顺序累计：

```text
M=1:  total 0.4149 s, 约 20.27 GFLOPS
M=16: total 5.7824 s, 约 23.27 GFLOPS
```

注意：这是把各 shape 单独跑的时间直接相加，不是完整模型端到端时间；
attention、DeltaNet、RMSNorm、embedding、输出 copy 等未计入。

## 6. 观察

1. `M=1` 时大多数 shape 在 8-9 GFLOPS，主要受 K 循环和 reduction 限制。
2. `M=16` 时多数 shape 提升到 16-20 GFLOPS，说明多行复用权重有一定收益。
3. `N=32` 的两个小投影只有 4.7-5.7 GFLOPS，固定开销占比高。
4. `lm_head` 虽然 N 很大，M=1 时达到 20.7 GFLOPS；M=16 时受 248320 输出和内存流量限制，约 20.4 GFLOPS。
5. MLP 的 gate/up 与 qkv 是总算力占比最大的部分。

## 7. 未统计的 matmul

以下不是 `linalg.matmul`/`llmcp.linear`，未纳入本表：

- full attention 的 QK^T 和 attention * V；
- DeltaNet 内部的递归/外积；
- causal conv1d；
- gated RMSNorm；
- lm_head 之外的 embedding。

这些算子目前由自定义 lowering 生成循环，不是独立的 named matmul。


## 7. Q8_0 matmul（2026-11-01）

`llmcp-matmul-benchmark --q8` 走：

```text
Q8_0 权重
+ activation 预量化
+ ARM sdot
+ f32 scale 累加
```

Apple M2，4 线程，全 1 输入：

```text
256x256x256:
  bf16: 29.9 GFLOPS
  q8:   32.7 GFLOPS

M=16, N=8192, K=2560:
  bf16: 55.9 GFLOPS
  q8:   189.5 GFLOPS
```

Q8 在大 N/K 下收益明显，主要来自权重流量减半和 sdot 吞吐。
当前 bf16 路径仍使用 `bfdot`，Q8 路径使用 `sdot`。
