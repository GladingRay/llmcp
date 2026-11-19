# Qwen3.5-4B 中的 matmul shape 与 AOT 性能

> 注：本文的逐 shape 表是 2026-10 的未启用 OpenMP 历史基线，保留用于分析
> shape 分布。当前工具默认并行。BF16 和 Q8 的历史性能见 [README](../README.md#性能)
> 和 [parallelization.md](parallelization.md)。1024 方阵的 Q8 SDOT 数据见本文末节。

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

AOT 代码在支持 BF16 的 arm64 上使用 `bfdot`；不支持时回退 vector/f32。
Q8 使用 `sdot`，无 DotProd 时回退 scalar。

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

## 8. Q8_0 SDOT 的 1024 方阵

Prefill 采用 BLIS 的缓存面板与寄存器分块，并参考 OpenBLAS Arm 内核按 K 的取数顺序
交错打包。激活量化后直接写入 A 微面板。每个 OpenMP 任务打包并复用一个 64 列的 B
面板，任务内遍历 32 行的 M 面板和 4×8 寄存器 tile。B 每组四列的四字节权重排列在同一个
128 位向量中，lane SDOT 的四个输出 lane 各对应一列。每 32 个 K 元素完成一次
整数点积和融合浮点累加，FP32 输出累加器保持在寄存器中直到 K 结束。
[BLIS GEMM 分块](https://www.cs.utexas.edu/~flame/pubs/blis3_ipdps14.pdf)
· [OpenBLAS Arm 交错打包](https://github.com/OpenMathLib/OpenBLAS/pull/6145)

打包内核用于支持 SDOT、M≥16、N≥64、N 为 8 的倍数且权重 K 维连续的矩阵。
动态 M 在运行时分派。权重仍使用现有 Q8_0 布局，逐 32 元素的 scale 相乘后用 FMA
累加，相比乘法和加法分别舍入的路径可以有末位舍入差异。

使用 `2 × M × N × K / 秒数` 计算等效 GOPS。GEMM 计时包括动态激活量化、权重
面板打包、缩放、累加和输出 copy。指令基准预热后执行 16 条独立的 SDOT 累加链，
每条 SDOT 计 32 次整数运算。按相同线程数采样并比较两者的中位数。

```bash
./build/bin/llmcp-matmul-benchmark --q8 -m 1024 -n 1024 -k 1024 --iters 80 --no-parallel
./build/bin/llmcp-matmul-benchmark --q8 -m 1024 -n 1024 -k 1024 --iters 80 --threads 4
./build/bin/llmcp-matmul-benchmark --q8 -m 1024 -n 1024 -k 1024 --iters 80 --threads 8
./build/bin/llmcp-microbench --sdot-threads 1 --seconds 0.5
./build/bin/llmcp-microbench --sdot-threads 4 --seconds 0.5
./build/bin/llmcp-microbench --sdot-threads 8 --seconds 0.5
```

Apple M2、LLVM 20.1.8，三次独立运行的中位数。基线使用 `c6158b7` 的
`llmcp-opt`，两组 GEMM 共用带缓冲区释放的 benchmark 计时流程。

| 线程 | 基线 GOPS | 打包内核 GOPS | SDOT 峰值 GOPS | 加速 | 峰值占比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 131.6 | 288.3 | 434.1 | 2.19× | 66.4% |
| 4 | 453.9 | 921.6 | 1609.9 | 2.03× | 57.2% |
| 8 | 491.6 | 1072.8 | 2171.9 | 2.18× | 49.4% |

对每个四列向量和 32 个 K 元素，内核发出 8 条 SDOT，以及转换整数点积的
`SCVTF`、相乘激活与权重尺度的 `FMUL` 和累加结果的 `FMLA`。如果四种指令
共享 SIMD 发射端口，仅用这个指令组合估算的吞吐上限为 `8 / 11 ≈ 72.7%`。
这个估算沿用 Firestorm 的端口实测，另外还有打包、加载和写回开销。
[Firestorm SIMD 指令测量](https://dougallj.github.io/applecpu/firestorm-simd.html)

在 M2 上另用寄存器循环做了指令混合对照。每轮执行 64 条 lane SDOT，以及 8 条
`SCVTF`、8 条 `FMUL` 和 8 条 `FMLA`，循环内无访存，只计 SDOT 运算量。
单线程三次交替测量的中位数为 288.3 GOPS，约为同轮纯 lane SDOT 的 65.6%
（439.8 GOPS）。这是该指令调度的实测值。

## 9. Q8 SDOT decode GEMV

2026-11-19 的 decode 内核使用 4 列分块。激活沿 K 每 32 个元素向量量化一次，
量化循环串行执行。四列的整数点积通过三条 `ADDP` 合成四个输出 lane，然后向量化
执行 scale 相乘、点积转 FP32 和累加。每列的 K 块累加顺序保留。FP16 权重 scale
使用允许非对齐访问的原生 half 加载，地址包含 memref 的 offset 和行 stride。

输出列循环在 N≥64 且 N×K≥262144 时使用 OpenMP，短尾列和小矩阵串行执行。
权重沿 K 连续时启用此内核，其他布局走原有 lowering。每次调用共用原始 Q8_0
权重和一次激活量化的结果。

基准需显式传 `--decode`，要求 M=1。未传该选项时使用 prefill lowering。

```bash
./build/bin/llmcp-matmul-benchmark --q8 --decode -m 1 -n 8192 -k 2560 --iters 400 --threads 8
./build/bin/llmcp-matmul-benchmark --q8 --decode -m 1 -n 32 -k 2560 --iters 3000 --threads 8
```

Apple M2、LLVM 20.1.8、8 线程，基线为 `16acef3` 的 decode lowering。
每组交替测三次，表中为单次 GEMV 耗时中位数，包含量化、临时缓冲区分配和释放、
输出拷贝。每次内部先预热一次，N=32 计时 3000 次，其余计时 400 次。
测量期间重复访问同一份权重。完整命令和样本见
[测量记录](benchmarks/2026-11-19-q8-decode.json)。

| N | K | 基线 μs | 优化后 μs | 加速 |
|---:|---:|---:|---:|---:|
| 32 | 2560 | 99.07 | 4.62 | 21.45× |
| 1024 | 2560 | 145.39 | 91.31 | 1.59× |
| 4096 | 2560 | 207.60 | 136.56 | 1.52× |
| 8192 | 2560 | 336.07 | 255.43 | 1.32× |
| 9216 | 2560 | 410.06 | 325.77 | 1.26× |
| 2560 | 4096 | 207.62 | 142.56 | 1.46× |
| 2560 | 9216 | 451.28 | 312.12 | 1.45× |

完整 Qwen3.5-4B 模型以 8 线程、32-token prefill 和 32-token decode 做交替复测，
两种 AOT 各预热一次，再测三次。decode 中位数从 8.2288 提升到 8.4932 tok/s，
增幅为 3.2%。该测量还包括 BF16 LM head、attention、DeltaNet 和运行时开销。
