# INT8 量化技术方案

## 1. 目标

为 Qwen3.5-4B 的 CPU AOT 推理增加 INT8 量化路径，参考 llama.cpp 的
`Q8_0` / `Q8_1` 方案：

- 权重使用 `Q8_0`；
- 激活在运行时量化为 `Q8_1`；
- 矩阵乘使用 int8 点积 + f32 scale；
- 量化后模型精度接近 bf16；
- 在 Apple M2 上使用 `sdot` 指令加速。

首期范围：

```text
llmcp.linear
llmcp.lm_head
embedding 查表（可选）
```

暂不包含：

- KV cache 量化；
- attention QK/V 的自定义矩阵乘；
- DeltaNet 递归 state 量化；
- 2/4/6 bit K-quants；
- 训练后校准。

## 2. llama.cpp 方案摘要

llama.cpp 的 Q8 量化是 **block-wise symmetric quantization**。

### 2.1 Q8_0

每个 block 包含 32 个权重：

```cpp
struct block_q8_0 {
  ggml_fp16_t d;      // scale, fp16
  int8_t qs[32];      // 32 个 int8
};
```

大小：

```text
2 + 32 = 34 bytes / 32 weights
= 8.5 bits / weight
```

量化：

```text
d = max(abs(x[0..31])) / 127
q[i] = round(x[i] / d)
q[i] = clamp(q[i], -127, 127)
```

反量化：

```text
x_hat[i] = d * q[i]
```

特点：

- 对称量化，无 zero point；
- 每个 block 独立 scale；
- 权重和激活都可使用相同 block 结构；
- `Q8_0` 通常作为权重格式，精度较高。

### 2.2 Q8_1

激活量化使用：

```cpp
struct block_q8_1 {
  float d;            // scale
  float s;            // sum(q) * d，兼容 llama.cpp 的 dot 实现
  int8_t qs[32];
};
```

大小：

```text
4 + 4 + 32 = 40 bytes / 32 values
```

量化：

```text
d = max(abs(x)) / 127
q[i] = round(x[i] / d)

sum_q = sum(q[i])
s = sum_q * d
```

`Q8_1` 主要用于 activation，和 `Q8_0` 权重的 dot 可以共用整数点积。

### 2.3 Q8_K（后续可选）

llama.cpp 还有 `Q8_K`：

```text
super-block = 256
int8 qs[256]
float d
int16 bsums[16]
```

它通常作为 K-quants 的中间 activation 格式；首期不需要。

### 2.4 点积

对每个 block：

```text
dot_int = sum(q_a[i] * q_b[i])       // int32
result += dot_int * d_a * d_b        // f32
```

注意：

- `dot_int` 用 int32 累加；
- 跨 block 用 f32 累加；
- 不需要把 int8 反量化回 f32 再做乘法。

## 3. 本项目的量化对象

当前模型的主要矩阵乘：

```text
N=8192,  K=2560   x 32
N=4096,  K=2560   x 24
N=32,    K=2560   x 48
N=9216,  K=2560   x 64
N=2560,  K=9216   x 32
N=2560,  K=4096   x 32
N=1024,  K=2560   x 16
N=248320,K=2560   x 1
```

所有 K 都是 32 的倍数：

```text
2560, 4096, 8192, 9216, 1024, 32
```

因此可以直接使用 32-element block，不需要处理 K 尾部。

首期量化：

| tensor | 格式 |
|---|---|
| `self_attn.q/k/v/o_proj` | Q8_0 |
| `linear_attn.in_proj_*` | Q8_0 |
| `linear_attn.out_proj` | Q8_0 |
| `mlp.gate/up/down_proj` | Q8_0 |
| `embed_tokens` / `lm_head` | Q8_0，可选 |
| RMSNorm / q_norm / k_norm | 保持 bf16/f32 |
| `A_log` / `dt_bias` | 保持 f32/bf16 |
| conv1d 权重 | 保持 bf16 |

Norm 类参数很小，量化收益低、精度风险高，不纳入首期。

## 4. 权重布局

### 4.1 Q8_0 存储

使用 llama.cpp 的紧凑存储：

```cpp
struct alignas(1) Q8_0Block {
  uint16_t d;        // fp16 bits
  int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34);
```

B 的形状仍然是：

```text
B: [N, K] row-major
```

但每个 row 按 K 切成：

```text
K / 32 个 block
```

一个 B row 的字节数：

```text
(N 维不变) row_stride = (K / 32) * 34
```

例如：

```text
N=9216, K=2560:
  row_stride = 80 * 34 = 2720 bytes

N=248320, K=2560:
  row_stride = 80 * 34 = 2720 bytes
```

### 4.2 A 的激活布局

输入 A：

```text
A: [M, K] bf16
```

量化后：

```text
Aq: [M, K/32] Q8_1 block
```

每个 block：

```text
d: f32
s: f32（可选，首期只用于兼容）
qs[32]: int8
```

为了 AOT kernel 访问方便，可以用 SoA：

```text
scales: f32[M][K/32]
qs:     int8[M][K]
```

这样：

- scale 连续；
- qs 连续；
- 避免每 32 个 int8 读一个 4/8 字节 header；
- 更适合 NEON load。

### 4.3 对齐

当前 arena 64 字节对齐。Q8_0 block 是 34 字节，不保证每个 block 对齐。

两种选择：

1. **llama.cpp 兼容布局**：34 字节 block，load 用非对齐指令；
2. **AOT 运行时布局**：加载后重排为

```text
scales[N][K/32] : f16/f32
qs[N][K]        : int8
```

首期建议：

- 磁盘/manifest 保存 llama.cpp 兼容的 34 字节 block；
- 加载后按 kernel 需要重排为 SoA；
- 重排只做一次，避免 hot loop 非对齐读取。

## 5. Arena 与 manifest

`llmcp-pack` 新增：

```bash
llmcp-pack --model ... --quant q8_0 --out build/weights-q8.bin
```

manifest 增加：

```json
{
  "quant": "q8_0",
  "block_size": 32,
  "scale_dtype": "f16",
  "tensors": [
    {
      "name": "model.language_model.layers.0.mlp.gate_proj.weight",
      "dtype": "Q8_0",
      "shape": [9216, 2560],
      "offset": 123,
      "size": 7833600
    }
  ]
}
```

计算：

```text
size = N * (K / 32) * 34
```

例如 `gate_proj`：

```text
9216 * 80 * 34 = 25,067,520 bytes
```

bf16 原大小：

```text
9216 * 2560 * 2 = 47,185,920 bytes
```

约为 53%。

整个 8.41 GB arena 的 Q8_0 权重约：

```text
4.5 GB 左右
```

## 6. Dialect / IR 设计

### 6.1 扩展 LinearOp

不新增一套完整 op，先给现有 `llmcp.linear` / `llmcp.lm_head` 增加量化属性：

```mlir
%0 = llmcp.linear %x, %w_q8
    {weight_format = "q8_0"}
    : tensor<?x2560xbf16>, memref<?xi8> -> tensor<?x9216xbf16>
```

verifier 检查：

```text
weight byte size == N * (K / 32) * 34
```

`transpose_b = true` 的首期不支持，直接 fallback bf16。

### 6.2 新增内部 op

为了 lowering 清晰，可以新增：

```mlir
llmcp.quantize_q8_1 %x
  : tensor<?xKxbf16> -> !llmcp.q8_1_tensor

llmcp.q8_matmul %xq, %wq
  : !llmcp.q8_1_tensor, memref<?xi8> -> tensor<?xNxbf16>
```

但首期为了减少方言改动，可以直接在 `LinearOpLowering` 内生成：

```text
quantize activation
  -> q8 dot loop
  -> f32 scale
  -> bf16 output
```

## 7. Lowering 设计

### 7.1 激活量化

对每个 row `m`、block `b`：

```text
amax = max(abs(A[m, 32b..32b+31]))
d = amax / 127
q[i] = round(A[m, 32b+i] / d)
```

实现：

- `linalg.generic` 计算 block max；
- `linalg.generic` 量化到 int8；
- 或一个 `scf.parallel` 按 `(m, block)` 处理。

为了减少中间 buffer，可以直接在 matmul kernel 的 K-block 循环里量化 A：

```text
for m:
  for n:
    for k_block:
      a_q8 = quantize(A[m, k_block])
      dot = sdot(a_q8, B_q8[n, k_block])
      acc += dot * a_scale * b_scale
```

但 decode 时 M=1，重复量化 A 的代价小；prefill 时 M 较大，建议先量化 A
一次，再复用。

### 7.2 Q8 dot

伪代码：

```text
acc = 0.0f
for kb in 0..K/32:
  int32 sum = 0
  for i in 0..32:
    sum += int32(a_q[kb*32+i]) * int32(b_q[kb*32+i])

  acc += float(sum) * a_d[kb] * b_d[kb]

out = bf16(acc)
```

### 7.3 ARM NEON / sdot

M2 支持 `FEAT_DotProd`，使用：

```asm
sdot v0.4s, v1.16b, v2.16b
```

一个 `sdot` 处理 16 个 int8 的 4 路点积。

每个 32-element block 需要 2 个 `sdot`：

```text
block 32 int8
  = 2 x 16 int8
  = 2 x sdot
```

然后：

```text
horizontal add int32x4
  -> int32
  -> f32 scale
```

LLVM intrinsic：

```text
llvm.aarch64.neon.sdot
```

或在 MLIR 中使用 `arm_neon.sdot`。

### 7.4 循环结构

```text
for n_block:                  # parallel, N 维
  for m:                      # serial
    acc = 0
    for kb in 0..K/32:
      a = Aq[m, kb]
      b = Bq[n_block, kb]
      acc += sdot(a, b) * a.d * b.d
    C[m, n_block] = bf16(acc)
```

prefill 时可以把 `m` 也做 tile，复用 B block。

## 8. 与现有 bf16/bfdot 路径的关系

保留两条路径：

```text
bf16:
  bfdot

int8:
  sdot
```

选择方式：

```text
--quant q8_0  -> int8 path
default       -> bf16 path
```

`llmcp.linear` 的 lowering 根据 `weight_format` 分派：

```text
q8_0  -> emitQ8Matmul
other -> emitBfDotMatmul
```

## 9. 精度设计

### 9.1 权重

Q8_0：

- 每个 32-element block 独立 scale；
- scale 使用 fp16，和 llama.cpp 一致；
- 无 zero point。

### 9.2 激活

Q8_1：

- 每个 32-element block 独立 scale；
- f32 scale；
- 动态量化，不需要校准数据。

### 9.3 累加

- int8 dot 使用 int32；
- block 间使用 f32；
- 最终输出 bf16（linear）或 f32（lm_head）。

### 9.4 不量化的部分

- RMSNorm 权重；
- q_norm / k_norm；
- A_log / dt_bias；
- 残差 add；
- attention softmax；
- DeltaNet 递归 state。

这些部分对精度敏感，首期保持 bf16/f32。

## 10. 测试计划

### 10.1 单元测试

- Q8_0 quantize/dequant 误差；
- Q8_1 quantize；
- Q8 dot vs f32 dot；
- block 边界、K 非 32 倍数（应拒绝）。

### 10.2 算子测试

新增 numeric：

```text
linear_q8
lm_head_q8
```

与 bf16 参考比较：

```text
max_abs
mean_abs
top1
```

### 10.3 模型对齐

```bash
llmcp-run --quant q8_0 --text Hello --golden build/golden_logits.bin
```

目标：

```text
top1 一致
top5 overlap 5/5
max_abs 相比 bf16 不显著恶化
```

### 10.4 生成对齐

```bash
llmcp-run --quant q8_0 --text 'Hello world' --max-new-tokens 4
```

目标：

```text
tokens: 0 271 760 2614
```

### 10.5 性能

```bash
llmcp-benchmark --quant q8_0 --prefill-tokens 32 --decode-tokens 32
```

记录：

- prefill token/s；
- decode token/s；
- 内存占用；
- 与 bf16 的倍率。

## 11. 里程碑

### M1：格式与打包

- Q8_0 block 定义；
- `llmcp-pack --quant q8_0`；
- manifest；
- 权重大小校验；
- 量化/反量化单测。

### M2：标量 Q8 matmul

- 激活 Q8_1；
- 标量 int8 dot；
- numeric `linear_q8` / `lm_head_q8`；
- 与 f32 参考对齐。

### M3：NEON sdot

- `llvm.aarch64.neon.sdot`；
- N 维并行；
- 真实模型 AOT；
- top1/greedy 对齐。

### M4：性能

- A 激活复用；
- N/K block tiling；
- 对齐/预取；
- 与 bf16 路径 benchmark。

### M5：可选扩展

- KV cache Q8_0；
- Q8_K activation；
- embedding 量化；
- 更低位宽 K-quants。

## 12. 风险

1. **fp16 scale 精度**
   - Q8_0 使用 fp16 scale，极端值可能损失；
   - 可以先实现 f32 scale 的 Q8_0F 变体做对比。

2. **非对齐 block**
   - 34 字节 block 不利于 NEON load；
   - 运行时重排为 SoA。

3. **激活量化开销**
   - prefill 时 M 大，需要复用 Aq；
   - decode 时 M=1，开销可接受。

4. **精度敏感层**
   - 首期不量化 norm / DeltaNet state / KV cache；
   - 逐步放开。

5. **K 不是 32 的倍数**
   - 当前所有 linear/lm_head 的 K 都是 32 的倍数；
   - 新增层需要 fallback bf16 或补零。

6. **tied embedding**
   - embedding 和 lm_head 共用权重；
   - 若量化，需要同时支持 embedding 查表反量化和 lm_head Q8 matmul。

## 13. 结论

首期采用：

```text
Q8_0 权重
+ Q8_1 激活
+ int32 点积
+ f32 block 累加
+ ARM sdot
```

它对应 llama.cpp 的 Q8_0/Q8_1 思路，改动集中在：

```text
llmcp-pack
WeightLayout / manifest
LinearOp / LmHeadOp lowering
ARM sdot kernel
```

不需要改模型结构，也不需要校准数据。

## 参考

- llama.cpp quantization types:
  <https://github.com/ggml-org/llama.cpp/discussions/17393>
- llama.cpp `ggml-quants.h`:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.h>
- llama.cpp `ggml-quants.c`:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c>


## 14. M1 实现状态

已完成：

- `Q8_0Block` 格式；
- `quantizeTensorQ8_0` / `quantizeRowQ8_0` / `dequantizeQ8_0`；
- `WeightLayout` 的 `q8_0` 模式；
- `llmcp-pack --quant q8_0`；
- manifest 中的 `quantization`、`block_size`、`source_size`；
- 单元测试和 `llmcp-pack-real-q8` dry-run。

真实模型打包结果：

```text
source bytes: 8,411,510,272
arena bytes:  4,469,519,936
tensors:      426
```

Arena 约 4.47 GB，权重约 4.2 GiB。


## 15. M2 实现状态

已完成：

- `llmcp.linear` / `llmcp.lm_head` 增加 `weight_format = "q8_0"`；
- Q8_1 激活动态量化（标量）；
- Q8_0 x Q8_1 int32 点积；
- f32 scale 累加；
- numeric `linear_q8` / `lm_head_q8`；
- Q8 reference 使用量化后的输入和权重计算。

当前 Q8 路径是 scalar lowering，尚未接入 ARM `sdot`，也尚未让真实模型
`llmcp-run` 直接消费 Q8 arena。下一步 M3 做 NEON sdot 和真实模型对齐。


## 16. M3 进展

已完成：

- activation 预量化到 `[rows, K] i8` + `[rows, K/32] f32`；
- 每个 activation block 只量化一次，不再对每个 N 列重复；
- ARM `llvm.aarch64.neon.sdot`；
- 每个 32-element block 用 2 条 sdot；
- int32 点积后 f32 scale 累加；
- `llmcp-matmul-benchmark --q8`；
- `llmcp-matmul-benchmark-q8` CTest。

实测（Apple M2，4 线程，全 1 输入）：

```text
256x256x256:
  bf16: 约 29.9 GFLOPS
  q8:   约 32.7 GFLOPS

M=16, N=8192, K=2560:
  bf16: 约 55.9 GFLOPS
  q8:   约 189.5 GFLOPS
```

Q8 的收益主要来自：

- 权重内存流量减半；
- sdot 吞吐；
- 大规模 N/K 下更低的带宽压力。

尚未完成：

- `llmcp-run` 直接消费 Q8 arena；
- `ModelBuilder` 根据 manifest 生成 `weight_format="q8_0"`；
- 真实模型 Q8 top1/greedy 对齐。
