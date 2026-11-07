# Linear + Activation 融合

## 1. 目标

把直接跟在 `llmcp.linear` 后面的激活函数融合到 linear 中，减少中间的
activation op，并让 activation 在 matmul 输出上执行。

目标模式：

```text
linear -> silu_mul
linear -> gated_rms_norm
linear -> gated_delta_net(b)
```

## 2. IR 扩展

### LinearOp

新增：

```mlir
llmcp.linear ... {activation = "silu"}
llmcp.linear ... {activation = "sigmoid"}
llmcp.linear ... {activation = "softplus"}
```

默认：

```text
activation = "none"
```

### MulOp

新增：

```mlir
%0 = llmcp.mul %gate, %up
```

用于 SwiGLU：

```text
silu(gate) * up
```

融合后：

```text
linear(activation=silu) -> mul
```

### GatedRmsNormOp

新增：

```mlir
gate_activated = true
```

表示 gate 已经由 linear 的 activation epilogue 计算过 SiLU。

### GatedDeltaNetOp

新增：

```mlir
b_activated = true
```

表示 `b` 已经由 linear 的 activation epilogue 计算过 sigmoid。

## 3. Pass

```bash
llmcp-opt input.mlir --llmcp-fuse-linear-activations
```

实现：

activation 在 matmul 的 f32 输出上直接计算，然后才 trunc/store，不再生成
单独的 activation tensor。

- `linear -> silu_mul`：
  - linear 设置 `activation = "silu"`；
  - `silu_mul` 替换为 `llmcp.mul`。
- `linear -> gated_rms_norm`：
  - linear 设置 `activation = "silu"`；
  - gated_rms_norm 设置 `gate_activated = true`。
- `linear -> gated_delta_net` 的 `b`：
  - linear 设置 `activation = "sigmoid"`；
  - gated_delta_net 设置 `b_activated = true`。

只融合单用户 linear，避免改变其他消费者的语义。

## 4. 当前融合范围

| pattern | 数量/forward | 激活 |
|---|---:|---|
| MLP gate -> silu_mul | 32 | SiLU |
| GatedRMSNorm gate | 24 | SiLU |
| DeltaNet b | 24 | Sigmoid |
| 合计 | 80 | |

`in_proj_a` 的 softplus 含 `+ dt_bias`，暂不融合；`in_proj_qkv` 到 conv
的 SiLU 也不是直接 linear -> activation。

## 5. 验证

- lit：`tests/lit/Conversion/fuse-linear-activation.mlir`
- AOT bf16/Q8：
  ```text
  tokens: 0 271 760 2614
  ```
- 32 prefill / 32 decode：
  ```text
  bf16: prefill 5.51 s, decode 6.15 s
  Q8:   prefill 4.26 s, decode 4.04 s
  ```
- 全量：
  ```text
  build/bin/llmcp-unittests
  ctest --test-dir build --output-on-failure
  ```

## 6. 文件

- `lib/Dialect/LLMCP/Transforms/FuseLinearActivations.cpp`
- `lib/Dialect/LLMCP/Transforms/LowerToLinalg.cpp`
- `lib/Dialect/LLMCP/Transforms/LowerGatedDeltaNet.cpp`
