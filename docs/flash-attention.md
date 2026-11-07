# FlashAttention-2 style full attention

## 1. 实现范围

当前 full attention 使用 **FlashAttention-2 风格的 online softmax + key tiling**：

- 不再分配整个 `scores[cache_length]`；
- key 维按 `Bc = 64` 分块；
- 每个 query/head 维护 running max、running sum 和 output accumulator；
- 一个 key block 内完成 score、max、softmax 权重和 V 累加；
- 保留 causal mask、GQA、partial RoPE、q/k RMSNorm 和 output gate。

代码：

- AOT lowering：[LowerAttention.cpp](lib/Dialect/LLMCP/Transforms/LowerAttention.cpp)
- 手写路径：[llmcp-run.cpp](tools/llmcp-run/llmcp-run.cpp)

## 2. 算法

对每个 query 位置 `q`、每个 head：

```text
m = -inf
l = 0
O[head_dim] = 0

for key_block in 0..query_end step Bc:
  tile_scores = []
  tile_max = -inf

  for k in block:
    s = q · K[k] * scale
    tile_scores[k] = s
    tile_max = max(tile_max, s)

  new_m = max(m, tile_max)
  correction = exp(m - new_m)

  l *= correction
  O *= correction

  for k in block:
    p = exp(tile_scores[k] - new_m)
    l += p
    O += p * V[k]

  m = new_m

out = O / l * sigmoid(gate)
```

关键点：

- `m`、`l`、`O` 只在当前 query/head 内维护；
- 不保存完整 attention matrix；
- 每个 query 只顺序扫一遍 K/V；
- causal：`query_end = position + query_pos + 1`；
- GQA：`kv_head = head / (num_heads / num_kv_heads)`。

## 3. AOT scratch

AOT lowering 中使用：

```text
acc[head][head_dim]      # f32 output accumulator
score_tile[head][64]     # f32 block scores
```

两者按 head 分开，因此 head 循环并行时没有写冲突。

## 4. 手写路径

`runModel` 的 `fullAttention` 使用同样的 online softmax：

```cpp
for blockStart step 64:
  compute tile scores and tile max
  newMax = max(runningMax, tileMax)
  correction = exp(runningMax - newMax)
  runningSum *= correction
  context *= correction
  for k in block:
    p = exp(score - newMax)
    runningSum += p
    context += p * V[k]
  runningMax = newMax
```

## 5. 验证

数值测试：

```text
attention:        OK
attention_decode: OK
```

全量：

```text
100% tests passed, 0 tests failed out of 16
```

真实模型：

```text
max_abs 1.875000e-01
mean_abs 2.476386e-02
top1 ours=11 hf=11
top5 overlap=5/5
```

AOT：

```text
aot logits: 248320 next 11
tokens: 0 271 760 2614
```

## 6. 限制

这是 CPU 上的 FA2 风格实现，不是 GPU tensor-core 版本：

- query 维仍按 token 串行；
- 没有 shared memory / TMA；
- 没有 warp-level 并行；
- key block 的 `Bc` 固定为 64；
- 性能受内存带宽和 page cache 影响很大。

后续可以继续做：

1. query block `Br > 1`；
2. 多 query block 并行；
3. K/V tile 复用；
4. 与 bfdot 的 K 维向量化结合。
