# Qwen3.5-4B 模型规格与编译契约

> 本文档是 `llmcp` 编译器的语义基准（semantic contract）。算子语义、张量形状、状态布局和数值策略以本文为准。
>
> 数据来源：本机模型目录 `/Users/guorui/Work/models/qwen3.5-4b` 的 `config.json`、`model.safetensors.index.json`、两个 safetensors 文件头，以及 HuggingFace Transformers 的 `qwen3_5` 参考实现。所有形状均为实测值，不是推测值。

---

## 1. 结论先行

1. **这不是一个标准 Transformer。** 文本模型共 32 层，其中 24 层是 Gated DeltaNet 线性注意力（`linear_attention`），8 层是全注意力（`full_attention`），模式固定为 `[L, L, L, F] × 8`。
2. **文本模型 4.206B 参数**，视觉塔 0.334B，MTP 头 0.121B，合计 4.660B；BF16 权重合计 9.32GB。
3. **Decode 阶段同时存在三类状态**：8 层全注意力的 KV cache、24 层线性注意力的 recurrent state（f32，每层 2MiB）、24 层线性注意力的 causal-conv state。编译器必须把状态作为一等公民，不能只实现 KV cache。
4. **词嵌入与 lm_head 权重绑定（tied embedding）。** 模型中没有 `lm_head.weight`；`lm_head` 直接复用 `model.language_model.embed_tokens.weight`。它占 1.27GB 权重，每个 token 约 1.27 GFLOP，是单 token 计算和带宽的重要组成部分。
5. **全注意力**：16 个 query head、4 个 KV head（GQA 4:1）、`head_dim = 256`、partial RoPE（仅前 64 维）、q/k per-head RMSNorm、输出门控（`sigmoid(gate)`）。
6. **线性注意力**：16 个 key head、32 个 value head（2:1）、key/value head dim 均为 128、depthwise causal conv（kernel 4）、门控 RMSNorm、`beta` 与 log-decay `g`、f32 recurrent state。
7. **RoPE 配置**：`rope_theta = 1e7`，`partial_rotary_factor = 0.25`，即 256 维中只有 64 维旋转；`mrope_section = [11, 11, 10]`，`mrope_interleaved = true`。纯文本输入时 mRoPE 退化为普通 RoPE。
8. **Norm 是 zero-centered 的**：`output = x_norm * (1 + weight)`，权重初始化是 0 而不是 1。这是移植时最容易写错的地方之一。
9. **首期只需要文本路径**：视觉塔和 MTP 不参与普通 `forward`，可以在后续阶段实现。

---

## 2. 参数预算（实测）

| 组件 | 参数量 | BF16 字节数 | 说明 |
|---|---:|---:|---|
| 词嵌入（与 lm_head 共享） | 635,699,200 | 1.27GB | `[248320, 2560]` |
| 24 × 线性注意力层 | 2,710,172,160 | 5.42GB | 每层 112,923,840 |
| 8 × 全注意力层 | 859,877,376 | 1.72GB | 每层 107,484,672 |
| 最终 RMSNorm | 2,560 | 5KB | `[2560]` |
| **文本模型合计** | **4,205,751,296** | **8.41GB** | 4.206B |
| 视觉塔 | 333,514,240 | 0.67GB | 24 个 ViT block + patch embed + merger |
| MTP 头 | 120,599,552 | 0.24GB | 1 个全注意力层 + fc |
| **模型合计** | **4,659,865,088** | **9.32GB** | 4.660B |

> 文件头 `metadata.total_size = 9,319,737,856` 字节，与上表一致（F32 小张量比 BF16 多出的 7,680 字节来自 24 个 `A_log` 和 24 个 `linear_attn.norm`）。

各层参数拆分：

| 子模块 | 线性注意力层 | 全注意力层 |
|---|---:|---:|
| Token mixer | 42,139,840 | 36,700,672 |
| MLP（SwiGLU） | 70,778,880 | 70,778,880 |
| 两个 RMSNorm | 5,120 | 5,120 |
| 每层合计 | 112,923,840 | 107,484,672 |

---

## 3. 文本配置（`text_config`）

| 字段 | 值 | 对编译器的含义 |
|---|---|---|
| `hidden_size` | 2560 | 残差流宽度 |
| `intermediate_size` | 9216 | SwiGLU 中间宽度 |
| `num_hidden_layers` | 32 | 层数 |
| `layer_types` | 24 × `linear_attention` + 8 × `full_attention` | 每 4 层一个 full |
| `num_attention_heads` | 16 | 全注意力 query head 数 |
| `num_key_value_heads` | 4 | GQA |
| `head_dim` | 256 | 注意：不是 `hidden_size / heads = 160` |
| `attention_bias` | false | 所有线性层无 bias |
| `hidden_act` | `silu` | MLP 与 conv 后激活 |
| `rms_norm_eps` | 1e-6 | 所有 RMSNorm |
| `vocab_size` | 248320 | 嵌入行数；tokenizer 基础词表 248044，另有 33 个 added token |
| `tie_word_embeddings` | true | lm_head 复用嵌入 |
| `max_position_embeddings` | 262144 | 运行时受内存/算力限制，不应默认开到最大 |
| `linear_conv_kernel_dim` | 4 | depthwise causal conv |
| `linear_key_head_dim` | 128 | 线性注意力 K 维 |
| `linear_value_head_dim` | 128 | 线性注意力 V 维 |
| `linear_num_key_heads` | 16 | K head 数 |
| `linear_num_value_heads` | 32 | V head 数（2:1） |
| `mamba_ssm_dtype` | `float32` | recurrent state 用 f32 |
| `full_attention_interval` | 4 | 层类型生成规则 |
| `rope_theta` | 10000000 | RoPE base |
| `partial_rotary_factor` | 0.25 | 旋转 64/256 维 |
| `mrope_section` | `[11, 11, 10]` | 频率对（pair）分配，合计 32 对 |
| `mrope_interleaved` | true | 交错重排 |

视觉配置（v1 不实现，记录备查）：`depth=24`、`hidden_size=1024`、`num_heads=16`、`intermediate_size=4096`、`patch_size=16`、`temporal_patch_size=2`、`spatial_merge_size=2`、`out_hidden_size=2560`、`num_position_embeddings=2304`、`deepstack_visual_indexes=[]`。

MTP：`mtp_num_hidden_layers=1`、`mtp_use_dedicated_embeddings=false`；权重含 `mtp.fc.weight [2560, 5120]`、一个全注意力层、`mtp.pre_fc_norm_*`、`mtp.norm`。MTP 不在普通 `forward` 中执行，仅用于投机解码。

---

## 4. 张量清单

### 4.1 线性注意力层（`layer_types[i] == "linear_attention"`）

| 权重名（相对 `model.language_model.layers.N.`） | 形状 | dtype | 备注 |
|---|---|---|---|
| `input_layernorm.weight` | `[2560]` | BF16 | zero-centered |
| `post_attention_layernorm.weight` | `[2560]` | BF16 | zero-centered |
| `linear_attn.in_proj_qkv.weight` | `[8192, 2560]` | BF16 | 输出 `q(2048) / k(2048) / v(4096)` |
| `linear_attn.in_proj_z.weight` | `[4096, 2560]` | BF16 | 门控分支 |
| `linear_attn.in_proj_a.weight` | `[32, 2560]` | BF16 | 时间步 / decay |
| `linear_attn.in_proj_b.weight` | `[32, 2560]` | BF16 | beta |
| `linear_attn.conv1d.weight` | `[8192, 1, 4]` | BF16 | depthwise，可压成 `[8192, 4]` 或转置成 `[4, 8192]` |
| `linear_attn.dt_bias` | `[32]` | BF16 | softplus 偏置 |
| `linear_attn.A_log` | `[32]` | **F32** | `-exp(A_log)` 为衰减率 |
| `linear_attn.norm.weight` | `[128]` | **F32** | 门控 RMSNorm，作用在 V head 维 |
| `linear_attn.out_proj.weight` | `[2560, 4096]` | BF16 | |
| `mlp.gate_proj.weight` | `[9216, 2560]` | BF16 | |
| `mlp.up_proj.weight` | `[9216, 2560]` | BF16 | |
| `mlp.down_proj.weight` | `[2560, 9216]` | BF16 | |

### 4.2 全注意力层（`layer_types[i] == "full_attention"`）

| 权重名（相对 `model.language_model.layers.N.`） | 形状 | dtype | 备注 |
|---|---|---|---|
| `input_layernorm.weight` | `[2560]` | BF16 | zero-centered |
| `post_attention_layernorm.weight` | `[2560]` | BF16 | zero-centered |
| `self_attn.q_proj.weight` | `[8192, 2560]` | BF16 | 每 head 512 维 = q(256) / gate(256)，head-major 交错 |
| `self_attn.k_proj.weight` | `[1024, 2560]` | BF16 | |
| `self_attn.v_proj.weight` | `[1024, 2560]` | BF16 | |
| `self_attn.o_proj.weight` | `[2560, 4096]` | BF16 | |
| `self_attn.q_norm.weight` | `[256]` | BF16 | per-head，zero-centered |
| `self_attn.k_norm.weight` | `[256]` | BF16 | per-head，zero-centered |
| `mlp.*` | 同上 | BF16 | |

### 4.3 全局张量

| 名称 | 形状 | dtype | 说明 |
|---|---|---|---|
| `model.language_model.embed_tokens.weight` | `[248320, 2560]` | BF16 | 同时作为 lm_head |
| `model.language_model.norm.weight` | `[2560]` | BF16 | 最终 norm，zero-centered |
| `lm_head.weight` | — | — | 不存在（tied） |

### 4.4 视觉塔与 MTP

- 视觉：`model.visual.patch_embed.proj.weight [1024, 3, 2, 16, 16]`、`pos_embed.weight [2304, 1024]`、24 × `blocks.{i}.attn.{qkv,proj}.{weight,bias}`、`blocks.{i}.mlp.linear_fc{1,2}.{weight,bias}`、`blocks.{i}.norm{1,2}.{weight,bias}`、`merger.{linear_fc1,linear_fc2,norm}`。
- MTP：`mtp.fc.weight [2560, 5120]`、`mtp.pre_fc_norm_embedding.weight [2560]`、`mtp.pre_fc_norm_hidden.weight [2560]`、`mtp.layers.0.*`（一个全注意力层）、`mtp.norm.weight [2560]`。

> 视觉 attention 使用 `qkv` 融合权重和 bias，`proj` 有 bias；与文本层不同。首期不实现。

---

## 5. 文本前向图

```text
tokens [S]
  -> embedding lookup                    # [S, 2560] bf16
  -> 32 × decoder layer
       r = h
       h = rms_norm_zero(h, input_norm)  # bf16, 内部 f32
       h = token_mixer(h)                # linear_attention 或 full_attention
       h = r + h
       r = h
       h = rms_norm_zero(h, post_norm)
       h = down_proj(silu(gate_proj(h)) * up_proj(h))
       h = r + h
  -> rms_norm_zero(h, final_norm)
  -> lm_head(h_last)                     # 只对最后一个位置算 logits
```

生成阶段只需要最后一个位置的 logits。Prefill 阶段如果对所有位置都算 lm_head，`[S, 248320]` 的输出会非常大（S=2048 时约 2GB f32），因此运行时必须只取最后一个 hidden state；训练/困惑度评估才需要逐位置计算。

---

## 6. 线性注意力（Gated DeltaNet）

### 6.1 投影、卷积、激活

设 `S` 为序列长度，`H = 2560`：

```text
mixed = x @ W_qkv^T                    # [S, 8192] = q 2048 | k 2048 | v 4096
z     = x @ W_z^T                      # [S, 4096]
b     = x @ W_b^T                      # [S, 32]
a     = x @ W_a^T                      # [S, 32]

# causal depthwise conv，kernel = 4，无 bias，激活 silu
mixed = silu(causal_conv1d(mixed, W_conv))   # [S, 8192]

q = mixed[:, 0:2048]   -> [S, 16, 128]
k = mixed[:, 2048:4096] -> [S, 16, 128]
v = mixed[:, 4096:8192] -> [S, 32, 128]
z = z.view(S, 32, 128)

beta = sigmoid(b)                                  # [S, 32]
g    = -exp(A_log) * softplus(a + dt_bias)         # [S, 32]，f32
```

- conv 的输入是投影后的 `mixed`，**不是** hidden state；conv 在 q/k/v 拆分之前。
- `conv1d` 是 depthwise，`groups = 8192`，kernel 4，无 bias，padding 3（左补 3 个零）后取前 `S` 个输出。decode 单 token 时数学上只需要最近 3 个历史输入；HF 的 cache 为了方便实际保存 `kernel_size = 4` 个输入（其中最早的一个不参与当前 token 的输出），llmcp 只存 3 个，输出等价。对比 golden 时只比较 conv 输出，不比较 conv_state 的原始缓冲。
- `g <= 0`，在 chunk 算法中以 log-space 累积。

### 6.2 Decode（单 token）recurrence

HF 在 `seq_len == 1` 且存在 cache 时走 recurrent kernel：

```text
# q/k: [16, 128] -> repeat_interleave(2) -> [32, 128]
# 先做 L2 normalize，再对 q 乘 128^-0.5
q = l2norm(q, eps=1e-6) * 128**-0.5
k = l2norm(k, eps=1e-6)

S_state = S_state * exp(g)                              # [32, 128, 128] f32
kv_mem  = einsum('hvk,hk->hv', S_state, k)               # [32, 128]
delta   = (v - kv_mem) * beta                            # [32, 128]
S_state = S_state + einsum('hk,hv->hkv', k, delta)
o       = einsum('hvk,hk->hv', S_state, q)               # [32, 128]
```

注意 `o` 使用的是**更新后**的 state。

### 6.3 Prefill（chunked）算法

HF 默认 `chunk_size = 64`，实现位于 `torch_chunk_gated_delta_rule`。核心步骤：

```text
# 1) 序列右侧 pad 到 chunk 的整数倍；q/k/v 转 f32
# 2) q/k L2 normalize；q 乘 128^-0.5
# 3) 每个 chunk 内：
cum_decay[i]      = sum_{j<=i} g[j]                      # 前缀和
pair_decay[i, j]  = exp(cum_decay[i] - cum_decay[j])     # i >= j，否则 0
ut_system         = (k_beta @ k^T) * pair_decay          # k_beta = k * beta
intra_attn        = (q @ k^T) * pair_decay
new_values        = solve_lower_unit(ut_system, v_beta)  # 前向代入
k_cumdecay        = solve_lower_unit(ut_system, k_beta * exp(cum_decay))
# 4) 跨 chunk 的顺序扫描：
v_new = new_values[c] - k_cumdecay[c] @ S
o[c]  = q_decayed[c] @ S + intra_attn[c] @ v_new
S     = S * exp(cum_decay[-1]) + k_decayed[c]^T @ v_new
```

其中 `q_decayed = q * exp(cum_decay)`，`k_decayed = k * exp(cum_decay[-1] - cum_decay)`。

序列右侧 pad 到 chunk 整数倍时，pad 位置的 `beta = 0`、`g = 0`（decay = 1），因此不会改变 state；输出在最后裁掉。实现时也可以按实际长度裁剪，避免无用的 pad 计算。

编译器实现建议：

- 首期 decode 使用 recurrent；prefill **先按 token 循环调用 recurrent**（慢但正确），验证通过后再实现 chunked。
- chunked 的 `solve_lower_unit` 可用 `scf.for` 前向代入，或后续优化为分块求逆；chunk 64 时矩阵很小（64×64 f32）。
- 所有中间量用 f32；`q/k/v` 从 bf16 升到 f32 后计算，输出转回 bf16。
- 32 个 v head 并行；k head 与 v head 的映射是 `k_head = v_head / 2`（`repeat_interleave` 的语义）。

### 6.4 输出门控与 out_proj

```text
o = o.reshape(S * 32, 128)
z = z.reshape(S * 32, 128)
o = gated_rms_norm(o, z, norm_weight)      # norm 在 f32，gate 为 silu(z)
o = o.reshape(S, 4096)
y = o @ W_out^T                            # [S, 2560]
```

`gated_rms_norm` 的精确语义：

```text
variance = mean(o_f32^2) + eps
o_norm   = o_f32 * rsqrt(variance)
o_norm   = norm_weight * o_norm.to(bf16)   # HF 先转 bf16 再乘 weight
o_out    = o_norm * silu(z_f32)
return o_out.to(bf16)
```

移植时可以在 f32 中完成整个计算再转 bf16；差异在容差内。若要严格对齐 HF，注意 `norm_weight * o_norm.to(bf16)` 这一步的精度截断。

### 6.5 状态布局

| 状态 | 逻辑形状（每层） | dtype | 24 层合计 |
|---|---|---|---|
| recurrent state | `[32, 128, 128]` | F32 | 48MiB |
| conv state | `[3, 8192]`（最近 3 个历史输入） | BF16 | 1.18MiB |

建议全局布局为 `[24, 32, 128, 128]` f32 和 `[24, 3, 8192]` bf16；每层切片连续，便于向量化。conv state 采用 `[3, channels]` 而不是 `[channels, 3]`，这样 4 个 tap 的读取都是连续向量。

---

## 7. 全注意力

### 7.1 投影与门控

```text
qg   = x @ W_q^T                      # [S, 8192] = 16 × [q_h(256) | gate_h(256)]
qg   = qg.view(S, 16, 2, 256)
q, gate = qg[..., 0, :], qg[..., 1, :]  # [S, 16, 256], [S, 16, 256]
k    = (x @ W_k^T).view(S, 4, 256)
v    = (x @ W_v^T).view(S, 4, 256)

q = rms_norm_zero_per_head(q, q_norm) # 每个 head 256 维
k = rms_norm_zero_per_head(k, k_norm)
q, k = rope_partial(q, k)             # 只旋转前 64 维
```

注意 `q_proj` 的输出顺序是**每 head 内 `[q(256), gate(256)]` 交错**，不是全局前 4096 维 q、后 4096 维 gate。若按 flat 前后切分，q 与 gate 会错位。`gate` 在 attention 输出之后逐元素作用，reshape 回 `[S, 4096]` 后与 attention 输出的 head-major 布局一致。

### 7.2 Attention

```text
# GQA: 16 个 q head 共享 4 个 kv head，repeat 4 次
attn = softmax_f32(q @ k^T * (256**-0.5) + causal_mask)
out  = attn.to(bf16) @ v
out  = out * sigmoid(gate)            # bf16
y    = out.reshape(S, 4096) @ W_o^T   # [S, 2560]
```

- `scale = 256^-0.5 = 0.0625`，是精确值。
- softmax 在 f32 中计算，再转 bf16 做 PV matmul；这与 HF eager 实现一致。
- decode 时 `q` 为单 token，`k/v` 来自 KV cache；prefill 时写入 cache。
- 没有 sliding window，causal mask 是唯一 mask。

### 7.3 KV cache 布局

每层每 token：`2 (K,V) × 4 heads × 256 dim × 2 bytes = 4KB`；8 层共 **32KB/token**。

建议布局：

```text
kv_cache[8][2][4][max_seq][256] bf16
```

即 `[layer, k_or_v, kv_head, position, head_dim]`，同一 head 的历史位置连续，attention 顺序读取友好。每个 token 写入 8 × 2 × 4 × 256 × 2 = 32KB。

缓存里保存的是**经过 k_norm 与 RoPE 之后的 k**，以及 v；不要缓存投影后的原始 k。`llmcp.attention` 通过 `position` / `cache_length` 操作数指定写入起点与当前有效长度：decode 在 `position` 写入后读取 `[0, position]` 闭区间，prefill 写入 `[position, position+S)` 并读取 `[0, position+S)`。

---

## 8. RoPE 与 mRoPE

### 8.1 频率

```text
dim       = int(head_dim * partial_rotary_factor) = int(256 * 0.25) = 64
inv_freq  = 1 / (theta ** (arange(0, dim, 2) / dim))   # 32 个频率
freqs     = position * inv_freq                        # [S, 32]
cos, sin  = cos(freqs), sin(freqs)                     # [S, 32]
```

`rotate_half` 在 64 维上操作：前 32 维与后 32 维交换并取负。只作用于 q/k 的前 64 维，后 192 维保持不变。

### 8.2 mRoPE（多模态）

文本模型内部把 `position_ids` 扩展为 4 组：`[text, T, H, W]`。纯文本时四组相同，mRoPE 退化为普通 RoPE。

多模态时的重排（`recomposition_frequencies`）：

```text
freqs_thw = freq[T]                          # 以 T 为底
freqs_thw[..., 1 : 33 : 3] = freq[H][..., 1 : 33 : 3]   # 11 个位置
freqs_thw[..., 2 : 30 : 3] = freq[W][..., 2 : 30 : 3]   # 10 个位置
cos/sin = cat([freqs_thw, freqs_thw], dim=-1)           # 64 维
```

`mrope_section = [11, 11, 10]` 指的是**频率对**的数量，合计 32 对 = 64 维。交错模式意味着 H/W 的频率散布在 T 的频率序列中，而不是简单的三段拼接。

首期文本路径：

- 位置 `pos` 单调递增，`cos/sin` 表按 `[S, 64]` 生成，f32 计算后按需转 bf16。
- Decode 每步只生成一行 64 维的 cos/sin，或复用 runtime 的预计算表。
- 视觉路径需要 3D position ids；在实现视觉塔时再启用完整 mRoPE。

---

## 9. RMSNorm 变体

### 9.1 zero-centered RMSNorm（文本层、q/k norm、最终 norm）

```text
variance = mean(x_f32^2) + 1e-6
y = x_f32 * rsqrt(variance) * (1 + weight_f32)
return y.to(bf16)
```

`weight` 在 checkpoint 中初始化为 0；如果把权重当成普通 RMSNorm 的 weight（初始化为 1）来用，模型会完全失效。

### 9.2 gated RMSNorm（线性注意力输出）

见 6.4。它与 9.1 的区别：

- 使用普通 weight（初始化为 1），不是 `1 + weight`；
- 乘 `silu(z)` 门控；
- 作用在 128 维的 V head 上，而不是 2560 维 hidden。

---

## 10. 内存与算力预算

### 10.1 常驻内存（文本，BF16）

| 项目 | 4K context | 32K context |
|---|---:|---:|
| 权重 | 9.32GB | 9.32GB |
| KV cache | 128MiB | 1.0GiB |
| 线性注意力 state | 48MiB | 48MiB |
| conv state | 1.18MB | 1.18MB |
| 激活/workspace（prefill 2048 bucket） | ~256MiB | ~256MiB |
| **合计（粗估）** | **~9.75GB** | **~10.6GB** |

M2/24GB 可以放下 BF16 模型；但要注意 mmap 的 page cache 压力、macOS 其他进程占用，以及 prefill 临时 buffer。W4 量化后权重约 2.6GB，余量充足。

### 10.2 单 token 计算量

```text
文本层参数 3.570B × 2 FLOP   = 7.14 GFLOP
lm_head 0.636B × 2 FLOP      = 1.27 GFLOP
合计                          ≈ 8.41 GFLOP/token
```

- Decode 是带宽/计算混合受限：BF16 权重约 9.32GB，M2 理论带宽约 100GB/s，仅权重读取就约 93ms/token；这是理论上限估算，实际受 page cache、内存压力和线程调度影响。
- Prefill 是计算受限：512 token ≈ 4.3 TFLOP；实际耗时取决于 OpenMP 线程数、bfdot/sdot 和内核对齐。
- 量化要真正提速，不能只做“dequant 后 f32 FMA”，而要用 `sdot/udot`/`i8mm` 做 int8 点积。当前 Q8 路径已使用 ARM `sdot`。

---

## 11. 权重命名与打包

### 11.1 名称映射

```text
model.language_model.embed_tokens.weight
model.language_model.layers.{i}.input_layernorm.weight
model.language_model.layers.{i}.post_attention_layernorm.weight
model.language_model.layers.{i}.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}.weight
model.language_model.layers.{i}.linear_attn.conv1d.weight
model.language_model.layers.{i}.linear_attn.{A_log,dt_bias,norm.weight}
model.language_model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
model.language_model.layers.{i}.self_attn.{q_norm,k_norm}.weight
model.language_model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight
model.language_model.norm.weight
model.visual.*
mtp.*
```

### 11.2 编译期布局

建议由 `llmcp-pack` 生成单一权重 blob，布局由 C++ 的 `WeightLayout` 统一计算，编译器和打包工具共享同一份代码：

```text
[header/manifest]
[embedding]                      # 248320 × 2560 bf16
[group 0: layers 0..3]           # 3 × linear + 1 × full
[group 1: layers 4..7]
...
[group 7: layers 28..31]
[final norm]
```

每个 group 的 4 层张量集合完全相同，因此 group 的字节跨度是固定的；层内偏移也是固定的。这样编译器可以用 `scf.for` 遍历 8 个 group，权重地址是 `group_base + constant_offset`，既避免展开 32 层导致 IR 爆炸，又能保持地址可计算。

**全注意力的 q/gate 交错**：`q_proj.weight [8192, 2560]` 的行按 head 组织为 `[q_0(256), gate_0(256), q_1(256), gate_1(256), ...]`。打包时可以：

1. 保持原始交错，kernel 以 512 行为一个 head 块处理；或
2. 在 `llmcp-pack` 中反交错成两个连续的 `[4096, 2560]` 矩阵（`q_all`、`gate_all`），kernel 更简单。

无论选哪种，**不要**把 flat 的前 4096 行当 q、后 4096 行当 gate。建议采用方案 2，并在 `WeightLayout` 中记录反交错后的逻辑张量名。

每个张量按 64 字节对齐；量化时额外存放 group scale/zero point，布局由同一个 `WeightLayout` 计算。

---

## 12. 数值语义与验证容差

### 12.1 精度策略

| 位置 | 存储 | 计算 |
|---|---|---|
| 权重 | BF16 | 读入后扩到 f32 |
| 激活/残差 | BF16 | 逐元素可 bf16 或 f32 |
| MatMul 累加 | — | f32 |
| RMSNorm | bf16 in/out | f32 |
| RoPE | bf16 in/out | f32（可选用 bf16 表以贴近 HF） |
| Attention score/softmax | — | f32 |
| Attention PV | bf16 | bf16 输入、f32 累加（HF 为 bf16 matmul） |
| 线性注意力 state | f32 | f32 |
| logits | f32 | f32 |

### 12.2 与 HF 的差异来源

1. 约简顺序不同（多线程、向量化、tile 划分）。
2. FMA 融合与不融合的差异。
3. HF 在 RoPE/attention gate 中使用 bf16 运算，我们可能用 f32。
4. 线性注意力 chunked 与 recurrent 的数学等价性在浮点下不严格成立。

因此**不要追求 bitwise 一致**，用分层容差 + token 一致率：

| 层级 | 指标 | 建议阈值 |
|---|---|---|
| 单算子（f32 参考） | max abs / rel err | `1e-4 / 1e-3` |
| 单层（bf16 参考） | max abs err / SNR | `< 5e-2` / `> 30dB` |
| 全模型 logits | max abs err；top-1；top-5 | `< 0.1`；100%；`> 95%` |
| 贪心生成 64 token | 与 HF 完全一致的比例 | `>= 95%` |
| 困惑度（小样本） | 相对误差 | `< 1%` |

> 阈值需要在 M0 阶段用 HF bf16 CPU 参考实测后固定；上表是初始目标。

### 12.3 Golden 数据

- 固定 8 组 prompt（中文、英文、代码、长上下文、特殊 token、空 prompt、重复 token、多轮对话）。
- 用 `transformers 5.20.2`（PyPI 版已含 `qwen3_5`）在 CPU 上以 `attn_implementation="eager"`、`dtype=torch.bfloat16` 跑参考；环境为 `/Users/guorui/Work/models/venv314`。
- 通过 forward hook 导出每层输入/输出、q/k/v、attention 输出、mlp 输出、logits；只保存选定位置（例如首 token、末 token、中间 token）以控制体积。
- 测试数据不入库；用脚本重新生成，记录模型 hash、transformers commit、prompt hash。

---

## 13. 工程验证项与当前状态

> 下列问题大部分已在真实权重路径验证；剩余项在后续长序列/多模态工作中继续确认。

1. **[已验证]** HF CPU bf16 matmul 的累加精度与 llmcp f32 累加在单 token/4 token 对齐容差内。
2. **[未完成]** `torch_chunk_gated_delta_rule` 与 recurrent 的长序列偏差；当前实现使用 recurrent。
3. **[未系统验证]** `linear_attn` 对 padding token 的 mask；当前 batch=1 无 padding。
4. **[已验证]** full attention prefill 的 causal mask；FlashAttention-2 路径已接入。
5. **[已验证]** RoPE cos/sin dtype 截断；真实模型 top-1/top-5 对齐。
6. **[未开始]** 视觉塔 position embedding、2D RoPE、merger。
7. **[未开始]** MTP fc 拼接顺序与 draft 采样。
8. **[基本验证]** tokenizer NFC / pre-tokenizer 边界；已有 tokenizer/chat-template 测试，长尾字符仍需补充。

---

## 14. 参考链接

- Qwen3.5 文本建模代码（Transformers main）：[modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- Qwen3.5 配置代码：[configuration_qwen3_5.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/configuration_qwen3_5.py)
- Qwen3.5 模型卡：[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- MLIR Linalg dialect：[Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- MLIR Vector dialect：[Vector Dialect](https://mlir.llvm.org/docs/Dialects/Vector/)
- MLIR Transform dialect：[Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
- MLIR Bufferization：[Bufferization](https://mlir.llvm.org/docs/Bufferization/)
