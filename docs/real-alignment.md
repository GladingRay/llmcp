# 真实 Qwen3.5 权重数值对齐

## 当前状态

当前已完成 **真实 Qwen3.5-4B 权重** 的以下验证：

- 单 token、全 32 层、bf16 手写路径对齐；
- 多 token prefill / 4 token greedy decode；
- 完整图 AOT 路径；
- Q8_0 权重 + Q8_1 预量化激活的 AOT 路径；
- Q8 与 bf16 的 logits/top-k 对比。

长序列 AOT decode 还没有做系统性对齐。

对齐对象：

- 参考实现：HuggingFace `transformers` 的 `Qwen3_5ForCausalLM`，bf16；
- 对照实现：[python/llmcp_run.py](../python/llmcp_run.py)，按 `llmcp` op 语义逐层重写：
  - zero-centered RMSNorm；
  - linear attention：causal conv、gated delta recurrence、gated RMSNorm；
  - full attention：q/gate 拆分、per-head q/k RMSNorm、partial RoPE、GQA、causal softmax、output gate；
  - MLP、最终 norm、tied lm_head。

C++ 执行器：

```bash
./build/bin/llmcp-run --weights build/weights.bin \
  --manifest build/weights.json --text Hello \
  --golden build/golden_logits.bin
```

它完成：

- mmap `weights.bin`；
- 读取 `weights.json` 并建立 tensor → offset 映射；
- 通过 `tokenizers-cpp` 的 `Tokenizer::FromBlobJSON` 加载完整 `tokenizer.json`（HF BPE / ByteLevel / Unicode）；
- 生成 position 0 的 cos/sin；
- 调用 `runModel`，执行 32 层 embedding / DeltaNet / attention / MLP / final norm / lm_head；
- 与 HF golden logits 比较；
- 支持多 token prefill 和 `--max-new-tokens` greedy decode。

Python 语义参考入口：

```bash
/Users/guorui/Work/models/venv314/bin/python python/check_alignment.py
```

## 对齐结果

单 token 测试输入：`Hello`，token id `9419`，position 0。

C++ `llmcp-run` 单 token 结果：

```text
token: 9419
logits: 248320, max_abs 1.875000e-01, mean_abs 2.476386e-02
top1 ours=11 hf=11, top5 overlap=5/5
```

多 token prefill：

```bash
./build/bin/llmcp-run --weights build/weights.bin \
  --manifest build/weights.json --text 'Hello world!' --max-new-tokens 0
# next: 271
```

greedy decode：

```bash
./build/bin/llmcp-run --weights build/weights.bin \
  --manifest build/weights.json --text 'Hello world' --max-new-tokens 2
# tokens: 0 271
```

多 token 的后续 decode 还可能有漂移（前两个 token 与 HF greedy 一致），
主要是 chunked prefill 的 state 与 decode recurrence 的衔接还没完全对齐。

AOT 路径：

```bash
./build/bin/llmcp-run --weights build/weights.bin \
  --manifest build/weights.json --text Hello \
  --aot build/libqwen35_aot.dylib
# aot logits: 248320 next 11
```

AOT 路径严格按正常顺序执行：先加载 tokenizer，再调用 AOT，不使用子进程。
之前的 NaN 根因是 lowering 中 `linalg.matmul` 的输出 accumulator 未初始化；
现在 `LinearOp` / `LmHeadOp` 会先用 `linalg.fill` 清零。

多 token AOT prefill：

```bash
./build/bin/llmcp-run --weights build/weights.bin \
  --manifest build/weights.json --text 'Hello world!' \
  --aot build/libqwen35_aot.dylib
# aot logits: 248320 next 271
```

多 token AOT 贪心生成：

```bash
./build/bin/llmcp-run --weights build/weights.bin \
  --manifest build/weights.json --text 'Hello world' \
  --max-new-tokens 4 --aot build/libqwen35_aot.dylib
# output: !
#
# The following
# tokens: 0 271 760 2614
```

`output:` 后面是逐 token decode 并立即 flush 的明文；每生成一个 token，
就调用 tokenizer decode 一次，不等待全部生成结束。这与 HF greedy 的
`[0, 271, 760, 2614]` 一致。

Python 语义参考结果：

当前 C++ 手写路径结果：

```text
max_abs: 1.875000e-01
mean_abs: 2.476386e-02
top1: ours=11 hf=11
top5 overlap: 5/5
```

AOT 结果：

```text
aot logits: 248320 next 11
```

判定：

- top-1 完全一致；
- top-5 集合完全一致；
- 单 token logits 的 max_abs 远小于 1.0。

这证明权重布局、embedding、32 层 token mixer / MLP、最终 norm 和 lm_head 的语义是通的。

## Q8 对齐

Q8_0 权重 + Q8_1 预量化激活的 AOT 路径：

```text
Q8 AOT:
  next 11
  tokens: 0 271 760 2614

Q8 vs bf16 logits:
  max_abs  0.509457
  mean_abs 0.087098
  top1     11 / 11
  top5     [11, 0, 1017, 13, 4858] 一致
```

Q8 目前只量化 linear 权重；embedding / lm_head、KV cache、DeltaNet state
仍保持 bf16/f32。

## 边界

当前执行路径：

```text
llmcp-run
  ├── 默认：手写 runModel
  ├── --aot：bf16 AOT dylib
  └── --aot + Q8 arena：Q8 AOT dylib
```

尚未完成：

1. 更长序列的 AOT decode 数值对齐（当前 4 token 贪心一致）；
2. temperature / top-k / top-p 采样（目前只有 greedy）；
3. KV cache / DeltaNet state 量化；
4. Q8 embedding / lm_head；
5. 通用 tiling/vectorize（当前依赖 bfdot/sdot + OpenMP）。

## 下一步

1. 加 temperature / top-k / top-p 采样；
2. 用更长 prompt / 生成序列验证 AOT 与 HF 的一致性；
3. 做 KV cache / DeltaNet state 量化；
4. 继续优化量化 matmul 的 tiling 和内存访问。
