# 真实 Qwen3.5 权重数值对齐

## 当前状态

已完成 **单 token、全 32 层、真实 Qwen3.5-4B 权重** 的 C++ 执行器对齐验证；多 token prefill 和 greedy decode 已接入，长序列 decode 仍在收敛。

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

Python 语义参考结果：

```text
max_abs: 2.363281
mean_abs: 0.376517
top1: ours=11 hf=11
top5 overlap: 5/5
top5 ours: [11, 0, 4858, 13, 1017]
top5 hf:   [11, 0, 13, 1017, 4858]
```

判定：

- top-1 完全一致；
- top-5 集合完全一致；
- `max_abs < 3.0`。

这证明权重布局、embedding、32 层 token mixer / MLP、最终 norm 和 lm_head 的语义是通的。

## 边界

当前 C++ 执行器是 **手写模型函数**；完整图已经可以 AOT 生成 `libqwen35_aot.dylib`（导出 4 个 `llmcp_*` 入口），但 AOT library 的运行时驱动尚未接入，因此当前数值对齐仍由手写执行器完成。

尚未完成：

1. 长序列 decode 的完整数值对齐（当前前两个 token 与 HF 一致）；
2. 用 lowered MLIR 执行并与 HF logits 对比；
3. 采样和 chat template。

## 下一步

1. 校准 chunked prefill 的最终 DeltaNet state 与 decode recurrence 的衔接，消除长序列 decode 漂移；
2. 加 temperature / top-k / top-p 采样和 chat template；
3. 为 AOT shared library 写运行时驱动，替换手写 `llmcp-run`；
4. 做 tiling/vectorize，降低单 token 的几十秒级延迟。
