# 真实 Qwen3.5 权重数值对齐

## 当前状态

已完成 **单 token、全 32 层、真实 Qwen3.5-4B 权重** 的 C++ 执行器对齐验证。

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
- 与 HF golden logits 比较。

Python 语义参考入口：

```bash
/Users/guorui/Work/models/venv314/bin/python python/check_alignment.py
```

## 对齐结果

测试输入：`Hello`，token id `9419`，position 0，单 token prefill。

C++ `llmcp-run` 结果：

```text
token: 9419
logits: 248320, max_abs 2.265625e+00, mean_abs 3.731686e-01
top1 ours=11 hf=11, top5 overlap=5/5
```

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

当前 C++ 执行器是 **手写模型函数**，不是 lowered MLIR 的 AOT 产物；它先验证“真实权重 + 模型数学”，给后续 AOT 执行一个 baseline。

尚未完成：

1. 多 token prefill 和 decode 追加的整模型对齐；
2. 用 lowered MLIR 执行并与 HF logits 对比；
3. 完整 Unicode tokenizer（当前 minimal tokenizer 主要覆盖 ASCII 单词）。

## 下一步

1. 把 `python/llmcp_run.py` 的数学搬到 C++ `llmcp-run`，先用 Accelate/BLAS 做 matvec；
2. 用真实权重跑单 token，对齐 top-1/top-5 和 `max_abs`；
3. 再扩到 2~4 token prefill，验证 KV cache、causal conv state、DeltaNet state；
4. 最后把 `llmcp-run` 替换为 lowered MLIR 的 AOT 执行结果。
