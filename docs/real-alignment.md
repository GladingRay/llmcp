# 真实 Qwen3.5 权重数值对齐

## 当前状态

已完成 **单 token、全 32 层、真实 Qwen3.5-4B 权重** 的语义对齐验证。

对齐对象：

- 参考实现：HuggingFace `transformers` 的 `Qwen3_5ForCausalLM`，bf16；
- 对照实现：[python/llmcp_run.py](../python/llmcp_run.py)，按 `llmcp` op 语义逐层重写：
  - zero-centered RMSNorm；
  - linear attention：causal conv、gated delta recurrence、gated RMSNorm；
  - full attention：q/gate 拆分、per-head q/k RMSNorm、partial RoPE、GQA、causal softmax、output gate；
  - MLP、最终 norm、tied lm_head。

入口：

```bash
/Users/guorui/Work/models/venv314/bin/python python/check_alignment.py
```

## 对齐结果

测试输入：`Hello`，token id `9419`，position 0，单 token prefill。

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

当前对照实现是 **Python 语义参考执行器**，不是 C++ AOT runtime，也没有直接执行 lowered MLIR。它的作用是把“模型数学 + 真实权重”先对齐，给后续 `llmcp-run` 一个可信 baseline。

尚未完成：

1. `llmcp-run` C++ 执行器：mmap weight arena、生成 cos/sin、调用 `llmcp_model`；
2. 多 token prefill 和 decode 追加的整模型对齐；
3. 用 lowered IR 执行并与 HF logits 对比。

## 下一步

1. 把 `python/llmcp_run.py` 的数学搬到 C++ `llmcp-run`，先用 Accelate/BLAS 做 matvec；
2. 用真实权重跑单 token，对齐 top-1/top-5 和 `max_abs`；
3. 再扩到 2~4 token prefill，验证 KV cache、causal conv state、DeltaNet state；
4. 最后把 `llmcp-run` 替换为 lowered MLIR 的 AOT 执行结果。
