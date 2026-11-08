# 历史性能记录

> 本文保留 2026-10-26 的原始 benchmark 输出，用于对比。
> 当前实现已经加入 OpenMP、elementwise 融合、FlashAttention-2 和 Q8/sdot；
> 最新功能状态见 [README](../README.md)，并行化结果见
> [parallelization.md](parallelization.md)，Q8 见
> [int8-quantization.md](int8-quantization.md)。
>
> 本机性能受 page cache / 内存压力影响明显，单次 wall time 不应作为稳定基线。

# 2026.10.26 13:41
```bash
➜  llmcp git:(main) ./build/bin/llmcp-benchmark \
  --weights build/weights.bin \
  --aot build/libqwen35_aot.dylib \
  --prefill-tokens 16 \
  --decode-tokens 16
prefill_tokens: 16
decode_tokens: 16
prefill_seconds: 69.4327
decode_seconds: 58.2510
total_seconds: 127.6837
prefill_tokens_per_second: 0.2304
decode_tokens_per_second: 0.2747
```
# 2026.10.26 14:00
```bash
➜  llmcp git:(main) ./build/bin/llmcp-benchmark \
  --weights build/weights.bin \
  --aot build/libqwen35_aot.dylib \
  --prefill-tokens 16 \
  --decode-tokens 16
prefill_tokens: 16
decode_tokens: 16
prefill_seconds: 40.9588
decode_seconds: 32.6504
total_seconds: 73.6092
prefill_tokens_per_second: 0.3906
decode_tokens_per_second: 0.4900
```