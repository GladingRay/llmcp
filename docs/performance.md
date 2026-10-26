# 2026.10.26 13:41
```bash
➜  llmcp git:(main) ./build/bin/llmcp-benchmark \
  --weights build/weights.bin \
  --manifest build/weights.json \
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
  --manifest build/weights.json \
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