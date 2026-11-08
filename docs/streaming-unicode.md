# Unicode 流式输出

Qwen3.5 的 ByteLevel tokenizer 可能把一个 UTF-8 字符的字节分散到多个
token 中。逐个独立调用 `Decode({token})` 会提前把不完整字节转换为 `�`，
导致中文、emoji 等输出与完整 token 序列的解码结果不同。

`llmcp-run` 的手写和 AOT 生成路径现在共享 `ByteLevelStreamingDecoder`：

1. 将新 token 加入尚未输出的 token 缓冲区，并一起解码。
2. 若文本以 Unicode 替换字符 `U+FFFD` 结尾，保留这些 token，等待后续字节。
3. 否则立即输出完整文本并清空缓冲区。
4. 遇到 EOS 或达到生成长度上限时调用 `finish()`，解码并输出所有剩余内容。

缓冲区只保留尚未输出的 token，不会每一步重新解码全部生成历史。
该实现针对当前模型的 ByteLevel decoder，其他 decoder 可能需要不同的上下文
管理方式。特殊 token 保持原有 `Decode` 行为，不额外过滤。

字面上的 `�` 也可能暂时被保留，但不会被丢弃。若生成结束时 UTF-8 字符
仍然不完整，最终输出保留 tokenizer 本身的替换字符；这与一次性解码相同，
不会擅自删除截断内容。完整字符的 token 序列拼接后的流式输出应与批量
`Decode` 完全一致，各次输出均为有效 UTF-8。

## 验证

```bash
ctest --test-dir build -R llmcp-unittests --output-on-failure
```

单元测试使用真实的 tokenizers-cpp ByteLevel 实现和完整的 256 字节词表，
无需模型权重，覆盖：

- ASCII 立即输出；中文三字节、emoji 四字节跨 token 缓冲；
- 混合中英文、组合字符、换行以及 emoji ZWJ 序列；
- 每个 token 位置终止时，与一次性解码对比；
- 一个 token 同时包含完整文本和下一个字符的部分字节；
- 特殊标记文本、字面替换字符和随机无效字节序列；
- `finish()` 不重复输出。
