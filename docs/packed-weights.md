# 单文件权重格式

`llmcp-pack` 将权重数据和 JSON manifest 写入同一个 `.bin` 文件。
`llmcp-run`、`llmcp-benchmark` 和 golden 测试只需要权重文件路径，
不再生成、读取独立的 manifest 文件，也不再接受 `--manifest` 参数。

## 文件布局（版本 1）

```text
[64 字节固定头部]
[权重张量及 64 字节对齐填充]
[UTF-8 JSON manifest]
```

头部所有整数均使用小端序：

| 字节偏移 | 类型 | 含义 |
|---|---|---|
| 0 | 8 字节 | magic：`LLMCPWGT` |
| 8 | u32 | 格式版本：1 |
| 12 | u32 | 头部大小：64 |
| 16 | u64 | arena 字节数，包含头部和权重，不含 manifest |
| 24 | u64 | manifest 文件偏移，等于 arena 字节数 |
| 32 | u64 | manifest 字节数 |
| 40 | 24 字节 | 保留，写入零 |

文件总大小为 arena 字节数加 manifest 字节数。manifest 中的 `total_bytes`
仍表示 arena 大小；每个 tensor 的 offset 仍是相对于文件起点的偏移。
原本预留的 64 字节头部现在用于记录容器信息，因此所有权重偏移、group stride
和 AOT 中的地址常量保持不变。

运行时 mmap 完整文件，读取头部定位 manifest，并校验字段、形状、dtype、
字节数及边界。tensor 不允许指向头部或尾部 manifest。传给 AOT 的 arena
memref 长度仅包含头部和权重，解除映射时则使用完整文件长度。重新加载失败
不会破坏已经加载的 arena。

## 使用与迁移

```bash
build/bin/llmcp-pack --model /path/to/qwen3.5-4b --out build/weights.bin
build/bin/llmcp-pack --model /path/to/qwen3.5-4b \
  --quant q8_0 --out build/weights-q8.bin

build/bin/llmcp-run --weights build/weights.bin \
  --aot build/libqwen35_aot.dylib --text 'Hello world' --max-new-tokens 4

build/bin/llmcp-benchmark --weights build/weights-q8.bin \
  --aot build/libqwen35_aot_q8.dylib --prefill-tokens 32 --decode-tokens 32
```

旧的裸 arena 加外部 JSON 不受新加载器支持，需要使用新版本 `llmcp-pack`
重新打包。单独修改文件扩展名或传入旧 JSON 无法完成迁移。已有同布局、同量化
模式的 AOT 库可继续使用，本次文件格式变更不改变模型调用 ABI。

`--dry-run` 保持只校验、不写文件的行为。golden 测试的 CMake 配置不再检查
独立 manifest；新生成权重和动态库后仍需重新配置 CMake 才能注册测试。
