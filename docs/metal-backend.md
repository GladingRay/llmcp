# Apple Metal 后端技术方案

## 1. 目标

为 llmcp 增加 Apple Metal GPU 后端，在 Apple Silicon 上执行 Qwen3.5-4B：

- 复用现有 `llmcp` 方言和 `ModelBuilder` 图；
- 复用 packed weight arena；
- 复用 tokenizer、chat template、greedy decode；
- Metal 路径与现有 CPU AOT 路径数值对齐；
- 保留 CPU 回退。

首期目标不是“把 LLVM IR 自动翻译成 Metal”，而是：

> 建立一套 handwritten Metal kernel library + host dispatcher，
> 先跑通完整模型，再考虑更自动化的 codegen。

## 2. 为什么不能直接用 MLIR GPU 后端

MLIR 上游没有 Apple Metal target：

```text
MLIR GPU dialect -> SPIR-V / NVVM / ROCDL
```

没有：

```text
MLIR GPU dialect -> Apple Metal Shading Language
```

因此不能简单用：

```text
linalg -> gpu -> metal
```

可行路线有三类：

| 路线 | 描述 | 优点 | 缺点 |
|---|---|---|---|
| A | 手写 MSL kernel + ObjC++ host | 可控、可定制 DeltaNet/attention | kernel 工作量大 |
| B | MPS / MPSGraph | matmul/softmax 快 | 自定义算子受限、图不透明 |
| C | 扩展 MLIR 自定义 Metal target | 自动化程度高 | 工程量大、上游不支持 |

建议：

```text
首期 A + 局部 B
```

即：

- 自定义 MSL kernel 实现 llmcp 算子；
- matmul 可先用 MPSMatrixMultiplication 或自定义 simdgroup GEMM；
- 后续再把 matmul 换成自定义 kernel。

## 3. 总体架构

```text
                 ┌────────────────────────────┐
                 │  llmcp graph / ModelBuilder │
                 └─────────────┬──────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
       CPU backend                         Metal backend
       llmcp-compile                     MetalModelRunner
       linalg/LLVM                       MSL kernels
       AOT dylib                         .metallib
              │                                 │
       OpenMP/bfdot                     MTLCommandQueue
              │                                 │
              └────────────┬────────────────────┘
                           │
                    llmcp-run
                    tokenizer / decode
```

组件：

```text
lib/Backend/Metal/
  MetalContext.{h,mm}        device / queue / library
  MetalBuffer.{h,mm}         MTLBuffer wrapper
  MetalKernelLibrary.{h,mm}  pipeline cache
  MetalModelRunner.{h,mm}    op dispatch
  Kernels/
    elementwise.metal
    rmsnorm.metal
    matmul.metal
    attention.metal
    delta_net.metal
    causal_conv.metal
```

## 4. Host 侧设计

### 4.1 MetalContext

```cpp
class MetalContext {
public:
  bool init();
  id<MTLDevice> device();
  id<MTLCommandQueue> queue();
  id<MTLLibrary> library();

private:
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLLibrary> library_;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
};
```

职责：

- `MTLCreateSystemDefaultDevice()`；
- 加载 `.metallib`；
- 创建 `MTLComputePipelineState`；
- 管理 command queue；
- 设备能力检查。

### 4.2 MetalBuffer

```cpp
class MetalBuffer {
public:
  static MetalBuffer wrapNoCopy(void *ptr, size_t bytes);
  id<MTLBuffer> buffer();
  size_t size();
};
```

Apple Silicon 使用统一内存：

```text
MTLResourceStorageModeShared
```

权重 arena 可以用：

```objc
[device newBufferWithBytesNoCopy:base
                          length:size
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
```

避免 CPU → GPU 再复制 8 GB 权重。

需要注意：

- `newBufferWithBytesNoCopy` 要求页对齐；
- `device.maxBufferLength` 可能小于 8 GB；
- 需要按 `maxBufferLength` 切分 arena；
- tensor offset 仍用 manifest 的字节偏移。

### 4.3 MetalModelRunner

```cpp
class MetalModelRunner {
public:
  bool init(const std::string &metallib,
            const Arena &arena,
            const MetalConfig &config);
  bool forward(const std::vector<int64_t> &tokens,
               int64_t position,
               int64_t cacheLength,
               std::vector<float> &logits);
};
```

内部按 `llmcp` 图顺序 dispatch：

```text
embedding
for layer:
  rms_norm
  linear q/k/v
  attention / delta_net
  linear o
  residual add
  rms_norm
  gate/up
  silu_mul
  down
  residual add
final rms_norm
lm_head
```

## 5. Metal kernel 规划

### 5.1 Elementwise / RMSNorm

适合首先实现：

- `add`
- `silu_mul`
- `embedding`
- `rms_norm`
- `gated_rms_norm`

策略：

```text
一个 threadgroup 处理一个 row 或一个 tile
线程数 128 / 256
reduction 用 simd_shuffle
```

RMSNorm：

```metal
sum = simd_sum(x * x)
rstd = rsqrt(sum / hidden + eps)
y = x * rstd * (1 + w)
```

### 5.2 Linear / LmHead

计算：

```text
C[M,N] = A[M,K] * B[N,K]^T
```

两阶段：

1. 先用 `MPSMatrixMultiplication` 验证；
2. 再用自定义 MSL GEMM。

自定义 GEMM：

```text
tile: 32x32 或 64x64
K tile: 16/32
threadgroup memory: A tile + B tile
simdgroup_matrix<half, 8, 8>
```

Decode：

```text
M = 1
```

使用 GEMV 路径，避免启动大 GEMM。

Prefill：

```text
M > 1
```

使用 tiled GEMM。

### 5.3 Attention

full attention：

- Q/K RMSNorm；
- partial RoPE；
- KV cache；
- GQA；
- causal；
- FlashAttention-2 online softmax；
- output gate。

kernel 粒度：

```text
一个 threadgroup 处理一个 (query position, head)
```

或者：

```text
一个 threadgroup 处理一个 query block 的多个 head
```

KV cache：

```text
[layer, 2, kv_heads, capacity, head_dim]
```

### 5.4 DeltaNet

linear attention 的 state：

```text
[layer, value_heads, 128, 128] f32
```

策略：

```text
一个 threadgroup 处理一个 value head
时间维串行
state 留在 threadgroup memory 或 device memory
```

prefill 的 chunk 形式可以后续优化。

### 5.5 Causal Conv1d

```text
一个 threadgroup 处理一个 time tile
每个 thread 处理若干 channel
```

state 更新在 kernel 内完成。

## 6. 数据类型

当前权重是 bf16。

Metal 方案：

1. 首选确认 MSL 对 `bfloat16_t` 的支持；
2. 如果不支持，加载时转换为 fp16；
3. 累加使用 f32；
4. 输出再转回 bf16/fp16。

需要单独验证：

```text
bf16 -> fp16 转换是否影响 logits/top1
```

Norm/softmax/state 保持 f32。

## 7. 命令缓冲与同步

避免每个 kernel 一个 command buffer。

推荐：

```text
一个 token / 一层一个 command buffer
或
一个 token 一个 command buffer
```

内部多个 compute encoder：

```objc
commandBuffer = [queue commandBuffer];
encoder = [commandBuffer computeCommandEncoder];
...
[encoder endEncoding];
[commandBuffer commit];
[commandBuffer waitUntilCompleted];
```

decode 时可以把整个 token 的多个 kernel 放进一个 command buffer，减少提交开销。

## 8. CMake / 构建

需要 Objective-C++：

```cmake
enable_language(OBJCXX)

add_library(llmcp_metal ...)
target_link_libraries(llmcp_metal PRIVATE
  "-framework Metal"
  "-framework Foundation"
  "-framework MetalPerformanceShaders")
```

Metal shader 编译：

```bash
xcrun -sdk macosx metal -c kernels.metal -o kernels.air
xcrun -sdk macosx metallib kernels.air -o kernels.metallib
```

CMake 中：

```cmake
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/llmcp.metallib
  COMMAND xcrun -sdk macosx metal ...
  COMMAND xcrun -sdk macosx metallib ...)
```

只在 `APPLE` 下构建；其他平台直接跳过。

## 9. 运行时入口

`llmcp-run` 增加：

```bash
--metal
--metal-lib build/llmcp.metallib
```

行为：

```text
if --metal:
  MetalModelRunner
else if --aot:
  CPU AOT dylib
else:
  hand-written CPU model
```

`llmcp-benchmark` 增加：

```bash
--metal
```

统计：

```text
prefill tokens/s
decode tokens/s
```

## 10. 正确性验证

### 10.1 单算子

Metal 测试程序：

- vector add；
- rmsnorm；
- linear；
- attention；
- delta net。

与 CPU/reference 比较：

```text
max_abs
mean_abs
```

### 10.2 模型

```bash
llmcp-run --metal --text Hello --golden build/golden_logits.bin
```

目标：

```text
top1 一致
top5 overlap 5/5
max_abs 与 CPU bf16 接近
```

### 10.3 生成

```bash
llmcp-run --metal --text 'Hello world' --max-new-tokens 4
```

目标：

```text
tokens: 0 271 760 2614
```

## 11. 性能目标

首期：

- 完整模型可运行；
- 不要求超过 CPU；
- 记录 prefill/decode 时间。

第二阶段：

- matmul 用 simdgroup GEMM；
- attention/DeltaNet 融合；
- 减少 command buffer 和 kernel launch；
- 目标 decode 明显快于 CPU 标量路径。

## 12. 里程碑

### M0：Metal 环境与最小 kernel

- `MTLCreateSystemDefaultDevice` 检查；
- vector add kernel；
- `.metallib` 构建；
- CMake 集成。

验收：

```text
Metal vector add 与 CPU 一致
```

### M1：buffer / arena / 基础算子

- MTLBuffer 封装；
- weight arena no-copy；
- embedding / add / silu_mul / rms_norm / gated_rms_norm。

验收：

```text
单算子全部通过
```

### M2：matmul / linear / lm_head

- MPS 或自定义 GEMM；
- decode GEMV；
- Q8 可选。

验收：

```text
linear/lm_head 与 CPU 对齐
```

### M3：attention / DeltaNet / causal conv

- KV cache；
- FlashAttention；
- DeltaNet state；
- conv state。

验收：

```text
完整模型 top1 / greedy 对齐
```

### M4：性能

- command buffer 合并；
- GEMM tile 调优；
- 融合 elementwise；
- Q8/Metal 版本。

验收：

```text
prefill/decode 相对 CPU 有提升
```

## 13. 风险

1. **Metal 编译器与 bf16**
   - 需要确认 MSL/GPU 是否支持 bf16；
   - 否则走 fp16 转换。

2. **8 GB arena 的 MTLBuffer 上限**
   - `maxBufferLength` 可能不足；
   - 需要分块 buffer + offset。

3. **统一内存并不等于零成本**
   - 首次 GPU 访问仍可能触发 page fault；
   - 需要预热/驻留策略。

4. **DeltaNet / attention 自定义 kernel**
   - 这是主要工程量；
   - 不能直接依赖 MPS。

5. **Metal 只能在 Apple 平台测试**
   - CI 需要 macOS runner；
   - 非 Apple 平台必须回退 CPU。

6. **数值差异**
   - fp16/bf16 转换、f32 累加、GPU 归约顺序；
   - 必须保留 CPU golden 对比。

## 14. 结论

推荐路线：

```text
llmcp 图
  -> MetalModelRunner
  -> handwritten MSL kernels
  -> .metallib
  -> MTLCommandQueue
```

而不是：

```text
llmcp -> MLIR GPU dialect -> Metal
```

因为 MLIR 没有 Apple Metal target，且 DeltaNet/attention 需要自定义 kernel。

首期先做：

```text
buffer + elementwise + rmsnorm + linear
```

再逐步接：

```text
attention -> delta net -> causal conv -> 完整模型
```

## 参考

- Apple Metal:
  <https://developer.apple.com/documentation/metal>
- Metal Shading Language:
  <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
- Metal Performance Shaders:
  <https://developer.apple.com/documentation/metalperformanceshaders>
- MLIR GPU dialect:
  <https://mlir.llvm.org/docs/Dialects/GPU/>
