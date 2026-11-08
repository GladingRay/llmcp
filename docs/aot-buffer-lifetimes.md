# AOT 中间缓冲区生命周期

## 问题与修复

One-Shot Bufferize 会为算子结果和临时工作区生成堆分配，但不会自动插入
释放操作。原来的 AOT 流水线直接进入 LLVM lowering，导致连续 decode
时中间缓冲区持续累积。

现在在 bufferization 之后、循环和 OpenMP lowering 之前依次执行：

```text
one-shot-bufferize=bufferize-function-boundaries=1
inline
symbol-dce
buffer-deallocation
```

先内联函数调用，使 embedding、hidden、lm_head 等跨函数返回的分配进入
最终消费它们的入口。仅添加旧版 `buffer-deallocation` 而不内联，无法处理
这些调用返回值的释放。私有辅助函数随后由 `symbol-dce` 清除。

LLVM 20 的 `buffer-deallocation-pipeline` 使用基于所有权的分析，但会拒绝
本项目的 `llvm.call_intrinsic`（bfdot/sdot），因为该操作没有声明内存副作用
接口。因此目前使用 LLVM 20 的旧版 `buffer-deallocation`，处理内联后的
结构化控制流。升级 LLVM 或引入新的控制流、调用与别名形式时，需要重新验证。

`llmcp_model_prefill` 和 `llmcp_model_decode` 的输入、权重 arena、KV cache、
DeltaNet state、卷积状态和输出均由调用者持有，生成代码只借用这些缓冲区。
中间分配在最后一次使用后释放；输出复制到调用者的 logits 缓冲区后，其临时
存储也会释放。保留的张量返回入口 `llmcp_model` 不同：其返回缓冲区的所有权
交给调用者，不属于这两个无返回值 AOT 入口的生命周期契约。

## 回归验证

`ctest --test-dir build -R llmcp-numeric --output-on-failure` 会执行算子数值
测试和跨函数生命周期测试。测试把生成 LLVM IR 的 `malloc/free` 调用重定向到
计数包装函数；程序退出时若仍有未释放分配，测试直接失败。计数为原子操作，
支持 OpenMP 内核，不统计 OpenMP 和 runner 库自身的内存。

跨函数测试重复调用入口 32 次，经过两次返回张量的辅助函数调用，将结果写入
借用的输出缓冲区，并验证结果。其他算子用例同时覆盖 attention、DeltaNet、
卷积和 bf16/Q8 matmul 的工作区。

本次修复在本机 LLVM 20 上验证了：

- lit、numeric、单元测试以及 bf16/Q8 真实模型 LLVM IR 编译均通过；
- 将新生成的 bf16/Q8 完整模型 IR 接入同一计数包装函数后，两条路径的
  4-token golden 均为 `0 271 760 2614`，退出时未释放分配数均为零；
- 在跨函数回归用例中移除内联步骤后，测试准确报告 64 个未释放分配。

已有 AOT 动态库不会随工具构建自动更新，使用修复时需要重新执行
`llmcp-compile --emit=shared` 生成模型库。
