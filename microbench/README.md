# CPU microbenchmark

`llmcp-microbench` 用 FMA / dot-product 热循环测本机 CPU 的峰值吞吐，并用大缓冲区测内存带宽。

## 构建

```bash
cmake --build build -j 8
```

## 运行

```bash
./build/bin/llmcp-microbench --seconds 1 --buffer-mb 256 --threads 8
```

参数：

- `--seconds`：每个 compute case 的采样时间；
- `--buffer-mb`：带宽测试缓冲区大小；
- `--threads`：带宽测试线程数，0 表示使用全部逻辑核。

## 覆盖范围

- `scalar_f32`：标量 FP32 FMA；
- `neon_f32`：NEON 4-lane FP32 FMA；
- `neon_f16`：NEON 8-lane FP16 FMA（需要 FEAT_FP16）；
- `neon_bf16_dot`：BF16 dot-product（需要 FEAT_BF16）；
- `neon_i8_dot`：INT8 dot-product（需要 FEAT_DotProd）；
- `read` / `write` / `copy`：多线程内存带宽。

CPU feature 通过 `sysctlbyname("hw.optional.arm.FEAT_*")` 检测。

> I8MM 只做 feature 检测，当前没有单独测 `vmmlaq_s32` 的矩阵吞吐；
> Apple AMX 没有公开的编译器 intrinsic，因此不纳入。
