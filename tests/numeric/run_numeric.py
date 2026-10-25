#!/usr/bin/env python3
"""Numeric smoke tests for the llmcp -> linalg lowering."""

import argparse
import math
import os
import pathlib
import subprocess
import sys

import torch


def fmt(value: float) -> str:
    return repr(float(value))


def dense_2d(values) -> str:
    return "[" + ", ".join("[" + ", ".join(fmt(v) for v in row) + "]" for row in values) + "]"


def dense_1d(values) -> str:
    return "[" + ", ".join(fmt(v) for v in values) + "]"


def emit_linear(x, w) -> str:
    m, k = len(x), len(x[0])
    n = len(w)
    lines = [
        "func.func private @printF32(f32)",
        "func.func private @printNewline()",
        "",
        f'memref.global "private" constant @w : memref<{n}x{k}xbf16> = dense<{dense_2d(w)}>',
        "",
        "func.func @main() -> i32 {",
        f"  %x = arith.constant dense<{dense_2d(x)}> : tensor<{m}x{k}xbf16>",
        f"  %w = memref.get_global @w : memref<{n}x{k}xbf16>",
        f"  %0 = llmcp.linear %x, %w : tensor<{m}x{k}xbf16>, "
        f"memref<{n}x{k}xbf16> -> tensor<{m}x{n}xbf16>",
    ]
    for i in range(m):
        for j in range(n):
            lines += [
                f"  %c{i}_{j} = arith.constant {i} : index",
                f"  %c{i}_{j}b = arith.constant {j} : index",
                f"  %e{i}_{j} = tensor.extract %0[%c{i}_{j}, %c{i}_{j}b] : "
                f"tensor<{m}x{n}xbf16>",
                f"  %f{i}_{j} = arith.extf %e{i}_{j} : bf16 to f32",
                f"  func.call @printF32(%f{i}_{j}) : (f32) -> ()",
                "  func.call @printNewline() : () -> ()",
            ]
    lines += ["  %ret = arith.constant 0 : i32", "  return %ret : i32", "}", ""]
    return "\n".join(lines)


def emit_rms(x, w, eps) -> str:
    m, d = len(x), len(w)
    lines = [
        "func.func private @printF32(f32)",
        "func.func private @printNewline()",
        "",
        f'memref.global "private" constant @w : memref<{d}xbf16> = dense<{dense_1d(w)}>',
        "",
        "func.func @main() -> i32 {",
        f"  %x = arith.constant dense<{dense_2d(x)}> : tensor<{m}x{d}xbf16>",
        f"  %w = memref.get_global @w : memref<{d}xbf16>",
        f"  %0 = llmcp.rms_norm %x, %w : tensor<{m}x{d}xbf16>, "
        f"memref<{d}xbf16> -> tensor<{m}x{d}xbf16>",
    ]
    for i in range(m):
        for j in range(d):
            lines += [
                f"  %c{i}_{j} = arith.constant {i} : index",
                f"  %c{i}_{j}b = arith.constant {j} : index",
                f"  %e{i}_{j} = tensor.extract %0[%c{i}_{j}, %c{i}_{j}b] : "
                f"tensor<{m}x{d}xbf16>",
                f"  %f{i}_{j} = arith.extf %e{i}_{j} : bf16 to f32",
                f"  func.call @printF32(%f{i}_{j}) : (f32) -> ()",
                "  func.call @printNewline() : () -> ()",
            ]
    lines += ["  %ret = arith.constant 0 : i32", "  return %ret : i32", "}", ""]
    return "\n".join(lines)


def run(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def numeric_test(args, case: str):
    torch.manual_seed(0)
    work = pathlib.Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    if case == "linear":
        m, k, n = 2, 4, 3
        x = torch.randn(m, k).to(torch.bfloat16)
        w = torch.randn(n, k).to(torch.bfloat16)
        mlir = emit_linear(x.tolist(), w.tolist())
        ref = (x.float() @ w.float().T).to(torch.bfloat16).float().flatten()
    elif case == "rms":
        m, d = 2, 8
        eps = 1e-6
        x = torch.randn(m, d).to(torch.bfloat16)
        w = torch.randn(d).to(torch.bfloat16)
        mlir = emit_rms(x.tolist(), w.tolist(), eps)
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        ref = (xf * rstd * (1.0 + w.float())).to(torch.bfloat16).float().flatten()
    else:
        raise ValueError(case)

    source = work / f"{case}.mlir"
    linalg = work / f"{case}.linalg.mlir"
    llvm_mlir = work / f"{case}.llvm.mlir"
    llvm_ir = work / f"{case}.ll"
    exe = work / f"{case}.bin"
    source.write_text(mlir)

    run([args.llmcp_opt, str(source), "--llmcp-lower-to-linalg", "-o", str(linalg)])
    run([
        args.mlir_opt, str(linalg),
        "--one-shot-bufferize=bufferize-function-boundaries=1",
        "--convert-linalg-to-loops", "--lower-affine", "--convert-scf-to-cf",
        "--convert-cf-to-llvm", "--convert-arith-to-llvm",
        "--convert-math-to-llvm", "--convert-func-to-llvm",
        "--finalize-memref-to-llvm", "--reconcile-unrealized-casts",
        "-o", str(llvm_mlir),
    ])
    run([args.mlir_translate, "--mlir-to-llvmir", str(llvm_mlir), "-o", str(llvm_ir)])
    run([
        args.clang, str(llvm_ir), "-o", str(exe),
        f"-L{args.runner_lib_dir}", "-lmlir_runner_utils", "-lmlir_c_runner_utils",
    ])

    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = args.runner_lib_dir + ":" + env.get("DYLD_LIBRARY_PATH", "")
    output = subprocess.run([str(exe)], check=True, env=env, capture_output=True, text=True).stdout
    actual = [float(line) for line in output.splitlines() if line.strip()]

    if len(actual) != ref.numel():
        raise SystemExit(f"{case}: expected {ref.numel()} values, got {len(actual)}: {actual}")

    for index, (got, expected) in enumerate(zip(actual, ref.tolist())):
        if not math.isclose(got, expected, rel_tol=2e-2, abs_tol=1e-2):
            raise SystemExit(
                f"{case}[{index}]: got {got}, expected {expected}, output={output!r}")
    print(f"{case}: OK ({len(actual)} values)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llmcp-opt", required=True)
    parser.add_argument("--mlir-opt", required=True)
    parser.add_argument("--mlir-translate", required=True)
    parser.add_argument("--clang", required=True)
    parser.add_argument("--runner-lib-dir", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    numeric_test(args, "linear")
    numeric_test(args, "rms")


if __name__ == "__main__":
    main()
