#!/usr/bin/env python3
"""Numeric tests for the llmcp -> linalg lowering.

Each case builds a small MLIR module, lowers it to LLVM, executes it, and
compares the printed values against a PyTorch reference.
"""

import argparse
import math
import os
import pathlib
import subprocess

import torch


def fmt(value: float) -> str:
    return repr(float(value))


def dense_2d(values) -> str:
    return "[" + ", ".join(
        "[" + ", ".join(fmt(v) for v in row) + "]" for row in values) + "]"


def dense_1d(values) -> str:
    return "[" + ", ".join(fmt(v) for v in values) + "]"


def dense_nd(values) -> str:
    if values and isinstance(values[0], list):
        return "[" + ", ".join(dense_nd(v) for v in values) + "]"
    return dense_1d(values)


def print_tensor(name: str, rows: int, columns: int, element_type: str):
    lines = []
    for i in range(rows):
        for j in range(columns):
            lines += [
                f"  %c{i}_{j} = arith.constant {i} : index",
                f"  %c{i}_{j}b = arith.constant {j} : index",
                f"  %e{i}_{j} = tensor.extract %{name}[%c{i}_{j}, %c{i}_{j}b] : "
                f"tensor<{rows}x{columns}x{element_type}>",
            ]
            value = f"%e{i}_{j}"
            if element_type == "bf16":
                lines.append(
                    f"  %f{i}_{j} = arith.extf %e{i}_{j} : bf16 to f32")
                value = f"%f{i}_{j}"
            lines += [
                f"  func.call @printF32({value}) : (f32) -> ()",
                "  func.call @printNewline() : () -> ()",
            ]
    return lines


def wrap(body, globals_, rows: int, columns: int, element_type: str) -> str:
    lines = [
        "func.func private @printF32(f32)",
        "func.func private @printNewline()",
        "",
    ]
    lines += globals_
    lines += ["", "func.func @main() -> i32 {"]
    lines += body
    lines += print_tensor("0", rows, columns, element_type)
    lines += ["  %ret = arith.constant 0 : i32", "  return %ret : i32", "}", ""]
    return "\n".join(lines)


def emit_linear(x, w) -> str:
    m, k = len(x), len(x[0])
    n = len(w)
    body = [
        f"  %x = arith.constant dense<{dense_2d(x)}> : tensor<{m}x{k}xbf16>",
        f"  %w = memref.get_global @w : memref<{n}x{k}xbf16>",
        f"  %0 = llmcp.linear %x, %w : tensor<{m}x{k}xbf16>, "
        f"memref<{n}x{k}xbf16> -> tensor<{m}x{n}xbf16>",
    ]
    globals_ = [
        f'memref.global "private" constant @w : memref<{n}x{k}xbf16> = '
        f'dense<{dense_2d(w)}>'
    ]
    return wrap(body, globals_, m, n, "bf16")


def emit_rms(x, w) -> str:
    m, d = len(x), len(w)
    body = [
        f"  %x = arith.constant dense<{dense_2d(x)}> : tensor<{m}x{d}xbf16>",
        f"  %w = memref.get_global @w : memref<{d}xbf16>",
        f"  %0 = llmcp.rms_norm %x, %w : tensor<{m}x{d}xbf16>, "
        f"memref<{d}xbf16> -> tensor<{m}x{d}xbf16>",
    ]
    globals_ = [
        f'memref.global "private" constant @w : memref<{d}xbf16> = '
        f'dense<{dense_1d(w)}>'
    ]
    return wrap(body, globals_, m, d, "bf16")


def emit_add(a, b) -> str:
    m, n = len(a), len(a[0])
    body = [
        f"  %a = arith.constant dense<{dense_2d(a)}> : tensor<{m}x{n}xbf16>",
        f"  %b = arith.constant dense<{dense_2d(b)}> : tensor<{m}x{n}xbf16>",
        f"  %0 = llmcp.add %a, %b : (tensor<{m}x{n}xbf16>, "
        f"tensor<{m}x{n}xbf16>) -> tensor<{m}x{n}xbf16>",
    ]
    return wrap(body, [], m, n, "bf16")


def emit_silu_mul(gate, up) -> str:
    m, n = len(gate), len(gate[0])
    body = [
        f"  %g = arith.constant dense<{dense_2d(gate)}> : "
        f"tensor<{m}x{n}xbf16>",
        f"  %u = arith.constant dense<{dense_2d(up)}> : "
        f"tensor<{m}x{n}xbf16>",
        f"  %0 = llmcp.silu_mul %g, %u : (tensor<{m}x{n}xbf16>, "
        f"tensor<{m}x{n}xbf16>) -> tensor<{m}x{n}xbf16>",
    ]
    return wrap(body, [], m, n, "bf16")


def emit_embedding(tokens, table) -> str:
    rows = len(tokens)
    vocab, d = len(table), len(table[0])
    token_text = "[" + ", ".join(str(int(t)) for t in tokens) + "]"
    body = [
        f"  %tokens = arith.constant dense<{token_text}> : "
        f"tensor<{rows}xindex>",
        f"  %table = memref.get_global @table : memref<{vocab}x{d}xbf16>",
        f"  %0 = llmcp.embedding %tokens, %table : (tensor<{rows}xindex>, "
        f"memref<{vocab}x{d}xbf16>) -> tensor<{rows}x{d}xbf16>",
    ]
    globals_ = [
        f'memref.global "private" constant @table : memref<{vocab}x{d}xbf16> = '
        f'dense<{dense_2d(table)}>'
    ]
    return wrap(body, globals_, rows, d, "bf16")


def emit_gated_rms_norm(x, gate, weight, group_size) -> str:
    m, h = len(x), len(x[0])
    d = len(weight)
    body = [
        f"  %x = arith.constant dense<{dense_2d(x)}> : tensor<{m}x{h}xbf16>",
        f"  %g = arith.constant dense<{dense_2d(gate)}> : "
        f"tensor<{m}x{h}xbf16>",
        f"  %w = memref.get_global @w : memref<{d}xf32>",
        f"  %0 = llmcp.gated_rms_norm %x, %g, %w "
        f"{{group_size = {group_size} : i64, eps = 1.000000e-06 : f64}} : "
        f"(tensor<{m}x{h}xbf16>, tensor<{m}x{h}xbf16>, memref<{d}xf32>) "
        f"-> tensor<{m}x{h}xbf16>",
    ]
    globals_ = [
        f'memref.global "private" constant @w : memref<{d}xf32> = '
        f'dense<{dense_1d(weight)}>'
    ]
    return wrap(body, globals_, m, h, "bf16")


def emit_lm_head(hidden, table) -> str:
    m, h = len(hidden), len(hidden[0])
    vocab = len(table)
    body = [
        f"  %hidden = arith.constant dense<{dense_2d(hidden)}> : "
        f"tensor<{m}x{h}xbf16>",
        f"  %table = memref.get_global @table : memref<{vocab}x{h}xbf16>",
        f"  %0 = llmcp.lm_head %hidden, %table : (tensor<{m}x{h}xbf16>, "
        f"memref<{vocab}x{h}xbf16>) -> tensor<{m}x{vocab}xf32>",
    ]
    globals_ = [
        f'memref.global "private" constant @table : memref<{vocab}x{h}xbf16> = '
        f'dense<{dense_2d(table)}>'
    ]
    return wrap(body, globals_, m, vocab, "f32")


def emit_causal_conv1d(x, weight, state) -> str:
    seq, channels = len(x), len(x[0])
    kernel = len(weight[0][0])
    state_len = kernel - 1
    weight_text = "[" + ", ".join(
        "[" + dense_1d(row[0]) + "]" for row in weight) + "]"
    state_text = "[" + dense_2d(state) + "]"
    lines = [
        "func.func private @printF32(f32)",
        "func.func private @printNewline()",
        "",
        f'memref.global "private" constant @w : memref<{channels}x1x{kernel}xbf16> = '
        f'dense<{weight_text}>',
        f'memref.global "private" @state : memref<1x{state_len}x{channels}xbf16> = '
        f'dense<{state_text}>',
        "",
        "func.func @main() -> i32 {",
        f"  %x = arith.constant dense<{dense_2d(x)}> : tensor<{seq}x{channels}xbf16>",
        f"  %w = memref.get_global @w : memref<{channels}x1x{kernel}xbf16>",
        f"  %state = memref.get_global @state : memref<1x{state_len}x{channels}xbf16>",
        f"  %0 = llmcp.causal_conv1d %x, %w, %state "
        f"{{layer_index = 0 : i64, kernel_size = {kernel} : i64, "
        f'activation = "silu"}} : (tensor<{seq}x{channels}xbf16>, '
        f"memref<{channels}x1x{kernel}xbf16>, "
        f"memref<1x{state_len}x{channels}xbf16>) -> tensor<{seq}x{channels}xbf16>",
    ]
    # Output tensor.
    for t in range(seq):
        for c in range(channels):
            lines += [
                f"  %oc{t}_{c} = arith.constant {t} : index",
                f"  %oc{t}_{c}b = arith.constant {c} : index",
                f"  %oe{t}_{c} = tensor.extract %0[%oc{t}_{c}, %oc{t}_{c}b] : "
                f"tensor<{seq}x{channels}xbf16>",
                f"  %of{t}_{c} = arith.extf %oe{t}_{c} : bf16 to f32",
                f"  func.call @printF32(%of{t}_{c}) : (f32) -> ()",
                "  func.call @printNewline() : () -> ()",
            ]
    # Updated state.
    lines.append("  %lz = arith.constant 0 : index")
    for i in range(state_len):
        for c in range(channels):
            lines += [
                f"  %sc{i}_{c} = arith.constant {i} : index",
                f"  %sc{i}_{c}b = arith.constant {c} : index",
                f"  %se{i}_{c} = memref.load %state[%lz, %sc{i}_{c}, "
                f"%sc{i}_{c}b] : memref<1x{state_len}x{channels}xbf16>",
                f"  %sf{i}_{c} = arith.extf %se{i}_{c} : bf16 to f32",
                f"  func.call @printF32(%sf{i}_{c}) : (f32) -> ()",
                "  func.call @printNewline() : () -> ()",
            ]
    lines += ["  %ret = arith.constant 0 : i32", "  return %ret : i32", "}", ""]
    return "\n".join(lines)


def emit_gated_delta_net(conv, a, b, a_log, dt_bias, state) -> str:
    seq = len(conv)
    channels = len(conv[0])
    num_v = len(a[0])
    state_shape = (len(state), len(state[0]), len(state[0][0]), len(state[0][0][0]))
    lines = [
        "func.func private @printF32(f32)",
        "func.func private @printNewline()",
        "",
        f'memref.global "private" constant @alog : memref<{num_v}xf32> = '
        f'dense<{dense_nd(a_log)}>',
        f'memref.global "private" constant @dt : memref<{num_v}xbf16> = '
        f'dense<{dense_nd(dt_bias)}>',
        f'memref.global "private" @state : '
        f'memref<{state_shape[0]}x{state_shape[1]}x{state_shape[2]}x'
        f'{state_shape[3]}xf32> = dense<{dense_nd(state)}>',
        "",
        "func.func @main() -> i32 {",
        f"  %conv = arith.constant dense<{dense_2d(conv)}> : "
        f"tensor<{seq}x{channels}xbf16>",
        f"  %a = arith.constant dense<{dense_2d(a)}> : "
        f"tensor<{seq}x{num_v}xbf16>",
        f"  %b = arith.constant dense<{dense_2d(b)}> : "
        f"tensor<{seq}x{num_v}xbf16>",
        f"  %alog = memref.get_global @alog : memref<{num_v}xf32>",
        f"  %dt = memref.get_global @dt : memref<{num_v}xbf16>",
        f"  %state = memref.get_global @state : "
        f"memref<{state_shape[0]}x{state_shape[1]}x{state_shape[2]}x"
        f"{state_shape[3]}xf32>",
        f"  %0 = llmcp.gated_delta_net %conv, %a, %b, %alog, %dt, %state "
        f"{{layer_index = 0 : i64, num_k_heads = 2 : i64, "
        f"num_v_heads = {num_v} : i64, head_k_dim = 2 : i64, "
        f"head_v_dim = 2 : i64, mode = \"prefill\"}} : "
        f"(tensor<{seq}x{channels}xbf16>, tensor<{seq}x{num_v}xbf16>, "
        f"tensor<{seq}x{num_v}xbf16>, memref<{num_v}xf32>, "
        f"memref<{num_v}xbf16>, memref<{state_shape[0]}x{state_shape[1]}x"
        f"{state_shape[2]}x{state_shape[3]}xf32>) -> "
        f"tensor<{seq}x{num_v * 2}xbf16>",
    ]
    for t in range(seq):
        for c in range(num_v * 2):
            lines += [
                f"  %oc{t}_{c} = arith.constant {t} : index",
                f"  %oc{t}_{c}b = arith.constant {c} : index",
                f"  %oe{t}_{c} = tensor.extract %0[%oc{t}_{c}, %oc{t}_{c}b] : "
                f"tensor<{seq}x{num_v * 2}xbf16>",
                f"  %of{t}_{c} = arith.extf %oe{t}_{c} : bf16 to f32",
                f"  func.call @printF32(%of{t}_{c}) : (f32) -> ()",
                "  func.call @printNewline() : () -> ()",
            ]
    lines.append("  %lz = arith.constant 0 : index")
    for h in range(state_shape[1]):
        for k in range(state_shape[2]):
            for v in range(state_shape[3]):
                lines += [
                    f"  %sc{h}_{k}_{v} = arith.constant {h} : index",
                    f"  %sc{h}_{k}_{v}b = arith.constant {k} : index",
                    f"  %sc{h}_{k}_{v}c = arith.constant {v} : index",
                    f"  %se{h}_{k}_{v} = memref.load %state[%lz, "
                    f"%sc{h}_{k}_{v}, %sc{h}_{k}_{v}b, %sc{h}_{k}_{v}c] : "
                    f"memref<{state_shape[0]}x{state_shape[1]}x"
                    f"{state_shape[2]}x{state_shape[3]}xf32>",
                    f"  func.call @printF32(%se{h}_{k}_{v}) : (f32) -> ()",
                    "  func.call @printNewline() : () -> ()",
                ]
    lines += ["  %ret = arith.constant 0 : i32", "  return %ret : i32", "}", ""]
    return "\n".join(lines)


def run(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def compile_and_run(args, case: str, mlir: str, ref: torch.Tensor, tolerance):
    work = pathlib.Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    source = work / f"{case}.mlir"
    linalg = work / f"{case}.linalg.mlir"
    llvm_mlir = work / f"{case}.llvm.mlir"
    llvm_ir = work / f"{case}.ll"
    exe = work / f"{case}.bin"
    source.write_text(mlir)

    run([args.llmcp_opt, str(source), "--llmcp-lower-to-linalg", "-o",
         str(linalg)])
    run([
        args.mlir_opt, str(linalg),
        "--one-shot-bufferize=bufferize-function-boundaries=1",
        "--convert-linalg-to-loops", "--lower-affine", "--convert-scf-to-cf",
        "--convert-cf-to-llvm", "--convert-arith-to-llvm",
        "--convert-math-to-llvm", "--convert-func-to-llvm",
        "--expand-strided-metadata", "--finalize-memref-to-llvm", "--reconcile-unrealized-casts",
        "-o", str(llvm_mlir),
    ])
    run([args.mlir_translate, "--mlir-to-llvmir", str(llvm_mlir), "-o",
         str(llvm_ir)])
    run([
        args.clang, str(llvm_ir), "-o", str(exe),
        f"-L{args.runner_lib_dir}", "-lmlir_runner_utils",
        "-lmlir_c_runner_utils",
    ])

    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = args.runner_lib_dir + ":" + env.get(
        "DYLD_LIBRARY_PATH", "")
    output = subprocess.run([str(exe)], check=True, env=env,
                            capture_output=True, text=True).stdout
    actual = [float(line) for line in output.splitlines() if line.strip()]

    if len(actual) != ref.numel():
        raise SystemExit(
            f"{case}: expected {ref.numel()} values, got {len(actual)}: {actual}")

    rel_tol, abs_tol = tolerance
    for index, (got, expected) in enumerate(zip(actual, ref.tolist())):
        if not math.isclose(got, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            raise SystemExit(
                f"{case}[{index}]: got {got}, expected {expected}, "
                f"output={output!r}")
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

    torch.manual_seed(0)
    bf16 = torch.bfloat16

    # linear
    x = torch.randn(2, 4).to(bf16)
    w = torch.randn(3, 4).to(bf16)
    ref = (x.float() @ w.float().T).to(bf16).float().flatten()
    compile_and_run(args, "linear", emit_linear(x.tolist(), w.tolist()), ref,
                    (2e-2, 1e-2))

    # rms_norm
    x = torch.randn(2, 8).to(bf16)
    w = torch.randn(8).to(bf16)
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + 1e-6)
    ref = (xf * rstd * (1.0 + w.float())).to(bf16).float().flatten()
    compile_and_run(args, "rms", emit_rms(x.tolist(), w.tolist()), ref,
                    (2e-2, 1e-2))

    # add
    a = torch.randn(2, 4).to(bf16)
    b = torch.randn(2, 4).to(bf16)
    ref = (a.float() + b.float()).to(bf16).float().flatten()
    compile_and_run(args, "add", emit_add(a.tolist(), b.tolist()), ref,
                    (2e-2, 1e-2))

    # silu_mul
    gate = torch.randn(2, 4).to(bf16)
    up = torch.randn(2, 4).to(bf16)
    ref = (gate.float() * torch.sigmoid(gate.float()) * up.float()).to(
        bf16).float().flatten()
    compile_and_run(args, "silu_mul", emit_silu_mul(gate.tolist(), up.tolist()),
                    ref, (2e-2, 1e-2))

    # embedding
    table = torch.randn(4, 3).to(bf16)
    tokens = [2, 0]
    ref = table[[2, 0]].float().flatten()
    compile_and_run(args, "embedding",
                    emit_embedding(tokens, table.tolist()), ref, (2e-2, 1e-2))

    # gated_rms_norm
    m, groups, group = 2, 2, 4
    x = torch.randn(m, groups * group).to(bf16)
    gate = torch.randn(m, groups * group).to(bf16)
    weight = torch.randn(group).float()
    xf = x.float().view(m, groups, group)
    gf = gate.float().view(m, groups, group)
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = (xf * rstd * weight.float() * torch.nn.functional.silu(gf)).view(
        m, groups * group).to(bf16).float().flatten()
    compile_and_run(args, "gated_rms_norm",
                    emit_gated_rms_norm(x.tolist(), gate.tolist(),
                                        weight.tolist(), group),
                    ref, (2e-2, 1e-2))

    # lm_head
    hidden = torch.randn(2, 3).to(bf16)
    table = torch.randn(4, 3).to(bf16)
    ref = (hidden.float() @ table.float().T).flatten()
    compile_and_run(args, "lm_head", emit_lm_head(hidden.tolist(),
                                                  table.tolist()),
                    ref, (1e-3, 1e-3))

    # causal_conv1d (prefill + state update)
    seq, channels, kernel = 3, 4, 3
    state_len = kernel - 1
    x = torch.randn(seq, channels).to(bf16)
    weight = torch.randn(channels, 1, kernel).to(bf16)
    state = torch.randn(1, state_len, channels).to(bf16)
    padded = torch.cat([state[0], x], dim=0)
    out = torch.empty(seq, channels, dtype=bf16)
    for t in range(seq):
        for c in range(channels):
            acc = 0.0
            for k in range(kernel):
                acc += float(padded[t + k, c].float()) * float(
                    weight[c, 0, k].float())
            out[t, c] = torch.tensor(acc).to(bf16)
    out = (out.float() * torch.sigmoid(out.float())).to(bf16)
    new_state = padded[-(state_len):]
    ref = torch.cat([out.float().flatten(),
                     new_state.float().flatten()])
    compile_and_run(args, "causal_conv1d",
                    emit_causal_conv1d(x.tolist(), weight.tolist(),
                                       state[0].tolist()),
                    ref, (2e-2, 1e-2))

    # causal_conv1d decode path: seq_len = 1, state shifts by one.
    seq = 1
    x = torch.randn(seq, channels).to(bf16)
    state = torch.randn(1, state_len, channels).to(bf16)
    padded = torch.cat([state[0], x], dim=0)
    out = torch.empty(seq, channels, dtype=bf16)
    for t in range(seq):
        for c in range(channels):
            acc = 0.0
            for k in range(kernel):
                acc += float(padded[t + k, c].float()) * float(
                    weight[c, 0, k].float())
            out[t, c] = torch.tensor(acc).to(bf16)
    out = (out.float() * torch.sigmoid(out.float())).to(bf16)
    new_state = padded[-(state_len):]
    ref = torch.cat([out.float().flatten(), new_state.float().flatten()])
    compile_and_run(args, "causal_conv1d_decode",
                    emit_causal_conv1d(x.tolist(), weight.tolist(),
                                       state[0].tolist()),
                    ref, (2e-2, 1e-2))

    # gated_delta_net: small recurrent reference.
    seq, num_k, num_v, head_k, head_v = 2, 2, 4, 2, 2
    channels = 2 * num_k * head_k + num_v * head_v
    conv = torch.randn(seq, channels).to(bf16)
    a = torch.randn(seq, num_v).to(bf16)
    b = torch.randn(seq, num_v).to(bf16)
    a_log = torch.randn(num_v).float()
    dt_bias = torch.randn(num_v).to(bf16)
    state = torch.randn(1, num_v, head_k, head_v).float()

    q = conv[:, : num_k * head_k].float().view(seq, num_k, head_k)
    k = conv[:, num_k * head_k : 2 * num_k * head_k].float().view(
        seq, num_k, head_k)
    v = conv[:, 2 * num_k * head_k :].float().view(seq, num_v, head_v)
    ref_state = state.clone()
    ref_out = torch.empty(seq, num_v * head_v, dtype=bf16)
    for t in range(seq):
        for h in range(num_v):
            kh = h // (num_v // num_k)
            qh = q[t, kh]
            khv = k[t, kh]
            qh = qh * torch.rsqrt((qh * qh).sum() + 1e-6)
            khv = khv * torch.rsqrt((khv * khv).sum() + 1e-6)
            qh = qh / (head_k ** 0.5)
            beta = torch.sigmoid(b[t, h].float())
            gate = -torch.exp(a_log[h]) * torch.nn.functional.softplus(
                a[t, h].float() + dt_bias[h].float())
            decay = torch.exp(gate)
            ref_state[0, h] = ref_state[0, h] * decay
            kv = (ref_state[0, h] * khv.unsqueeze(-1)).sum(dim=0)
            delta = (v[t, h] - kv) * beta
            ref_state[0, h] = ref_state[0, h] + khv.unsqueeze(-1) * delta
            o = (ref_state[0, h] * qh.unsqueeze(-1)).sum(dim=0)
            ref_out[t, h * head_v : (h + 1) * head_v] = o.to(bf16)
    ref = torch.cat([ref_out.float().flatten(), ref_state.flatten()])
    compile_and_run(args, "gated_delta_net",
                    emit_gated_delta_net(conv.tolist(), a.tolist(), b.tolist(),
                                         a_log.tolist(), dt_bias.tolist(),
                                         state.tolist()),
                    ref, (2e-2, 1e-2))

    # gated_delta_net decode path: one token, state update only.
    seq = 1
    conv = torch.randn(seq, channels).to(bf16)
    a = torch.randn(seq, num_v).to(bf16)
    b = torch.randn(seq, num_v).to(bf16)
    state = torch.randn(1, num_v, head_k, head_v).float()
    q = conv[:, : num_k * head_k].float().view(seq, num_k, head_k)
    k = conv[:, num_k * head_k : 2 * num_k * head_k].float().view(
        seq, num_k, head_k)
    v = conv[:, 2 * num_k * head_k :].float().view(seq, num_v, head_v)
    ref_state = state.clone()
    ref_out = torch.empty(seq, num_v * head_v, dtype=bf16)
    for t in range(seq):
        for h in range(num_v):
            kh = h // (num_v // num_k)
            qh = q[t, kh]
            khv = k[t, kh]
            qh = qh * torch.rsqrt((qh * qh).sum() + 1e-6)
            khv = khv * torch.rsqrt((khv * khv).sum() + 1e-6)
            qh = qh / (head_k ** 0.5)
            beta = torch.sigmoid(b[t, h].float())
            gate = -torch.exp(a_log[h]) * torch.nn.functional.softplus(
                a[t, h].float() + dt_bias[h].float())
            decay = torch.exp(gate)
            ref_state[0, h] = ref_state[0, h] * decay
            kv = (ref_state[0, h] * khv.unsqueeze(-1)).sum(dim=0)
            delta = (v[t, h] - kv) * beta
            ref_state[0, h] = ref_state[0, h] + khv.unsqueeze(-1) * delta
            o = (ref_state[0, h] * qh.unsqueeze(-1)).sum(dim=0)
            ref_out[t, h * head_v : (h + 1) * head_v] = o.to(bf16)
    ref = torch.cat([ref_out.float().flatten(), ref_state.flatten()])
    compile_and_run(args, "gated_delta_net_decode",
                    emit_gated_delta_net(conv.tolist(), a.tolist(), b.tolist(),
                                         a_log.tolist(), dt_bias.tolist(),
                                         state.tolist()),
                    ref, (2e-2, 1e-2))


if __name__ == "__main__":
    main()
