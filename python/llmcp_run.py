#!/usr/bin/env python3
"""Minimal reference executor for the llmcp Qwen3.5 graph.

This is a semantic oracle used for real-weight alignment before the C++ AOT
runtime lands. It intentionally follows the llmcp op semantics: zero-centered
RMSNorm, causal conv, gated delta recurrence, partial RoPE, GQA, and tied
lm_head.
"""

import argparse
import json
import math
import pathlib
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def rms_norm(x, weight, eps):
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(variance + eps) * (1.0 + weight.float())).to(x.dtype)


def rms_norm_gated(x, gate, weight, eps):
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    normalized = xf * torch.rsqrt(variance + eps)
    normalized = normalized.to(x.dtype)
    gated = (weight * normalized) * F.silu(gate.float())
    return gated.to(x.dtype)


def l2norm(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)


def linear(x, weight):
    return F.linear(x, weight)


def mlp(x, prefix, sd):
    gate = linear(x, sd[f"{prefix}.mlp.gate_proj.weight"])
    up = linear(x, sd[f"{prefix}.mlp.up_proj.weight"])
    return linear(F.silu(gate) * up, sd[f"{prefix}.mlp.down_proj.weight"])


def full_attention(x, prefix, sd):
    # One-token prefill: the causal attention softmax has a single entry, so
    # the attention context for every q head is the corresponding v head.
    qg = linear(x, sd[f"{prefix}.self_attn.q_proj.weight"])
    v = linear(x, sd[f"{prefix}.self_attn.v_proj.weight"])
    cfg = sd["_cfg"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    dim = cfg["head_dim"]
    qg = qg.view(1, 1, heads, 2 * dim)
    gate = qg[..., dim:].reshape(1, 1, heads * dim)
    v = v.view(1, 1, kv_heads, dim)
    group = heads // kv_heads
    context = v.repeat_interleave(group, dim=2).reshape(1, 1, heads * dim)
    gated = context * torch.sigmoid(gate)
    return linear(gated, sd[f"{prefix}.self_attn.o_proj.weight"])


def linear_attention(x, prefix, sd):
    cfg = sd["_cfg"]
    num_k = cfg["linear_num_key_heads"]
    num_v = cfg["linear_num_value_heads"]
    head_k = cfg["linear_key_head_dim"]
    head_v = cfg["linear_value_head_dim"]
    kernel = cfg["linear_conv_kernel_dim"]
    eps = cfg["rms_norm_eps"]

    qkv = linear(x, sd[f"{prefix}.linear_attn.in_proj_qkv.weight"])
    z = linear(x, sd[f"{prefix}.linear_attn.in_proj_z.weight"])
    a = linear(x, sd[f"{prefix}.linear_attn.in_proj_a.weight"])
    b = linear(x, sd[f"{prefix}.linear_attn.in_proj_b.weight"])

    # Causal conv with a zero initial state and seq_len=1: only the last tap
    # contributes; the rest is the zero left padding.
    conv_w = sd[f"{prefix}.linear_attn.conv1d.weight"]  # [C, 1, K]
    conv = qkv * conv_w[:, 0, -1]
    conv = F.silu(conv)

    q, k, v = torch.split(conv, [num_k * head_k, num_k * head_k, num_v * head_v],
                          dim=-1)
    q = q.view(1, num_k, head_k)
    k = k.view(1, num_k, head_k)
    v = v.view(1, num_v, head_v)
    q = l2norm(q) / math.sqrt(head_k)
    k = l2norm(k)
    q = q.repeat_interleave(num_v // num_k, dim=1)
    k = k.repeat_interleave(num_v // num_k, dim=1)

    beta = torch.sigmoid(b.view(1, num_v, 1))
    gate = -sd[f"{prefix}.linear_attn.A_log"].float().exp() * F.softplus(
        a.float() + sd[f"{prefix}.linear_attn.dt_bias"].float())
    decay = gate.exp().view(1, num_v, 1)

    # State starts at zero: S = k^T ((v - 0) * beta), o = S^T q.
    state = k.unsqueeze(-1) * (v * beta).unsqueeze(-2)
    state = state * decay.unsqueeze(-1)
    out = (state * q.unsqueeze(-1)).sum(dim=2).to(torch.bfloat16)
    out = out.reshape(1, 1, num_v * head_v)

    z = z.view(1, 1, num_v, head_v)
    norm_w = sd[f"{prefix}.linear_attn.norm.weight"]
    out = out.view(1, 1, num_v, head_v)
    out = rms_norm_gated(out, z, norm_w, eps)
    out = out.reshape(1, 1, num_v * head_v)
    return linear(out, sd[f"{prefix}.linear_attn.out_proj.weight"])


def forward(model, token_id):
    sd = model.state_dict()
    cfg = model.config
    sd["_cfg"] = {
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "linear_num_key_heads": cfg.linear_num_key_heads,
        "linear_num_value_heads": cfg.linear_num_value_heads,
        "linear_key_head_dim": cfg.linear_key_head_dim,
        "linear_value_head_dim": cfg.linear_value_head_dim,
        "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
        "rms_norm_eps": cfg.rms_norm_eps,
    }
    hidden = sd["model.embed_tokens.weight"][token_id].view(1, 1, -1)
    for layer in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        residual = hidden
        norm = rms_norm(hidden, sd[f"{prefix}.input_layernorm.weight"],
                        cfg.rms_norm_eps)
        if cfg.layer_types[layer] == "linear_attention":
            mixed = linear_attention(norm, prefix, sd)
        else:
            mixed = full_attention(norm, prefix, sd)
        hidden = residual + mixed
        residual = hidden
        norm = rms_norm(hidden, sd[f"{prefix}.post_attention_layernorm.weight"],
                        cfg.rms_norm_eps)
        hidden = residual + mlp(norm, prefix, sd)
    hidden = rms_norm(hidden, sd["model.norm.weight"],
                      cfg.rms_norm_eps)
    logits = F.linear(hidden.view(1, -1),
                      sd["model.embed_tokens.weight"])
    return logits.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/Users/guorui/Work/models/qwen3.5-4b")
    parser.add_argument("--text", default="Hello")
    parser.add_argument("--compare-hf", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    ids = tokenizer.encode(args.text, add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(f"this minimal runner expects one token, got {ids}")
    with torch.no_grad():
        ours = forward(model, ids[0])
        if args.compare_hf:
            ref = model(torch.tensor([ids], dtype=torch.long)).logits[0, -1].float()
            diff = (ours[0] - ref).abs()
            print(f"token={ids[0]} max_abs={diff.max().item():.6f} "
                  f"mean_abs={diff.mean().item():.6f}")
            print("top5 ours", torch.topk(ours[0], 5).indices.tolist())
            print("top5 hf  ", torch.topk(ref, 5).indices.tolist())
        else:
            print("top5", torch.topk(ours[0], 5).indices.tolist())


if __name__ == "__main__":
    main()
