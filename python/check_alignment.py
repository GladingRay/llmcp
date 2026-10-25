#!/usr/bin/env python3
"""One-token real-weight alignment against the HF Qwen3.5 reference."""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llmcp_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/Users/guorui/Work/models/qwen3.5-4b")
    parser.add_argument("--text", default="Hello")
    parser.add_argument("--max-abs", type=float, default=3.0)
    parser.add_argument("--min-top5-overlap", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    ids = tokenizer.encode(args.text, add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(f"expected one token, got {ids}")

    with torch.no_grad():
        ours = llmcp_run.forward(model, ids[0])[0]
        ref = model(torch.tensor([ids], dtype=torch.long)).logits[0, -1].float()

    diff = (ours - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ours_top = torch.topk(ours, 5).indices.tolist()
    ref_top = torch.topk(ref, 5).indices.tolist()
    overlap = len(set(ours_top) & set(ref_top))

    print(f"token: {ids[0]}")
    print(f"max_abs: {max_abs:.6f}")
    print(f"mean_abs: {mean_abs:.6f}")
    print(f"top1: ours={ours_top[0]} hf={ref_top[0]}")
    print(f"top5 overlap: {overlap}/5")
    print(f"top5 ours: {ours_top}")
    print(f"top5 hf:   {ref_top}")

    if ours_top[0] != ref_top[0]:
        raise SystemExit("top1 mismatch")
    if overlap < args.min_top5_overlap:
        raise SystemExit(f"top5 overlap {overlap} < {args.min_top5_overlap}")
    if max_abs > args.max_abs:
        raise SystemExit(f"max_abs {max_abs} > {args.max_abs}")


if __name__ == "__main__":
    main()
