#!/usr/bin/env python3
"""Dump HF last-token logits for the one-token llmcp-run alignment."""

import argparse
import pathlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/Users/guorui/Work/models/qwen3.5-4b")
    parser.add_argument("--text", default="Hello")
    parser.add_argument("--out", default="build/golden_logits.bin")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    ids = tokenizer.encode(args.text, add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(f"expected one token, got {ids}")
    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long)).logits[0, -1].float()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.asarray(logits.cpu().numpy(), dtype=np.float32).tofile(args.out)
    print(f"token={ids[0]} logits={logits.numel()} -> {args.out}")


if __name__ == "__main__":
    main()
