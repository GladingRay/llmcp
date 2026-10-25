#!/usr/bin/env python3
"""Compare the handwritten tokenizer against known HF token ids."""

import argparse
import subprocess

CASES = [
    ("Hello", "9419"),
    ("Hello world", "9419,1814"),
    ("  leading", "220,6187"),
    ("Hello\nworld", "9419,198,14194"),
    ("def foo(x):\n    return x+1",
     "727,14785,2007,1590,198,262,460,830,10,16"),
    ("你好", "109266"),
    ("北京", "98116"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llmcp-run", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    for text, expected in CASES:
        result = subprocess.run(
            [args.llmcp_run, "--tokenize-only", "--text", text,
             "--tokenizer", args.tokenizer],
            check=True, capture_output=True, text=True)
        got = result.stdout.strip()
        if got != expected:
            raise SystemExit(f"{text!r}: got {got}, expected {expected}")
        print(f"OK {text!r}: {got}")


if __name__ == "__main__":
    main()
