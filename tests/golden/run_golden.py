#!/usr/bin/env python3
"""Run llmcp-run and compare generated token ids with a golden file."""

import argparse
import pathlib
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llmcp-run", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--aot", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--text", default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    expected_path = pathlib.Path(args.expected)
    expected = expected_path.read_text().split()
    if not expected:
        print(f"empty golden file: {expected_path}")
        return 1

    command = [
        args.llmcp_run,
        "--weights", args.weights,
        "--aot", args.aot,
        "--tokenizer", args.tokenizer,
        "--text", args.text,
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print(f"llmcp-run failed with {result.returncode}")
        return 1

    actual = None
    for line in result.stdout.splitlines():
        if line.startswith("tokens:"):
            actual = line.split(":", 1)[1].split()
    if actual is None:
        print("llmcp-run produced no tokens: line")
        return 1
    if actual != expected:
        print(f"golden mismatch: expected={expected} actual={actual}")
        return 1
    print(f"golden OK: {' '.join(actual)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
