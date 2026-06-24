"""
eval/run_real.py

用真实 LLM 在本地合成任务上跑评测 harness，拿第一个真实 resolve 数。

跑法：
    GEMINI_API_KEY=... PATH="$PWD/.venv/bin:$PATH" \
        .venv/bin/python -m eval.run_real --provider gemini --model gemini-2.5-flash -n 3
"""

from __future__ import annotations

import argparse
import sys

from llm.router import create_backend
from eval.harness import run_case, print_report
from eval.local_cases import build_local_cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("-n", type=int, default=3, help="跑几个 case")
    args = ap.parse_args()

    backend = create_backend(provider=args.provider, model=args.model, max_tokens=4096)
    cases = build_local_cases(args.n)

    print(f"\n>>> provider={args.provider} model={args.model}  cases={len(cases)}\n")
    records = []
    for c in cases:
        rec = run_case(c, backend)
        print(rec.to_row())
        records.append(rec)

    print()
    print_report(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
