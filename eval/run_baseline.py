"""
eval/run_baseline.py

「无脚手架」基线：同一个模型（默认 deepseek-v4-flash）对 SWE-bench 实例做**一次**
completion 直接出补丁，没有 ReAct 循环 / 工具 / repo-map / 反思。

目的：和 eval/run_swebench.py（完整 agent）对照，量化脚手架到底有没有提升能力。
  - 两边同模型、同判分器（eval/swebench_local.grade，无 Docker）。
  - 唯一区别就是「有没有 agent 这套」。

公平性：base 给的是 **oracle 定位**——直接把 gold patch 改到的文件内容喂给模型，
它只需要"改对"，不用自己找文件。这对 base 偏宽容；agent 则要自己定位。
所以 agent 若能追平/超过，说明脚手架（探索+测试反馈+反思）确有增量。

模型按 SEARCH/REPLACE 块输出（比整文件回吐省 token，能处理大文件）：
    ### FILE: <path>
    <<<<<<< SEARCH
    <原文片段，需逐字匹配>
    =======
    <替换后内容>
    >>>>>>> REPLACE

跑法：
    DEEPSEEK_API_KEY=... .venv/bin/python -m eval.run_baseline -n 2
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from eval.swebench_local import (
    DEFAULT_PYBIN, DEFAULT_REPO_CACHE, Instance,
    checkout_base, ensure_clone, grade, load_instances,
)

PROMPT = """You are fixing a bug in the {repo} project.

## Issue
{issue}

## File(s) to edit (current content)
{files}

## Your task
Edit the file(s) above to resolve the issue. Reply ONLY with one or more
SEARCH/REPLACE blocks, no prose:

### FILE: <path>
<<<<<<< SEARCH
<exact lines from the current content to replace>
=======
<the replacement lines>
>>>>>>> REPLACE

The SEARCH text must match the file exactly (whitespace included). Make the
smallest change that fixes the issue. Only touch files that need changing.
"""

_BLOCK = re.compile(
    r"### FILE:\s*(?P<path>.+?)\n<<<<<<< SEARCH\n(?P<search>.*?)\n=======\n"
    r"(?P<replace>.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)


@dataclass
class BaselineResult:
    instance_id: str
    resolved: bool
    applied: int           # 成功应用的 SEARCH/REPLACE 块数
    f2p: str
    p2p: str
    tokens: int
    note: str = ""

    def to_row(self) -> str:
        flag = "✓" if self.resolved else "✗"
        return (f"  {flag} {self.instance_id:<22} applied={self.applied:<2} "
                f"tok={self.tokens:<7} F2P {self.f2p}  P2P {self.p2p}"
                + (f"  [{self.note}]" if self.note else ""))


def _files_in_patch(patch: str) -> list[str]:
    return [f for f in re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.M) if f.endswith(".py")]


def _make_client():
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not set (source .secrets.env)")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def _one_shot(client, model: str, repo: str, issue: str,
              files: dict[str, str]) -> tuple[str, int]:
    """一次 completion，返回 (raw_output, total_tokens)。"""
    blocks = "\n".join(f"### FILE: {p}\n```python\n{c}\n```" for p, c in files.items())
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": PROMPT.format(repo=repo, issue=issue, files=blocks)}],
        max_tokens=16384,
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="high",
    )
    out = r.choices[0].message.content or ""
    return out, (r.usage.total_tokens if r.usage else 0)


def _apply_blocks(repo_dir: Path, raw: str) -> int:
    """把模型输出的 SEARCH/REPLACE 应用到工作区，返回成功块数。"""
    applied = 0
    for m in _BLOCK.finditer(raw):
        rel, search, replace = m.group("path").strip(), m.group("search"), m.group("replace")
        path = repo_dir / rel
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        if search and content.count(search) >= 1:
            path.write_text(content.replace(search, replace, 1), encoding="utf-8")
            applied += 1
    return applied


def run_baseline_instance(inst: Instance, client, model: str, *,
                          pybin: str = DEFAULT_PYBIN,
                          cache: Path = DEFAULT_REPO_CACHE) -> BaselineResult:
    repo_dir = ensure_clone(inst.repo, cache)
    checkout_base(repo_dir, inst.base_commit)

    target = _files_in_patch(inst.patch)                 # oracle 定位：gold 改的文件
    files = {p: (repo_dir / p).read_text(encoding="utf-8")
             for p in target if (repo_dir / p).exists()}
    if not files:
        return BaselineResult(inst.instance_id, False, 0, "-", "-", 0, "无法读取目标文件")

    raw, tokens = _one_shot(client, model, inst.repo, inst.problem_statement, files)
    applied = _apply_blocks(repo_dir, raw)
    prediction = subprocess.run(["git", "-C", str(repo_dir), "diff"],
                                capture_output=True, text=True).stdout

    if not prediction.strip():
        return BaselineResult(inst.instance_id, False, applied, "-", "-", tokens,
                              "模型没产出可应用的改动")

    g = grade(inst, prediction, pybin=pybin, cache=cache)
    return BaselineResult(
        inst.instance_id, g.resolved, applied,
        f"{g.f2p_before_fail}→{g.f2p_after_pass}/{g.f2p_total}",
        f"{g.p2p_after_pass}/{g.p2p_total}", tokens, g.note,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FORGE_MODEL", "deepseek-v4-flash"))
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--pybin", default=DEFAULT_PYBIN)
    ap.add_argument("--cache", type=Path, default=DEFAULT_REPO_CACHE)
    args = ap.parse_args()

    client = _make_client()
    insts = load_instances(ids=args.ids, n=None if args.ids else args.n)
    print(f"\n>>> BASELINE (no scaffold, one-shot)  {args.model}  n={len(insts)}\n")
    records = []
    for inst in insts:
        rec = run_baseline_instance(inst, client, args.model, pybin=args.pybin, cache=args.cache)
        print(rec.to_row())
        records.append(rec)
    resolved = sum(r.resolved for r in records)
    print("-" * 60)
    print(f"  baseline resolved {resolved}/{len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
