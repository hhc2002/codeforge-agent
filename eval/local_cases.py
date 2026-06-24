"""
eval/local_cases.py

本地合成 bug-fix 任务集（不依赖 SWE-bench / Docker）。
每个 case = 一个含 bug 的小模块 + 一个隐藏测试 + 任务描述。
用于在上 SWE-bench 之前，先用真实 LLM 验证 Agent 真能修 bug。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from eval.harness import EvalCase

# 每个 spec: id -> (源文件名, 含bug内容, 测试文件名, 测试内容, 任务描述)
_SPECS = [
    (
        "arith-add",
        "mathutils.py", "def add(a, b):\n    return a - b  # bug\n",
        "test_mathutils.py",
        "from mathutils import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n",
        "add() in mathutils.py returns wrong results; test_mathutils.py fails. Fix the bug.",
    ),
    (
        "parity-iseven",
        "parity.py", "def is_even(n):\n    return n % 2 == 1  # bug\n",
        "test_parity.py",
        "from parity import is_even\n\n\ndef test_is_even():\n    assert is_even(4) is True\n    assert is_even(3) is False\n",
        "is_even() in parity.py is inverted; test_parity.py fails. Fix it.",
    ),
    (
        "list-last",
        "listops.py", "def last(items):\n    return items[0]  # bug: should be last element\n",
        "test_listops.py",
        "from listops import last\n\n\ndef test_last():\n    assert last([1, 2, 3]) == 3\n    assert last(['a', 'b']) == 'b'\n",
        "last() in listops.py returns the first element instead of the last; test_listops.py fails. Fix it.",
    ),
]


def _git_init(d: Path) -> None:
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.email=e@e.com", "-c", "user.name=e", "commit", "-qm", "buggy"]):
        subprocess.run(["git", *args], cwd=d, capture_output=True)


def build_local_cases(n: int | None = None) -> list[EvalCase]:
    specs = _SPECS if n is None else _SPECS[:n]
    cases: list[EvalCase] = []
    for cid, src, buggy, tname, tcontent, problem in specs:
        d = Path(tempfile.mkdtemp(prefix=f"forge_{cid}_"))
        (d / src).write_text(buggy, encoding="utf-8")
        (d / tname).write_text(tcontent, encoding="utf-8")
        _git_init(d)
        cases.append(EvalCase(
            case_id=cid,
            repo_path=str(d),
            problem_statement=problem,
            verify_cmd=f"python -m pytest {tname} -q",
            max_steps=12,
        ))
    return cases
