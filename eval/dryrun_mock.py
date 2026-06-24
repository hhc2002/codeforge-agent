"""
eval/dryrun_mock.py

用 MockBackend 在本地合成任务上干跑评测 harness 全链路（零 API / 零 Docker）。
验证：建工作区 → Agent 跑 → git diff 抓 patch → verify 判分 → 出指标。

含正反两个对照：
- fix-case ：mock 写入正确修复 → 期望 resolved=True
- noop-case：mock 不改代码直接 finish → 期望 resolved=False（证明判分能区分对错）

跑法：  .venv/bin/python -m eval.dryrun_mock
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from agent.task import Action, ActionType, ToolCall
from llm.base import MockBackend
from eval.harness import EvalCase, run_case, print_report

BUGGY = "def add(a, b):\n    return a - b  # bug: should be +\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from mathutils import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
PROBLEM = (
    "The function add() in mathutils.py returns a wrong result and "
    "test_mathutils.py fails. Find and fix the bug."
)


def _make_workdir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="forge_eval_"))
    (d / "mathutils.py").write_text(BUGGY, encoding="utf-8")
    (d / "test_mathutils.py").write_text(TEST, encoding="utf-8")
    # git init 让 _get_git_diff 能抓到 patch
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.email=e@e.com",
                 "-c", "user.name=e", "commit", "-qm", "buggy"]):
        subprocess.run(["git", *args], cwd=d, capture_output=True)
    return d


def _case(workdir: Path) -> EvalCase:
    return EvalCase(
        case_id=workdir.name,
        repo_path=str(workdir),
        problem_statement=PROBLEM,
        verify_cmd="python -m pytest test_mathutils.py -q",
        max_steps=10,
    )


def _fix_script(workdir: Path) -> list[Action]:
    p = str(workdir / "mathutils.py")
    return [
        Action(ActionType.TOOL_CALL, "read the buggy file",
               tool_call=ToolCall("file_read", {"path": p})),
        Action(ActionType.TOOL_CALL, "rewrite with the fix",
               tool_call=ToolCall("file_write", {"path": p, "content": FIXED})),
        Action(ActionType.FINISH, "fixed add() to use +", message="Fixed."),
    ]


def _noop_script() -> list[Action]:
    return [Action(ActionType.FINISH, "did nothing", message="No change.")]


def main() -> int:
    records = []

    wd1 = _make_workdir()
    records.append(run_case(_case(wd1), MockBackend(_fix_script(wd1))))
    records[-1].case_id = "fix-case"

    wd2 = _make_workdir()
    records.append(run_case(_case(wd2), MockBackend(_noop_script())))
    records[-1].case_id = "noop-case"

    print_report(records)

    fix_ok = records[0].resolved is True
    noop_ok = records[1].resolved is False
    print("\n[assert] fix-case resolved == True :", fix_ok)
    print("[assert] noop-case resolved == False:", noop_ok)
    if fix_ok and noop_ok:
        print("\n✅ 管道打通：建工作区→Agent→patch→判分→指标，且判分能区分对错。")
        return 0
    print("\n❌ 管道有问题，见上。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
