"""
eval/run_swebench.py

端到端：让 agent 在 SWE-bench Lite 真题（sympy 子集）上跑 → 出 patch → 本地无 Docker 判分。
拼接 concern ①（agent 出 patch）与 concern ②（eval/swebench_local 判分）。

隔离设计：
  - 判分用主 clone（cache/sympy）；agent 在一个 `git worktree`（独立目录）里改，互不污染。
  - agent 的 shell/test 工具经 Sb39Runtime 把 conda py3.9 的 bin 前插 PATH，
    于是 `python -m pytest` 用 py3.9 + mpmath，且从 worktree 的源码树 import sympy
    （sb39 里**不装** sympy，避免 agent 的改动被 editable 安装“盖掉”）。
  - agent 只看到 problem_statement；FAIL_TO_PASS / test_patch 全程对它不可见。

消融：config_overrides={"enable_reflection": False} 即关掉 reflection 注入，
其余不变——这就是 reflection on/off 的配对实验入口。

跑法：
    GEMINI_API_KEY=... .venv/bin/python -m eval.run_swebench \
        --provider gemini --model gemini-2.5-flash -n 2
    # 关 reflection：加 --no-reflection
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import EventType, RunResult, Task
from llm.router import create_backend
from tools.runtime import LocalRuntime, RunResult as RtResult

from eval.swebench_local import (
    DEFAULT_PYBIN, DEFAULT_REPO_CACHE, Instance,
    ensure_clone, grade, load_instances,
)

AGENT_ROOT = DEFAULT_REPO_CACHE.parent / "agent_workdirs"
DEFAULT_MAX_STEPS = 25


# ---------------------------------------------------------------------------
# Runtime：把 conda py3.9 的 bin 前插 PATH，让 agent 的 python/pytest 命中 sb39
# ---------------------------------------------------------------------------

class Sb39Runtime(LocalRuntime):
    def __init__(self, pybin: str) -> None:
        self._bin_dir = str(Path(pybin).parent)

    @property
    def name(self) -> str:
        return f"local+path({self._bin_dir})"

    def exec(self, cmd: str, cwd: str | None = None, timeout: int = 30) -> RtResult:
        env = os.environ.copy()
        env["PATH"] = f"{self._bin_dir}:{env.get('PATH', '')}"
        env.pop("VIRTUAL_ENV", None)          # 别让父进程的 .venv 干扰解释器选择
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=cwd, env=env,
            )
            return RtResult(proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            return RtResult(-1, "", f"Command timed out after {timeout}s: {cmd!r}")
        except Exception as e:  # noqa: BLE001
            return RtResult(-1, "", str(e))


# ---------------------------------------------------------------------------
# 单实例：worktree → 跑 agent → 取 diff → 判分
# ---------------------------------------------------------------------------

@dataclass
class SweRecord:
    instance_id: str
    resolved: bool
    status: str
    steps: int
    steps_to_first_edit: int | None
    num_reflections: int
    total_tokens: int
    patch_lines: int
    f2p: str                 # "fail→pass" 摘要
    p2p: str
    note: str = ""

    def to_row(self) -> str:
        flag = "✓" if self.resolved else "✗"
        ste = self.steps_to_first_edit if self.steps_to_first_edit is not None else "-"
        return (
            f"  {flag} {self.instance_id:<22} {self.status:<9} "
            f"steps={self.steps:<3} 1st_edit={ste!s:<3} refl={self.num_reflections:<2} "
            f"tok={self.total_tokens:<7} patch={self.patch_lines}L  F2P {self.f2p}  P2P {self.p2p}"
            + (f"  [{self.note}]" if self.note else "")
        )


_EDIT_TOOLS = {"file_write", "file_edit", "edit"}


def _first_edit_step(log: EventLog) -> int | None:
    step = 0
    for action in log.get_actions():
        step += 1
        if action.tool_call and action.tool_call.name in _EDIT_TOOLS:
            return step
    return None


def _count_reflections(log: EventLog) -> int:
    return sum(1 for e in log.iter_events() if e.event_type == EventType.REFLECTION)


def _worktree_add(repo_dir: Path, workdir: Path, base_commit: str) -> None:
    subprocess.run(["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(workdir)],
                   capture_output=True)
    workdir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo_dir), "worktree", "add", "--detach", "-f",
                    str(workdir), base_commit], check=True, capture_output=True)


def _worktree_remove(repo_dir: Path, workdir: Path) -> None:
    subprocess.run(["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(workdir)],
                   capture_output=True)


def _capture_prediction(workdir: Path) -> str:
    subprocess.run(["git", "-C", str(workdir), "add", "-A"], capture_output=True)
    return subprocess.run(["git", "-C", str(workdir), "diff", "--cached"],
                          capture_output=True, text=True).stdout


def run_instance(
    inst: Instance, backend, *,
    enable_reflection: bool = True,
    max_steps: int = DEFAULT_MAX_STEPS,
    pybin: str = DEFAULT_PYBIN,
    cache: Path = DEFAULT_REPO_CACHE,
    log_dir: str = "./logs/swebench",
) -> SweRecord:
    from config.schema import load_config
    from entry.cli import _build_registry

    repo_dir = ensure_clone(inst.repo, cache)
    workdir = AGENT_ROOT / inst.instance_id
    _worktree_add(repo_dir, workdir, inst.base_commit)

    runtime = Sb39Runtime(pybin)
    registry = _build_registry(load_config(), runtime=runtime)
    task = Task(description=inst.problem_statement, repo_path=str(workdir), max_steps=max_steps)
    agent = Agent(backend, registry,
                  AgentConfig(max_steps=max_steps, enable_reflection=enable_reflection,
                              require_edit_before_finish=True))

    log_dir_abs = str(Path(log_dir).resolve())
    old_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        with EventLog.create(task, log_dir=log_dir_abs) as log:
            result: RunResult = agent.run(task, log)
            first_edit = _first_edit_step(log)
            n_refl = _count_reflections(log)
    finally:
        os.chdir(old_cwd)

    prediction = _capture_prediction(workdir)
    _worktree_remove(repo_dir, workdir)

    if not prediction.strip():
        g = None
        resolved, f2p, p2p, note = False, "-", "-", "agent 没产出 diff"
    else:
        g = grade(inst, prediction, pybin=pybin, cache=cache)
        resolved = g.resolved
        f2p = f"{g.f2p_before_fail}→{g.f2p_after_pass}/{g.f2p_total}"
        p2p = f"{g.p2p_after_pass}/{g.p2p_total}"
        note = g.note

    return SweRecord(
        instance_id=inst.instance_id, resolved=resolved, status=result.status.value,
        steps=result.steps_taken, steps_to_first_edit=first_edit, num_reflections=n_refl,
        total_tokens=result.total_tokens, patch_lines=len((prediction or "").splitlines()),
        f2p=f2p, p2p=p2p, note=note,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=os.environ.get("FORGE_PROVIDER", "deepseek"))
    ap.add_argument("--model", default=os.environ.get("FORGE_MODEL", "deepseek-v4-flash"))
    ap.add_argument("-n", type=int, default=2, help="跑几道 sympy 题")
    ap.add_argument("--ids", nargs="*", help="指定 instance_id（覆盖 -n）")
    ap.add_argument("--no-reflection", action="store_true", help="关掉 reflection（消融）")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--pybin", default=DEFAULT_PYBIN)
    ap.add_argument("--cache", type=Path, default=DEFAULT_REPO_CACHE)
    args = ap.parse_args()

    # thinking + effort=high 的推理很长，max_tokens 太小会把响应截断（finish_reason=length）
    backend = create_backend(provider=args.provider, model=args.model, max_tokens=16384)
    insts = load_instances(ids=args.ids, n=None if args.ids else args.n)
    enable_reflection = not args.no_reflection

    print(f"\n>>> {args.provider}/{args.model}  n={len(insts)}  "
          f"reflection={'ON' if enable_reflection else 'OFF'}\n")
    records = []
    for inst in insts:
        rec = run_instance(inst, backend, enable_reflection=enable_reflection,
                           max_steps=args.max_steps, pybin=args.pybin, cache=args.cache)
        print(rec.to_row())
        records.append(rec)

    resolved = sum(r.resolved for r in records)
    print("-" * 72)
    print(f"  resolved {resolved}/{len(records)}  "
          f"reflections fired on {sum(1 for r in records if r.num_reflections)} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
