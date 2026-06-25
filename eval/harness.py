"""
eval/harness.py

评测 harness 的「管道层」：与具体任务来源（本地合成 / SWE-bench）解耦。
职责：给定一个 EvalCase + 一个 LLMBackend，
    1. 在隔离工作目录里跑 Agent
    2. 抓取 git diff 作为 prediction patch
    3. 用 case 自带的 verify 命令判分（pass/fail）
    4. 从 EventLog + RunResult 提取指标

设计原则：
- 对 backend 完全解耦——MockBackend 用来零成本打通管道，真实 LLM 用来出 resolve rate
- 判分用 sys.executable，避免子进程 python 没装 pytest 的坑
- 一条 case 一个独立 EventLog，可回放
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import ActionType, RunResult, RunStatus, Task
from llm.base import LLMBackend


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """一个评测样例。"""
    case_id: str
    repo_path: str                 # 工作目录（已 setup 好的代码库）
    problem_statement: str         # 喂给 Agent 的任务描述
    verify_cmd: str                # 判分命令，returncode==0 视为 resolved
    max_steps: int = 15

    # ── 诊断 metadata（分桶消融用，默认空，不影响 run_case 逻辑）─────────────
    bucket: str = ""                       # sanity | repo_map | reflection | noise ...
    source: str = ""                       # handwritten | quixbugs | swebench-lite
    source_id: str = ""                    # 原始 instance id / 程序名（provenance）
    why_this_case: str = ""                # 这道题为什么存在
    expected_capability: str = ""          # 需要 agent 的什么能力
    expected_without_component: str = ""   # 关掉目标模块时预测的失败信号（可证伪）


@dataclass
class RunRecord:
    """单条 case 的运行结果 + 指标。"""
    case_id: str
    resolved: bool
    status: str                    # RunStatus.value
    steps: int
    steps_to_first_edit: int | None
    total_tokens: int
    patch_lines: int
    verify_output: str = ""
    bucket: str = ""               # 来自 EvalCase.bucket，便于按桶聚合

    def to_row(self) -> str:
        ste = self.steps_to_first_edit if self.steps_to_first_edit is not None else "-"
        flag = "✓" if self.resolved else "✗"
        return (
            f"  {flag} {self.case_id:<24} status={self.status:<9} "
            f"steps={self.steps:<3} 1st_edit={ste!s:<3} "
            f"tok={self.total_tokens:<7} patch={self.patch_lines}L"
        )


# ---------------------------------------------------------------------------
# 核心：跑一条 case
# ---------------------------------------------------------------------------

_EDIT_TOOLS = {"file_write", "file_edit", "edit"}


def run_case(
    case: EvalCase,
    backend: LLMBackend,
    *,
    log_dir: str = "./logs/eval",
    registry_builder: Callable | None = None,
    config_overrides: dict | None = None,
) -> RunRecord:
    """
    在隔离工作目录里跑一条 case，返回带指标的 RunRecord。

    config_overrides: 注入 AgentConfig 的字段覆盖（消融开关用），
        如 {"enable_repo_map": False}。默认 None = 完整系统。
    """
    registry = (registry_builder or _default_registry)()

    task = Task(
        description=case.problem_statement,
        repo_path=case.repo_path,
        max_steps=case.max_steps,
    )
    agent = Agent(
        backend, registry,
        AgentConfig(max_steps=case.max_steps, **(config_overrides or {})),
    )

    # 工具默认以进程 cwd 为工作目录（复现 CLI 在仓库内运行的行为），
    # 故切到 case 工作区；日志目录先转绝对路径，避免被写进临时目录。
    log_dir_abs = str(Path(log_dir).resolve())
    old_cwd = os.getcwd()
    try:
        os.chdir(case.repo_path)
        with EventLog.create(task, log_dir=log_dir_abs) as log:
            result: RunResult = agent.run(task, log)
            steps_to_first_edit = _first_edit_step(log)
    finally:
        os.chdir(old_cwd)

    resolved, verify_out = _grade(case)

    return RunRecord(
        case_id=case.case_id,
        resolved=resolved,
        status=result.status.value,
        steps=result.steps_taken,
        steps_to_first_edit=steps_to_first_edit,
        total_tokens=result.total_tokens,
        patch_lines=len((result.patch or "").splitlines()),
        verify_output=verify_out[-500:],
        bucket=case.bucket,
    )


def _grade(case: EvalCase) -> tuple[bool, str]:
    """用 sys.executable 跑 verify_cmd（避免子进程 python 没 pytest）。"""
    cmd = case.verify_cmd.replace("python", sys.executable, 1)
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=case.repo_path,
            capture_output=True, text=True, timeout=120,
        )
        return proc.returncode == 0, (proc.stdout + proc.stderr)
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _first_edit_step(log: EventLog) -> int | None:
    """从事件日志找首次文件编辑发生在第几步（1-indexed）。"""
    step = 0
    for action in log.get_actions():
        step += 1
        if (
            action.action_type == ActionType.TOOL_CALL
            and action.tool_call
            and action.tool_call.name in _EDIT_TOOLS
        ):
            return step
    return None


def _default_registry():
    """复用 entry.cli 的工具注册逻辑（默认本地 runtime）。"""
    from config.schema import load_config
    from entry.cli import _build_registry
    return _build_registry(load_config())


# ---------------------------------------------------------------------------
# 指标聚合
# ---------------------------------------------------------------------------

def aggregate(records: list[RunRecord]) -> dict:
    n = len(records) or 1
    resolved = sum(r.resolved for r in records)
    edits = [r.steps_to_first_edit for r in records if r.steps_to_first_edit]
    return {
        "n": len(records),
        "resolved": resolved,
        "resolve_rate": round(resolved / n, 4),
        "avg_steps": round(sum(r.steps for r in records) / n, 2),
        "avg_steps_to_first_edit": round(sum(edits) / len(edits), 2) if edits else None,
        "avg_tokens": round(sum(r.total_tokens for r in records) / n, 1),
        "max_steps_hit": sum(r.status == RunStatus.MAX_STEPS.value for r in records),
        "gave_up": sum(r.status == RunStatus.GAVE_UP.value for r in records),
    }


def print_report(records: list[RunRecord]) -> dict:
    print("\n".join(r.to_row() for r in records))
    agg = aggregate(records)
    print("-" * 72)
    print(
        f"  resolve_rate={agg['resolve_rate']:.1%} "
        f"({agg['resolved']}/{agg['n']})  "
        f"avg_steps={agg['avg_steps']}  "
        f"avg_1st_edit={agg['avg_steps_to_first_edit']}  "
        f"avg_tokens={agg['avg_tokens']}  "
        f"max_steps_hit={agg['max_steps_hit']}  gave_up={agg['gave_up']}"
    )
    return agg
