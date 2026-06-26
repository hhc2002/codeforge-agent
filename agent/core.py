"""
agent/core.py

ReAct 主循环。整个 agent 的大脑。

职责（只做这些，不做别的）：
- 维护对话历史，每轮组装 messages 调用 LLM
- 拿到 Action 后调用 ToolRegistry 执行
- 把 Action + Observation 写入 EventLog
- 检测三种终止/Reflection 触发条件
- 返回 RunResult

不负责：
- 任何 LLM 细节（交给 LLMBackend）
- 任何工具实现（交给 Tool）
- 上下文压缩（由 context/ 模块负责）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from agent.event_log import EventLog
from context.history import ConversationHistory
from context.repo_map import RepoMap
from context.token_budget import TokenBudget
from agent.prompt import (
    build_system_prompt,
    build_task_prompt,
    reflection_no_edit,
    reflection_test_failed,
)
from agent.task import (
    Action, ActionType, Event, EventType,
    Observation, ObservationStatus, RunResult, RunStatus, Task, ToolCall,
)
from llm.base import LLMBackend, LLMMessage, LLMToolSchema
from tools.base import ToolRegistry

logger = logging.getLogger(__name__)


# 控制工具：把"完成/放弃"做成显式工具调用，而不是靠模型停说话来判定。
# 配合 tool_choice="required"，杜绝模型只吐文字计划就被误判为完成。
_CONTROL_TOOL_SCHEMAS = [
    LLMToolSchema(
        name="finish",
        description=(
            "Call this ONLY after you have actually made and verified the code "
            "changes that solve the task. Provide a summary of what you changed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Summary of the changes you made"},
            },
            "required": ["summary"],
        },
    ),
    LLMToolSchema(
        name="give_up",
        description="Call this only if the task is impossible or you are truly stuck after several attempts.",
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why you cannot complete the task"},
            },
            "required": ["reason"],
        },
    ),
]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Agent 运行时配置，从 config/default.yaml 加载后传入。"""
    max_steps: int = 40
    reflection_no_edit_steps: int = 6   # 连续 N 步无文件写操作触发 Reflection
    loop_detection_window: int = 3       # 连续 N 步完全相同 action 判定死循环
    test_tool_names: tuple[str, ...] = ("test", "pytest")  # 触发 Reflection 的工具名
    budget_tokens: int = 80_000            # 总 token 预算
    history_max_messages: int = 40         # 历史最大条数
    llm_max_retries: int = 3               # LLM 调用失败最大重试次数
    llm_retry_delay: float = 2.0           # 重试间隔（秒，指数退避）
    stream: bool = False                   # 是否启用流式输出
    stream_callback: object = None         # StreamCallback，最终回答流式回调
    thought_callback: object = None        # StreamCallback，推理过程流式回调（推理模型专用）
    confirm_dangerous: bool = False        # 是否对危险命令要求用户确认
    confirm_callback: object = None        # ConfirmCallback，None=跳过确认

    # 防"空 finish"：没有任何代码改动就声称完成（auto/推理模型常见——在 reasoning
    # 里把解法想了一遍却没落到 file_edit）。开启后退回并要求先编辑，最多退回 N 次。
    # 默认关（保留 chat/问答类无改动 finish 的合法性），SWE-bench 等修复任务由调用方开启。
    require_edit_before_finish: bool = False
    max_empty_finishes: int = 3
    max_no_ops: int = 5                     # 模型连续不发 tool_call 多少次后放弃

    # ── 消融开关（ablation switches）──────────────────────────────────────
    # 默认全开 = 完整系统。每个开关都走显式分支真正绕过对应组件，
    # 不靠"极端参数近似关闭"（那会保留组件逻辑、引入伪实验条件）。
    enable_repo_map: bool = True        # 关 → 不 build/注入 repo-map，system prompt 用占位符
    enable_reflection: bool = True      # 关 → 不触发 reflection 注入（工具执行/历史写入不变）
    enable_loop_detection: bool = True  # 关 → 不做死循环检测
    enable_token_budget: bool = True    # 关 → 跳过 token 裁剪，发送完整历史



# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    ReAct 主循环实现。

    用法：
        agent = Agent(backend, registry, config)
        result = agent.run(task, log)
    """

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry,
        config: AgentConfig | None = None,
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._cfg = config or AgentConfig()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(self, task: Task, log: EventLog) -> RunResult:
        """
        执行一次完整的 agent 运行。

        Args:
            task: 任务描述
            log:  已初始化的 EventLog（由调用方创建并传入）

        Returns:
            RunResult，包含最终状态和统计信息
        """
        self._current_repo_path = task.repo_path
        self._task_query = task.description   # 喂给 repo-map 做任务相关性排序
        # 按 repo_path 隔离 repo_map 缓存（换 repo 重建；chat 多轮同 repo 复用，
        # 只用首轮 query 做相关性，可接受。SWE-bench 每实例是独立 worktree 路径 +
        # 新建 Agent，天然各自重建，不会串用别题的 query）。
        cache_key = task.repo_path
        if getattr(self, "_repo_map_cache_key", None) != cache_key:
            if hasattr(self, "_repo_map_cache"):
                del self._repo_map_cache
            self._repo_map_cache_key = cache_key
        log.log_task_start(task)
        logger.info("Agent starting task %s", task.task_id)

        # 初始化上下文管理器
        # 如果调用方（ChatSession）注入了共享 history，直接复用；
        # 否则新建（单次 run 模式）
        if hasattr(self, "_pending_history") and self._pending_history is not None:
            history = self._pending_history
        else:
            history = ConversationHistory(max_messages=self._cfg.history_max_messages)
            # 单次模式：把任务描述作为第一条 user 消息
            from agent.prompt import build_task_prompt
            history.add(LLMMessage(
                role="user",
                content=build_task_prompt(task.description, task.repo_path, task.issue_url),
            ))
        token_budget = TokenBudget(total=self._cfg.budget_tokens)
        repo_map = RepoMap(task.repo_path)

        total_tokens = 0
        steps_without_edit = 0
        empty_finishes = 0
        no_ops = 0

        for step in range(1, task.max_steps + 1):
            logger.debug("Step %d/%d", step, task.max_steps)

            # ── 1. 组装 messages，调用 LLM ──────────────────────────────
            messages = self._build_messages(history, token_budget, repo_map)
            tools = self._registry.get_schemas() + _CONTROL_TOOL_SCHEMAS

            try:
                response = self._call_with_retry(messages, tools)
            except Exception as exc:
                logger.error("LLM call failed at step %d after retries: %s", step, exc)
                log.log_task_failed(steps=step, reason=f"LLM error: {exc}")
                return RunResult(
                    task_id=task.task_id,
                    status=RunStatus.FAILED,
                    summary=f"LLM call failed: {exc}",
                    steps_taken=step,
                    total_tokens=total_tokens,
                    error=str(exc),
                )

            total_tokens += response.total_tokens
            action = response.action

            # ── 2. 写入 Action event ────────────────────────────────────
            log.log_action(step=step, action=action, raw_content=response.raw_content)
            logger.info("Step %d: %r", step, action)

            # ── 3. 检测死循环（连续相同 action）────────────────────────
            if self._is_looping(log):
                reason = f"Loop detected: same action repeated {self._cfg.loop_detection_window} times"
                logger.warning(reason)
                log.log_task_failed(steps=step, reason=reason)
                return RunResult(
                    task_id=task.task_id,
                    status=RunStatus.GAVE_UP,
                    summary=reason,
                    steps_taken=step,
                    total_tokens=total_tokens,
                )

            # ── 4. 终止 action ──────────────────────────────────────────
            if action.action_type == ActionType.FINISH:
                summary = action.message or "Task complete."
                patch = self._get_git_diff(task.repo_path)
                # 护栏：零改动的 finish 多半是"想完了没动手"。退回要求先编辑，
                # 而不是接受空补丁；超过上限才放行（交给后续判分判失败）。
                if (
                    self._cfg.require_edit_before_finish
                    and not (patch and patch.strip())
                    and empty_finishes < self._cfg.max_empty_finishes
                ):
                    empty_finishes += 1
                    nudge = (
                        "You called finish but the working tree has no changes "
                        "(git diff is empty). You cannot resolve the issue without "
                        "editing files. Locate the cause and apply a concrete edit "
                        "with file_edit/file_write, then finish."
                    )
                    history.add(LLMMessage(role="assistant",
                                           content=self._format_action_for_history(action)))
                    history.add(LLMMessage(role="user", content=nudge))
                    logger.info("Rejected empty finish #%d at step %d", empty_finishes, step)
                    continue
                log.log_task_complete(steps=step, summary=summary)
                return RunResult(
                    task_id=task.task_id,
                    status=RunStatus.SUCCESS,
                    summary=summary,
                    steps_taken=step,
                    total_tokens=total_tokens,
                    patch=patch,
                )

            if action.action_type == ActionType.GIVE_UP:
                reason = action.message or "Agent gave up."
                log.log_task_failed(steps=step, reason=reason)
                return RunResult(
                    task_id=task.task_id,
                    status=RunStatus.GAVE_UP,
                    summary=reason,
                    steps_taken=step,
                    total_tokens=total_tokens,
                )

            # ── 4b. NO_OP：模型只输出文字没发 tool_call ─────────────────
            # （tool_choice="auto" + 思考模型常见）。nudge 它去调工具并继续；
            # 反复 nudge 仍不调，超过上限就诚实 give_up，不当成功。
            if action.action_type == ActionType.NO_OP:
                no_ops += 1
                if no_ops > self._cfg.max_no_ops:
                    reason = f"Model produced no tool call {no_ops} times; giving up."
                    log.log_task_failed(steps=step, reason=reason)
                    return RunResult(
                        task_id=task.task_id,
                        status=RunStatus.GAVE_UP,
                        summary=reason,
                        steps_taken=step,
                        total_tokens=total_tokens,
                    )
                nudge = (
                    "Your last reply contained no tool call, so nothing happened. "
                    "You must respond with a tool call to act: file_view to inspect, "
                    "file_edit/file_write to change code, test to run tests, "
                    "or finish when the fix is complete. Issue a tool call now."
                )
                if action.message:
                    history.add(LLMMessage(role="assistant", content=action.message))
                history.add(LLMMessage(role="user", content=nudge))
                logger.info("NO_OP nudge #%d at step %d", no_ops, step)
                continue

            # ── 5. 执行工具 ─────────────────────────────────────────────
            if action.action_type == ActionType.TOOL_CALL and action.tool_call:
                no_ops = 0   # 连续计数：调到工具就清零，散落的 NO_OP 不累计成 give_up
                tc = action.tool_call
                result = self._registry.execute_tool(tc.name, tc.params)
                observation = result.to_observation(tc.name)

                # 追踪是否有文件写操作
                if tc.name in ("file_write", "file_edit", "edit"):
                    steps_without_edit = 0
                else:
                    steps_without_edit += 1

                log.log_observation(step=step, observation=observation)

                # 把 action 和 observation 加入对话历史
                history.add(LLMMessage(
                    role="assistant",
                    content=self._format_action_for_history(action),
                ))
                history.add(LLMMessage(
                    role="user",
                    content=self._format_observation_for_history(observation),
                ))

                # ── 6. Reflection 触发判断 ──────────────────────────────
                # 消融：enable_reflection=False → 只跳过 reflection 注入与 log，
                # 上面的工具执行与历史写入保持不变。
                if self._cfg.enable_reflection:
                    # 触发条件 A：测试工具失败
                    if (
                        tc.name in self._cfg.test_tool_names
                        and not observation.is_success()
                    ):
                        reflect_prompt = reflection_test_failed()
                        log.log_reflection(
                            step=step,
                            reason="test_failed",
                            prompt=reflect_prompt,
                        )
                        history.add(LLMMessage(role="user", content=reflect_prompt))
                        logger.debug("Reflection triggered: test_failed at step %d", step)

                    # 触发条件 B：连续 N 步无编辑
                    elif steps_without_edit >= self._cfg.reflection_no_edit_steps:
                        reflect_prompt = reflection_no_edit(steps_without_edit)
                        log.log_reflection(
                            step=step,
                            reason="no_edit",
                            prompt=reflect_prompt,
                        )
                        history.add(LLMMessage(role="user", content=reflect_prompt))
                        steps_without_edit = 0  # 重置计数，避免每步都触发
                        logger.debug("Reflection triggered: no_edit at step %d", step)

            elif action.action_type == ActionType.REFLECTION:
                # LLM 主动要求 reflection（预留，当前 MockBackend 不产生）
                history.add(LLMMessage(
                    role="assistant",
                    content=action.thought,
                ))

        # ── 7. 超出步数上限 ─────────────────────────────────────────────
        reason = f"Reached max_steps limit ({task.max_steps})"
        log.log_task_failed(steps=task.max_steps, reason=reason)
        return RunResult(
            task_id=task.task_id,
            status=RunStatus.MAX_STEPS,
            summary=reason,
            steps_taken=task.max_steps,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        history: ConversationHistory,
        token_budget: TokenBudget,
        repo_map: RepoMap,
    ) -> list[LLMMessage]:
        """
        组装发给 LLM 的完整 messages，含 token 裁剪。
        """
        schemas = self._registry.get_schemas()

        # repo-map（带缓存：只在第一步生成，之后复用）。
        # 消融：enable_repo_map=False → 不 build，传 None，build_system_prompt
        # 会退回"自己去探索"的占位符，等价于没有 repo-map 组件。
        if self._cfg.enable_repo_map:
            if not hasattr(self, "_repo_map_cache"):
                self._repo_map_cache = repo_map.build(
                    budget=token_budget.default_plan().repo_map,
                    query=getattr(self, "_task_query", ""),
                )
            repo_summary = self._repo_map_cache
        else:
            repo_summary = None

        system_content = build_system_prompt(
            repo_path=getattr(self, "_current_repo_path", "."),
            tools=schemas,
            repo_summary=repo_summary,
        )

        # 历史裁剪。消融：enable_token_budget=False → 跳过 token 裁剪，
        # 发送完整历史（而非"给个超大预算"近似，避免还走裁剪逻辑）。
        if self._cfg.enable_token_budget:
            trimmed_history_dicts = token_budget.trim_history(
                history.to_dicts(),
                token_budget.default_plan().history,
            )
        else:
            trimmed_history_dicts = history.to_dicts()

        # 组装：system + 裁剪后的 history
        messages = [LLMMessage(role="system", content=system_content)]
        for d in trimmed_history_dicts:
            messages.append(LLMMessage(role=d["role"], content=d["content"]))
        return messages

    def _format_action_for_history(self, action: Action) -> str:
        """把 Action 格式化为 assistant 消息，写入对话历史。"""
        parts = [f"Thought: {action.thought}"]
        if action.tool_call:
            parts.append(f"Action: {action.tool_call.name}")
            parts.append(f"Params: {json.dumps(action.tool_call.params, ensure_ascii=False)}")
        elif action.message:
            parts.append(f"Message: {action.message}")
        return "\n".join(parts)

    def _format_observation_for_history(self, observation: Observation) -> str:
        """把 Observation 格式化为 user 消息，写入对话历史。"""
        status = "SUCCESS" if observation.is_success() else "ERROR"
        lines = [f"[Tool: {observation.tool_name} | {status}]"]
        if observation.output:
            lines.append(observation.output)
        if observation.error and not observation.is_success():
            lines.append(f"Error: {observation.error}")
        return "\n".join(lines)

    def _is_looping(self, log: EventLog) -> bool:
        """
        检测是否陷入死循环：最近 N 条 action 完全相同。
        比较 (tool_name, params) 元组。
        """
        # 消融：enable_loop_detection=False → 显式关闭（不靠 window=0 近似，
        # 那在下面的长度比较里并非安全的关闭方式）。
        if not self._cfg.enable_loop_detection:
            return False
        n = self._cfg.loop_detection_window
        actions = log.get_actions()
        if len(actions) < n:
            return False

        recent = actions[-n:]
        # 只对 TOOL_CALL 类型做检测
        if not all(a.action_type == ActionType.TOOL_CALL for a in recent):
            return False
        if not all(a.tool_call for a in recent):
            return False

        first = recent[0].tool_call
        return all(
            a.tool_call.name == first.name and a.tool_call.params == first.params
            for a in recent[1:]
        )

    def _call_with_retry(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSchema],
    ):
        """
        带指数退避重试的 LLM 调用。
        stream=True 时走 backend.stream()，否则走 complete()。
        不重试：认证失败（401/403）、参数错误（400）。
        """
        import time as _time

        last_exc: Exception | None = None
        delay = self._cfg.llm_retry_delay

        for attempt in range(1, self._cfg.llm_max_retries + 1):
            try:
                if self._cfg.stream:
                    cb = self._cfg.stream_callback
                    thought_cb = self._cfg.thought_callback
                    if hasattr(self._backend, "stream"):
                        return self._backend.stream(
                            messages, tools,
                            on_text=cb,
                            on_thought=thought_cb,
                        )
                return self._backend.complete(messages, tools)
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                if any(kw in exc_str for kw in (
                    "401", "403", "invalid api key", "authentication",
                    "400", "bad request",
                )):
                    raise
                if attempt < self._cfg.llm_max_retries:
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, self._cfg.llm_max_retries, exc, delay,
                    )
                    _time.sleep(delay)
                    delay *= 2

        raise last_exc  # type: ignore[misc]

    def _get_git_diff(self, repo_path: str) -> str | None:
        """抓取 git diff HEAD 作为 patch，失败时静默返回 None。"""
        import subprocess
        try:
            proc = subprocess.run(
                ["git", "diff", "HEAD"],
                capture_output=True, text=True, timeout=10, cwd=repo_path,
            )
            diff = proc.stdout.strip()
            return diff if diff else None
        except Exception:
            return None