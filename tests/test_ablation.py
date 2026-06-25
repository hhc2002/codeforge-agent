"""
tests/test_ablation.py

Phase 1 消融基础设施测试：覆盖 AgentConfig 的消融开关 + LLM 链路的 temperature 透传。

每个开关都断言「显式分支真正绕过对应组件」，而不是靠极端参数近似关闭：
- enable_repo_map=False   → 不调用 RepoMap.build()，system prompt 用占位符
- enable_token_budget=False → 不调用 trim_history()，发送完整历史
- enable_loop_detection=False → _is_looping 恒为 False
- enable_reflection=False  → 不注入 reflection（但工具执行/历史写入不变）
- temperature             → None 时不传给 API；显式值时所有调用路径都带上
"""

from __future__ import annotations

import pytest

from agent.core import Agent, AgentConfig
from agent.task import Action, ActionType, Task, ToolCall
from agent.event_log import EventLog
from context.history import ConversationHistory
from context.token_budget import TokenBudget
from llm.base import LLMMessage, MockBackend
from tools.base import BaseTool, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class _FakeRepoMap:
    """记录 build() 是否被调用；build 返回一个可识别的 sentinel 摘要。"""
    SENTINEL = "<<REPO_MAP_SUMMARY_SENTINEL>>"

    def __init__(self) -> None:
        self.build_calls = 0

    def build(self, budget: int | None = None) -> str:
        self.build_calls += 1
        return self.SENTINEL


class _FakeLog:
    """只实现 _is_looping 需要的 get_actions()。"""
    def __init__(self, actions: list[Action]) -> None:
        self._actions = actions

    def get_actions(self) -> list[Action]:
        return self._actions


class _FailingPytest(BaseTool):
    """一个名为 pytest、永远失败的工具，用来触发 reflection 条件 A。"""
    @property
    def name(self) -> str:
        return "pytest"

    @property
    def description(self) -> str:
        return "run the test suite"

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    def execute(self, params: dict) -> ToolResult:
        return ToolResult(success=False, output="", error="1 failed, 0 passed")


def _make_agent(cfg: AgentConfig, registry: ToolRegistry | None = None) -> Agent:
    return Agent(MockBackend([]), registry or ToolRegistry(), cfg)


def _repeated_tool_actions(n: int) -> list[Action]:
    return [
        Action(
            action_type=ActionType.TOOL_CALL,
            thought="t",
            tool_call=ToolCall(name="shell", params={"cmd": "ls"}),
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# enable_repo_map
# ---------------------------------------------------------------------------

class TestRepoMapSwitch:
    def test_on_builds_and_injects_summary(self):
        agent = _make_agent(AgentConfig(enable_repo_map=True))
        fake = _FakeRepoMap()
        msgs = agent._build_messages(ConversationHistory(), TokenBudget(80_000), fake)
        system = msgs[0].content
        assert fake.build_calls == 1
        assert _FakeRepoMap.SENTINEL in system

    def test_off_skips_build_and_uses_placeholder(self):
        agent = _make_agent(AgentConfig(enable_repo_map=False))
        fake = _FakeRepoMap()
        msgs = agent._build_messages(ConversationHistory(), TokenBudget(80_000), fake)
        system = msgs[0].content
        # 关键：根本不 build（而非 build 了但不用）
        assert fake.build_calls == 0
        assert _FakeRepoMap.SENTINEL not in system
        # 退回 prompt.py 的占位符
        assert "Repository summary not yet available" in system


# ---------------------------------------------------------------------------
# enable_token_budget
# ---------------------------------------------------------------------------

_TRUNCATION_MARKER = "truncated to fit context window"


def _fat_history(n: int) -> ConversationHistory:
    """构造一个远超预算的历史：首条任务 + n 条大消息。"""
    h = ConversationHistory(max_messages=10_000)   # 条数窗口设大，只看 token 预算这一层
    h.add(LLMMessage(role="user", content="TASK: fix the bug"))
    for i in range(n):
        h.add(LLMMessage(role="user", content=f"observation {i} " + "lorem ipsum dolor " * 400))
    return h


class TestTokenBudgetSwitch:
    def test_on_trims_history(self):
        agent = _make_agent(AgentConfig(enable_token_budget=True))
        hist = _fat_history(40)
        msgs = agent._build_messages(hist, TokenBudget(total=2_000), _FakeRepoMap())
        # 预算很小 → 必然裁剪，留下截断标记，且消息数远少于完整历史
        assert any(_TRUNCATION_MARKER in m.content for m in msgs)
        assert len(msgs) < 1 + hist.message_count

    def test_off_sends_full_history(self):
        agent = _make_agent(AgentConfig(enable_token_budget=False))
        hist = _fat_history(40)
        msgs = agent._build_messages(hist, TokenBudget(total=2_000), _FakeRepoMap())
        # 完整发送：system + 全部历史，且没有任何截断标记
        assert len(msgs) == 1 + hist.message_count
        assert not any(_TRUNCATION_MARKER in m.content for m in msgs)


# ---------------------------------------------------------------------------
# enable_loop_detection
# ---------------------------------------------------------------------------

class TestLoopDetectionSwitch:
    def test_on_detects_repeated_actions(self):
        agent = _make_agent(AgentConfig(enable_loop_detection=True, loop_detection_window=3))
        log = _FakeLog(_repeated_tool_actions(3))
        assert agent._is_looping(log) is True

    def test_off_never_detects(self):
        agent = _make_agent(AgentConfig(enable_loop_detection=False, loop_detection_window=3))
        log = _FakeLog(_repeated_tool_actions(3))
        assert agent._is_looping(log) is False


# ---------------------------------------------------------------------------
# enable_reflection（走真实 run loop，用 MockBackend 脚本）
# ---------------------------------------------------------------------------

def _run_failing_test_then_finish(cfg: AgentConfig, tmp_path) -> MockBackend:
    """脚本：先调失败的 pytest，再 finish。返回 backend 以便检查收到的 messages。"""
    script = [
        Action(action_type=ActionType.TOOL_CALL, thought="run tests",
                tool_call=ToolCall(name="pytest", params={})),
        Action(action_type=ActionType.FINISH, thought="done", message="done"),
    ]
    backend = MockBackend(script)
    registry = ToolRegistry().register(_FailingPytest())
    agent = Agent(backend, registry, cfg)
    task = Task(description="d", repo_path=str(tmp_path), max_steps=5)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        agent.run(task, log)
    return backend


class TestReflectionSwitch:
    def test_on_injects_reflection_after_failed_test(self, tmp_path):
        backend = _run_failing_test_then_finish(
            AgentConfig(enable_reflection=True), tmp_path
        )
        # 第 2 次 LLM 调用（finish 那步）收到的 messages 里应含 reflection 提示
        second_call_msgs = backend.received_messages[1]
        assert any("[REFLECTION]" in m.content for m in second_call_msgs)

    def test_off_does_not_inject_reflection(self, tmp_path):
        backend = _run_failing_test_then_finish(
            AgentConfig(enable_reflection=False), tmp_path
        )
        second_call_msgs = backend.received_messages[1]
        assert not any("[REFLECTION]" in m.content for m in second_call_msgs)
        # 但工具确实执行了（pytest observation 进了历史）
        assert any("pytest" in m.content for m in second_call_msgs)


# ---------------------------------------------------------------------------
# temperature 透传
# ---------------------------------------------------------------------------

class TestTemperaturePlumbing:
    def test_sampling_params_none_omits(self):
        openai = pytest.importorskip("openai")  # noqa: F841
        from llm.openai_compat import OpenAICompatBackend
        b = OpenAICompatBackend(model="gpt-4o", api_key="x", temperature=None)
        assert b._sampling_params() == {}

    def test_sampling_params_zero_passes(self):
        pytest.importorskip("openai")
        from llm.openai_compat import OpenAICompatBackend
        b = OpenAICompatBackend(model="gpt-4o", api_key="x", temperature=0)
        assert b._sampling_params() == {"temperature": 0}

    def test_create_backend_forwards_temperature(self):
        pytest.importorskip("openai")
        from llm.router import create_backend
        b = create_backend(provider="gemini", model="gemini-2.5-flash",
                           api_key="x", temperature=0)
        assert b._temperature == 0

    def test_config_parses_temperature(self):
        from config.schema import LLMConfig
        assert LLMConfig().temperature is None

    def test_create_backend_from_config_forwards(self):
        pytest.importorskip("openai")
        from llm.router import create_backend_from_config
        b = create_backend_from_config({
            "provider": "gemini", "model": "gemini-2.5-flash",
            "api_key": "x", "temperature": 0.0,
        })
        assert b._temperature == 0.0


class TestDeepSeekThinkingDisable:
    """DeepSeek v4 默认开思考、与 tool_choice='required' 冲突，需自动关思考。"""

    def test_v4_flash_disables_thinking(self):
        pytest.importorskip("openai")
        from llm.router import create_backend
        b = create_backend(provider="deepseek", model="deepseek-v4-flash", api_key="x")
        assert b._extra_body == {"thinking": {"type": "disabled"}}
        # extra_body 必须经 _sampling_params 注入到每个 API 调用
        assert b._sampling_params()["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_non_v4_no_extra_body(self):
        pytest.importorskip("openai")
        from llm.router import create_backend
        # 非思考别名 deepseek-chat 不需要关思考
        b = create_backend(provider="deepseek", model="deepseek-chat", api_key="x")
        assert b._extra_body is None
        assert "extra_body" not in b._sampling_params()

    def test_gemini_unaffected(self):
        pytest.importorskip("openai")
        from llm.router import create_backend
        b = create_backend(provider="gemini", model="gemini-2.5-flash", api_key="x")
        assert b._extra_body is None
