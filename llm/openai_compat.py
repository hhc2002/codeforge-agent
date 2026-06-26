"""
llm/openai_compat.py

OpenAI-compatible backend。覆盖：
- OpenAI (api.openai.com)
- DeepSeek (api.deepseek.com) — deepseek-chat 支持 function calling，R1 不支持
- Groq (api.groq.com)
- Ollama (localhost:11434/v1)

全部用 openai SDK，切换只改 base_url + api_key。

function calling 不支持时（如 DeepSeek R1）走文本解析 fallback：
从 LLM 输出的文本里提取 JSON 格式的 tool call。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.task import Action, ActionType, ToolCall
from llm.base import LLMBackend, LLMMessage, LLMResponse, LLMToolSchema

logger = logging.getLogger(__name__)

# 不支持 function calling 的模型（前缀匹配）
_NO_FUNCTION_CALLING: tuple[str, ...] = (
    "deepseek-reasoner",    # DeepSeek R1
    "deepseek-r1",
)


class OpenAICompatBackend(LLMBackend):
    """
    OpenAI-compatible API backend。

    Args:
        model:    模型名，如 "gpt-4o", "deepseek-chat", "llama3-70b-8192"
        api_key:  API key
        base_url: API base URL，None 时用 OpenAI 官方地址
        max_tokens: 最大输出 token 数
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        # extra_body：透传给 API 的额外 JSON 字段。例如 DeepSeek v4 默认开思考模式，
        # 而思考模式不支持 tool_choice="required"，需 {"thinking": {"type": "disabled"}}。
        self._extra_body = extra_body
        self._reasoning_effort = reasoning_effort
        # tool_choice：默认 required（强制每轮调一个工具）。但思考模式不支持 required，
        # 故当 extra_body 显式开启 thinking 时降为 "auto"。
        self._tool_choice = "required"
        if extra_body and extra_body.get("thinking", {}).get("type") == "enabled":
            self._tool_choice = "auto"
        self._use_function_calling = not any(
            model.lower().startswith(prefix) for prefix in _NO_FUNCTION_CALLING
        )

    def _sampling_params(self) -> dict[str, Any]:
        """
        统一注入采样及额外请求参数，所有 API 调用路径
        （complete / stream，function-calling / text）都经此，确保各路径一致、
        不会变成两套实验条件。
        - temperature：仅在显式设置时传（某些兼容端点对默认参数敏感）。
        - extra_body：provider 特定的额外字段（如 DeepSeek 关思考）。
        """
        params: dict[str, Any] = {}
        if self._temperature is not None:
            params["temperature"] = self._temperature
        if self._reasoning_effort:
            params["reasoning_effort"] = self._reasoning_effort
        if self._extra_body:
            params["extra_body"] = self._extra_body
        return params

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_function_calling(self) -> bool:
        return self._use_function_calling

    def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSchema],
    ) -> LLMResponse:
        api_messages = _to_openai_messages(messages)

        logger.debug(
            "OpenAI-compat request: model=%s messages=%d tools=%d fc=%s",
            self._model, len(api_messages), len(tools), self._use_function_calling,
        )

        if self._use_function_calling:
            response = self._complete_with_tools(api_messages, tools)
        else:
            response = self._complete_text_only(api_messages, tools)

        return response

    # ------------------------------------------------------------------
    # function calling 路径
    # ------------------------------------------------------------------

    def _complete_with_tools(
        self,
        api_messages: list[dict],
        tools: list[LLMToolSchema],
    ) -> LLMResponse:
        api_tools = [_to_openai_tool(t) for t in tools]

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=api_messages,
            tools=api_tools,
            # required：强制每轮必须调用一个工具（含显式 finish/give_up），
            # 杜绝模型只输出文字计划被误判为任务完成。
            # 思考模式不支持 required，此时 self._tool_choice 已降为 "auto"。
            tool_choice=self._tool_choice,
            **self._sampling_params(),
        )

        choice = response.choices[0]
        message = choice.message
        # 推理模型（DeepSeek 思考模式等）把链路放在 reasoning_content，最终答案才在 content。
        # 强制工具调用时 content 常为空，故 content 缺失时回落到 reasoning_content，
        # 修掉日志里清一色 "(no thought)"。
        thought = message.content or getattr(message, "reasoning_content", None) or "(no thought)"

        logger.debug(
            "OpenAI-compat response: finish_reason=%s input=%d output=%d",
            choice.finish_reason,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

        action = _parse_openai_response(choice, thought)

        return LLMResponse(
            action=action,
            raw_content=thought,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    # ------------------------------------------------------------------
    # 文本解析 fallback（R1 等不支持 function calling 的模型）
    # ------------------------------------------------------------------

    def _complete_text_only(
        self,
        api_messages: list[dict],
        tools: list[LLMToolSchema],
    ) -> LLMResponse:
        # 在 system prompt 里注入工具描述，要求模型输出 JSON
        tool_desc = _build_tool_description_for_text(tools)
        # 在第一条 system 消息后插入工具说明
        augmented = list(api_messages)
        if augmented and augmented[0]["role"] == "system":
            augmented[0] = {
                "role": "system",
                "content": augmented[0]["content"] + "\n\n" + tool_desc,
            }

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=augmented,
            **self._sampling_params(),
        )

        choice = response.choices[0]
        raw_text = choice.message.content or ""

        action = _parse_text_response(raw_text)

        return LLMResponse(
            action=action,
            raw_content=raw_text,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )


# ---------------------------------------------------------------------------
# 格式转换
# ---------------------------------------------------------------------------

def _to_openai_messages(messages: list[LLMMessage]) -> list[dict]:
    """把 LLMMessage 列表转为 OpenAI messages 格式。"""
    result = []
    for msg in messages:
        if msg.tool_call_id:
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })
        else:
            result.append({"role": msg.role, "content": msg.content})
    return result


def _to_openai_tool(schema: LLMToolSchema) -> dict:
    """转换为 OpenAI tool schema 格式。"""
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


def _parse_openai_response(choice: Any, thought: str) -> Action:
    """
    解析 OpenAI API 的 choice，返回 Action。
    """
    finish_reason = choice.finish_reason
    message = choice.message

    if finish_reason in ("tool_calls", "stop") and message.tool_calls:
        # 取第一个 tool call（agent 每轮只调一个工具）
        tc = message.tool_calls[0]
        try:
            params = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            params = {"raw": tc.function.arguments}

        # 控制工具：finish / give_up 映射为终止 Action，而非普通工具调用
        if tc.function.name == "finish":
            return Action(
                action_type=ActionType.FINISH,
                thought=thought,
                message=params.get("summary") or thought,
            )
        if tc.function.name == "give_up":
            return Action(
                action_type=ActionType.GIVE_UP,
                thought=thought,
                message=params.get("reason") or thought,
            )

        return Action(
            action_type=ActionType.TOOL_CALL,
            thought=thought,
            tool_call=ToolCall(name=tc.function.name, params=params),
        )

    if finish_reason == "stop":
        # function-calling 路径下 stop 但没有 tool_call：模型只是输出了文字/推理、
        # 忘了发工具调用（tool_choice="auto" + 思考模型常见）。这**不是**完成——
        # 之前误判成 FINISH 会让"光说不练"被当成功。返回 NO_OP，由循环 nudge 它去调工具。
        return Action(
            action_type=ActionType.NO_OP,
            thought=thought,
            message=thought if thought and thought != "(no thought)" else "",
        )

    # length（token 超限）或其他
    return Action(
        action_type=ActionType.GIVE_UP,
        thought=thought,
        message=f"Unexpected finish_reason: {finish_reason}",
    )


# ---------------------------------------------------------------------------
# 文本解析 fallback
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_INLINE_JSON_RE = re.compile(r"\{[^{}]+\}", re.DOTALL)

_FINISH_KEYWORDS = ("task complete", "task is complete", "i have finished", "all done")
_GIVE_UP_KEYWORDS = ("cannot solve", "give up", "unable to", "i cannot")


def _build_tool_description_for_text(tools: list[LLMToolSchema]) -> str:
    """
    给不支持 function calling 的模型注入工具描述。
    要求模型输出特定 JSON 格式：
    {"tool": "tool_name", "params": {...}}
    或者输出 FINISH / GIVE_UP 关键词。
    """
    if not tools:
        return ""

    lines = [
        "## Available tools",
        "To call a tool, output ONLY a JSON block in this exact format:",
        '```json\n{"tool": "<tool_name>", "params": {<params>}}\n```',
        "",
        "To finish the task, output: TASK_COMPLETE: <summary>",
        "To give up, output: GIVE_UP: <reason>",
        "",
        "Tools:",
    ]
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)


def _parse_text_response(text: str) -> Action:
    """
    从纯文本中解析 Action。
    优先匹配 JSON block，其次匹配关键词。
    """
    text_stripped = text.strip()

    # 检查 TASK_COMPLETE
    if text_stripped.upper().startswith("TASK_COMPLETE:"):
        summary = text_stripped[len("TASK_COMPLETE:"):].strip()
        return Action(
            action_type=ActionType.FINISH,
            thought=text_stripped,
            message=summary or "Task complete",
        )

    # 检查 GIVE_UP
    if text_stripped.upper().startswith("GIVE_UP:"):
        reason = text_stripped[len("GIVE_UP:"):].strip()
        return Action(
            action_type=ActionType.GIVE_UP,
            thought=text_stripped,
            message=reason or "Agent gave up",
        )

    # 尝试提取 JSON block（```json ... ```）
    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        return _try_parse_tool_json(block_match.group(1), thought=text_stripped)

    # 尝试提取内联 JSON
    for m in _INLINE_JSON_RE.finditer(text):
        action = _try_parse_tool_json(m.group(0), thought=text_stripped)
        if action is not None:
            return action

    # 关键词匹配兜底
    text_lower = text.lower()
    if any(kw in text_lower for kw in _FINISH_KEYWORDS):
        return Action(
            action_type=ActionType.FINISH,
            thought=text_stripped,
            message=text_stripped,
        )
    if any(kw in text_lower for kw in _GIVE_UP_KEYWORDS):
        return Action(
            action_type=ActionType.GIVE_UP,
            thought=text_stripped,
            message=text_stripped,
        )

    # 无法解析，GIVE_UP
    logger.warning("Could not parse action from text: %s", text_stripped[:100])
    return Action(
        action_type=ActionType.GIVE_UP,
        thought=text_stripped,
        message="Could not parse a valid action from model output",
    )


def _try_parse_tool_json(json_str: str, thought: str) -> Action | None:
    """尝试把 JSON 字符串解析为 TOOL_CALL Action，失败返回 None。"""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    tool_name = data.get("tool") or data.get("name") or data.get("function")
    params = data.get("params") or data.get("arguments") or data.get("input") or {}

    if not tool_name or not isinstance(tool_name, str):
        return None

    return Action(
        action_type=ActionType.TOOL_CALL,
        thought=thought,
        tool_call=ToolCall(name=tool_name, params=params if isinstance(params, dict) else {}),
    )


# ---------------------------------------------------------------------------
# 流式支持
# ---------------------------------------------------------------------------

from llm.base import StreamCallback


def _openai_stream(
    self: "OpenAICompatBackend",
    messages: list,
    tools: list,
    on_text: StreamCallback | None = None,
    on_thought: StreamCallback | None = None,
) -> "LLMResponse":
    """
    OpenAI-compatible 流式调用实现。
    on_text:    最终回答（message）的流式回调
    on_thought: 推理过程（reasoning_content）的流式回调，仅推理模型有内容
    """
    api_messages = _to_openai_messages(messages)

    if self._use_function_calling:
        return _stream_with_tools(self, api_messages, tools, on_text, on_thought)
    else:
        return _stream_text_only(self, api_messages, tools, on_text)


def _stream_with_tools(self, api_messages, tools, on_text, on_thought=None):
    api_tools = [_to_openai_tool(t) for t in tools] if tools else None

    kwargs = dict(
        model=self._model,
        max_tokens=self._max_tokens,
        messages=api_messages,
        stream=True,
        **self._sampling_params(),
    )
    if api_tools:
        kwargs["tools"] = api_tools
        kwargs["tool_choice"] = "required"

    # 收集流式 chunks
    full_text = ""
    full_reasoning = ""  # reasoning_content（推理模型专有）
    finish_reason = None
    tool_calls_raw = []      # 收集 tool call deltas

    stream = self._client.chat.completions.create(**kwargs)
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            continue

        delta = choice.delta
        finish_reason = choice.finish_reason or finish_reason

        # reasoning_content delta（DeepSeek R1 / Claude thinking）
        reasoning_delta = getattr(delta, "reasoning_content", None)
        if reasoning_delta:
            full_reasoning += reasoning_delta
            if on_thought:
                on_thought(reasoning_delta)

        # text delta（最终回答）
        if delta.content:
            full_text += delta.content
            if on_text:
                on_text(delta.content)

        # tool call delta 拼接
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                fn = getattr(tc_delta, "function", None)
                idx = tc_delta.index
                if idx is None:
                    # 某些 OpenAI 兼容端点（如 Gemini）流式分片里不带 index：
                    # 带 name 的分片视为一次新的 tool call，否则并入最后一个
                    if (fn and fn.name) or not tool_calls_raw:
                        tool_calls_raw.append({"name": "", "arguments": ""})
                    idx = len(tool_calls_raw) - 1
                else:
                    while len(tool_calls_raw) <= idx:
                        tool_calls_raw.append({"name": "", "arguments": ""})
                if fn and fn.name:
                    tool_calls_raw[idx]["name"] += fn.name
                if fn and fn.arguments:
                    tool_calls_raw[idx]["arguments"] += fn.arguments

    # 构造 mock choice 供 _parse_openai_response 复用
    import json as _json
    from types import SimpleNamespace

    if tool_calls_raw and finish_reason in ("tool_calls", "stop"):
        tcs = []
        for tc in tool_calls_raw:
            try:
                params = _json.loads(tc["arguments"])
            except Exception:
                params = {"raw": tc["arguments"]}
            fn = SimpleNamespace(name=tc["name"], arguments=tc["arguments"])
            tcs.append(SimpleNamespace(function=fn))
        mock_message = SimpleNamespace(content=full_text or None, tool_calls=tcs)
    else:
        mock_message = SimpleNamespace(content=full_text or None, tool_calls=None)

    mock_choice = SimpleNamespace(finish_reason=finish_reason or "stop", message=mock_message)
    # 有 reasoning_content 时，thought = 推理过程，message = 最终回答
    # 没有时（普通 chat 模型），thought 置空，message = 模型输出
    thought_for_parse = full_text or "(no thought)"
    action = _parse_openai_response(mock_choice, thought_for_parse)
    # 如果有推理内容，覆盖 action.thought
    if full_reasoning and action.action_type.value == "finish":
        action = action.__class__(
            action_type=action.action_type,
            thought=full_reasoning,
            tool_call=action.tool_call,
            message=action.message,
        )

    # 流式模式拿不到精确 token 数，估算
    from context.token_budget import estimate_tokens
    input_tokens = sum(estimate_tokens(m.get("content", "")) for m in api_messages)
    output_tokens = estimate_tokens(full_text)

    return LLMResponse(
        action=action,
        raw_content=full_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _stream_text_only(self, api_messages, tools, on_text):
    """R1 等不支持 function calling 的模型的流式路径。"""
    tool_desc = _build_tool_description_for_text(tools)
    augmented = list(api_messages)
    if augmented and augmented[0]["role"] == "system":
        augmented[0] = {
            "role": "system",
            "content": augmented[0]["content"] + "\n\n" + tool_desc,
        }

    full_text = ""
    stream = self._client.chat.completions.create(
        model=self._model,
        max_tokens=self._max_tokens,
        messages=augmented,
        stream=True,
        **self._sampling_params(),
    )
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            continue
        delta = choice.delta
        if delta.content:
            full_text += delta.content
            if on_text:
                on_text(delta.content)

    action = _parse_text_response(full_text)

    from context.token_budget import estimate_tokens
    return LLMResponse(
        action=action,
        raw_content=full_text,
        input_tokens=sum(estimate_tokens(m.get("content", "")) for m in augmented),
        output_tokens=estimate_tokens(full_text),
    )


# 把 stream() 方法绑定到 OpenAICompatBackend
OpenAICompatBackend.stream = _openai_stream