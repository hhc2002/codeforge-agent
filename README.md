# Forge Agent

自主编程智能体。给它一个任务描述，它会自己探索代码库、修改文件、运行测试，直到完成。

默认使用 **Gemini**，同时支持 **Claude、OpenAI、DeepSeek、Groq、Ollama**。内置 tree-sitter repo-map、外科式文件编辑（file_edit）、流式输出、Docker 沙箱、Reflection 自纠错、GitHub Issue 自动修复，并附可复现评测 harness。

---

## 快速开始

```bash
# 安装
git clone <repo-url> && cd codeforge-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 配置 API key（默认走 Gemini）
export GEMINI_API_KEY=AIza...    # 或 ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY

# 验证
python smoke_test.py

# 使用：进入项目目录，直接运行 agent
cd your-project
agent                            # 不带子命令即进入交互式 chat
```

---

## 使用方式

### chat 模式（推荐）

持续对话，每轮历史保留，最接近 Claude Code 的体验。**不带子命令直接运行 `agent` 即进入 chat**：

```bash
agent                                 # 等价于 agent chat（当前目录）
agent chat --repo /path/to/project    # 指定目录
agent chat --model gemini-2.5-pro     # 切换模型（默认 gemini-2.5-flash）
agent chat --sandbox                  # Docker 沙箱
```

对话内命令：`/exit` 退出、`/stats` 查看统计、`/clear` 清空历史、`/help` 帮助

### run 模式

一次性任务，适合明确的批处理场景：

```bash
agent run --task "修复所有 failing 的测试"
agent run --task-file task.txt           # 从文件读任务
agent run --task "..." --confirm         # 危险命令需确认
agent run --task "..." --sandbox         # Docker 沙箱
```

### GitHub Issue 自动修复

```bash
export GITHUB_TOKEN=ghp_xxx
python -m entry.github_issue \
    --repo owner/repo --issue 42 --local-path /tmp/myrepo
```

自动拉取 Issue → 运行 agent → 提交 PR。

---

## 配置

编辑 `config/default.yaml`：

```yaml
llm:
  provider: gemini                        # gemini | anthropic | openai | deepseek | groq | ollama
  model: gemini-2.5-flash                 # 默认；gemini-2.5-pro 更强
  api_key: ${GEMINI_API_KEY}              # 从环境变量读取
  base_url:                               # 留空自动选择端点；自定义 OpenAI-compatible 时填写

agent:
  max_steps: 40           # 每轮最大步数
  budget_tokens: 80000    # token 预算

context:
  repo_map_budget: 8000   # repo-map 注入量
  history_window: 20      # 保留历史轮数
```

---

## 项目结构

```
coding-agent/
├── agent/              # 核心：ReAct 主循环、事件日志、数据结构
│   ├── core.py         # Agent 类，驱动整个运行循环
│   ├── task.py         # Task / Action / Observation / RunResult 数据类
│   ├── event_log.py    # JSONL append-only 事件流，支持回放
│   └── prompt.py       # System prompt 模板
│
├── llm/                # LLM 后端
│   ├── base.py         # LLMBackend 抽象基类，含默认 stream()
│   ├── anthropic_backend.py   # Claude 原生（tool_use + 流式）
│   ├── openai_compat.py       # OpenAI / DeepSeek / Groq / Ollama
│   └── router.py       # 按配置选择 backend
│
├── tools/              # 工具层（agent 可调用的操作）
│   ├── base.py         # BaseTool + ToolRegistry
│   ├── file_tool.py    # 文件读 / 查看 / 整文件写 / 外科式编辑（file_edit）
│   ├── shell_tool.py   # Shell 执行（四层安全防护）
│   ├── search_tool.py  # 文本搜索 / 文件查找 / 符号定位
│   ├── test_tool.py    # pytest 执行 + 结构化结果解析
│   ├── git_tool.py     # git status / diff / add / commit
│   └── runtime.py      # LocalRuntime / DockerRuntime
│
├── context/            # 上下文管理
│   ├── repo_map.py     # tree-sitter 多语言符号提取，生成 repo 摘要
│   ├── token_budget.py # Token 预算分配与裁剪
│   └── history.py      # 对话历史滑动窗口
│
├── entry/              # 入口层
│   ├── cli.py          # Click CLI（run / chat / log 子命令）
│   ├── chat.py         # ChatSession，跨轮持久化历史
│   └── github_issue.py # GitHub Issue → PR 自动化
│
├── config/
│   ├── default.yaml    # 默认配置
│   └── schema.py       # 配置加载与校验
│
├── eval/               # 评测 harness（可复现评测 + 逐组件消融）
│   ├── harness.py      # 跑 case、抓 patch、判分、聚合指标
│   ├── local_cases.py  # 本地合成 bug-fix 任务集（不依赖 Docker）
│   └── run_real.py     # 用真实 LLM 跑评测，出 resolve rate
│
├── tests/              # 387 个测试，覆盖所有模块
├── smoke_test.py       # 端到端联通验证
├── quicksort_task.py   # 示例任务脚本
└── USAGE.md            # 完整使用教程
```

---

## 核心特性

**多模型支持**
- Google Gemini（默认，OpenAI-compatible 端点）
- Anthropic Claude（原生 tool_use）
- OpenAI、DeepSeek、Groq、Ollama（OpenAI-compatible）
- DeepSeek R1 等不支持 function calling 的模型走文本解析 fallback
- 配置文件一行切换，或 `--model` 参数临时覆盖

**多语言 Repo-map**
用 tree-sitter 精确提取符号（函数、类、方法），生成 repo 摘要注入 system prompt，
支持 Python / JavaScript / TypeScript / Go / Rust / Java / C++ / C / Ruby。

**外科式文件编辑**
`file_edit` 用唯一片段 str_replace 做定点修改，唯一性校验保证不误删无关代码——
改已有文件时比整文件覆盖更安全、更省 token（命中 0 次或多次都拒绝并提示）。

**流式输出**
模型 thought 逐 token 实时打印，工具调用实时显示，体验接近 Claude Code。

**安全机制（三层）**
- 硬拦截黑名单：`rm -rf /`、`mkfs` 等永不执行
- 只读白名单：`ls`、`grep`、`git status`、`pytest` 等直接执行
- 写操作确认：`--confirm` 模式下 `git commit`、`pip install` 等需 y/n 确认

**Docker 沙箱**
`--sandbox` 参数，所有命令在 `python:3.11-slim` 容器里执行，
repo 通过 bind mount 双向同步，默认断网。

**Reflection 机制**
- 测试失败 → 自动触发反思 prompt，重新分析错误原因
- 连续 6 步无文件修改 → 触发反思，防止探索死循环
- 连续 3 步相同操作 → 判定死循环，自动终止

**事件日志**
每次运行生成 JSONL 日志，记录所有 action / observation / reflection，
支持完整回放和统计分析。

---

## 评测

`eval/` 提供可复现评测 harness：本地合成 bug-fix 任务集（不依赖 Docker），用真实 LLM 跑出
resolve rate，并记录 steps / steps-to-first-edit / tokens 等指标，便于跨模型对比与逐组件消融。

```bash
python -m eval.run_real --provider gemini --model gemini-2.5-flash -n 3
```

---

## 安全说明

`--confirm` 模式（`run`）和 `chat` 模式默认对写操作要求确认，执行前显示：

```
  ⚠  Agent wants to run:
     $ git commit -m "fix parser bug"
  Allow? [y/N]
```

`--sandbox` 模式在 Docker 容器中执行，宿主机环境完全隔离。

---

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest                     # 全量（387 passed，7 skipped）
pytest tests/test_day3.py  # 单个文件

# 可选：更多语言的 tree-sitter 支持
pip install tree-sitter-javascript tree-sitter-typescript \
            tree-sitter-go tree-sitter-rust tree-sitter-java

# 可选：精确 token 计数
pip install tiktoken
```

---

## 命令参考

```bash
# chat
agent chat [--repo PATH] [--model MODEL] [--sandbox] [-v]

# run
agent run --task TEXT [--repo PATH] [--task-file FILE]
          [--model MODEL] [--confirm] [--sandbox] [--no-stream] [-v]

# log
agent log list [--dir DIR]
agent log show LOG_FILE

# github issue
python -m entry.github_issue \
    -r owner/repo -i ISSUE_NUM -l LOCAL_PATH [--no-pr] [-v]
```

详细用法见 [USAGE.md](USAGE.md)。
