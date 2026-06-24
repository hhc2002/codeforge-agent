# Eval Harness 实验记录 / Runbook

记录"把项目跑起来 + 搭评测 harness + 接 Gemini + 首次真跑"的全过程，可复现。

---

## 1. 环境（Phase 0）

系统 `python3` 是 **3.15.0a2（alpha）**，带 C 扩展的包装不上，**不能用**。改用稳定版：

```bash
# 用 miniconda 的 Python 3.13.9 建隔离 venv
/opt/miniconda3/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"        # 装项目 + pytest
```

验证：
```bash
.venv/bin/python -m pytest -q              # 383 collected, 376 passed, 7 skipped(Docker), ~19s
```
> 3 个 `TestTestTool` 失败仅因子进程 `python` 没装 pytest（环境问题非代码 bug）；
> 把 venv 放进 PATH 即全过：`PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest tests/test_day3.py::TestTestTool`

**真实测试数 = 383 collected / 376 passed / 7 skipped。**

---

## 2. 新增的评测 harness（C1）

| 文件 | 作用 |
|---|---|
| `eval/harness.py` | 管道层：`run_case(case, backend)` → 切到工作区跑 Agent → git diff 抓 patch → verify 判分 → 出指标（resolve/steps/首次编辑步/token）；`aggregate`/`print_report` 聚合 |
| `eval/local_cases.py` | 本地合成 bug-fix 任务集（注入 bug + 隐藏测试，git init），不依赖 Docker |
| `eval/dryrun_mock.py` | 用 MockBackend 零成本打通管道 + 判分正反对照 |
| `eval/run_real.py` | 用真实 LLM 跑本地任务出 resolve 数 |

设计要点：
- 对 backend 解耦——MockBackend 验管道（免费），真实 LLM 出数。
- 判分用 `sys.executable` 跑 pytest，避开子进程 python 没 pytest 的坑。
- **`run_case` 内 `os.chdir(repo_path)`**：工具默认以进程 cwd 为工作目录（复现 CLI 在仓库内运行），不切目录 Agent 会去操作 forge 仓库根目录。

---

## 3. 接入 Gemini

`llm/router.py` 加两行（provider 走现有 `OpenAICompatBackend`）：
```python
_PROVIDER_BASE_URLS["gemini"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
_ENV_KEY_MAP["gemini"]        = "GEMINI_API_KEY"
```
- Key 验证：列模型 OK；`gemini-2.0-flash` 已下线，用 **`gemini-2.5-flash`**。
- function calling 验证：返回正确 `tool_calls`，可直接驱动 Agent。

---

## 4. 怎么跑

```bash
# (a) MockBackend 干跑（免费，验管道）
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m eval.dryrun_mock

# (b) 真实 Gemini 跑本地任务
GEMINI_API_KEY=<key> PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m eval.run_real --provider gemini --model gemini-2.5-flash -n 3

# 日志（可回放）在 ./logs/eval/*.jsonl，按步看：
.venv/bin/python - <<'PY'
import json,glob,os
f=max(glob.glob("logs/eval/*.jsonl"),key=os.path.getmtime)
for l in open(f):
    e=json.loads(l)
    if e["event_type"]=="action":
        a=e["payload"]["action"]; tc=a.get("tool_call")
        print(a["action_type"], tc["name"] if tc else a.get("message","")[:80])
PY
```

---

## 5. 实验记录与发现

### 5.1 MockBackend 干跑 ✅
fix-case → resolved；noop-case → 未 resolved（判分能区分对错）。管道全通。

### 5.2 首次真跑（Gemini 2.5-flash, 3 本地任务）→ resolve 0/3
两个 bug 被评测抓出：

1. **[已修] cwd bug**：Agent 工具操作进程 cwd 而非 repo_path → 在 forge 根目录里找不到 case 文件。
   修复：`run_case` 内 `os.chdir(case.repo_path)`（日志目录转绝对路径避免写进临时目录）。
2. **[已修] Agent 过早终止**：模型回一段文字计划、没调工具时，
   `openai_compat._parse_openai_response` 见 `finish_reason=="stop"` + 有 content → 判为 **FINISH**。
   flash 和 **pro 都中招**（0/3、零编辑）→ 是 Agent 协议设计问题，非模型弱。
   **修复**：把 finish/give_up 做成**显式工具** + `tool_choice="required"`，模型每轮必须调工具，
   done = 调 finish 工具。改了 3 处：
   - `agent/core.py`：`_CONTROL_TOOL_SCHEMAS`（finish/give_up）追加进发给 LLM 的 tools
   - `llm/openai_compat.py`：`tool_choice="required"` + 解析 finish/give_up → FINISH/GIVE_UP
   - `llm/anthropic_backend.py`：同样解析 finish/give_up（Claude 路径一致）

### 5.4 修复后重跑（Gemini 2.5-flash, 3 本地任务）→ **resolve 3/3 = 100%**
```
✓ arith-add      steps=6  1st_edit=4  tok=14254  patch=9L
✓ parity-iseven  steps=6  1st_edit=4  tok=14468  patch=9L
✓ list-last      steps=8  1st_edit=4  tok=19622  patch=0L
resolve_rate=100% (3/3)  avg_steps=6.67  avg_1st_edit=4.0  avg_tokens=16.1k
```
- Agent 真的做题了：探索→第 4 步开始编辑→跑测→finish。token 涨到 ~16k/case 是因为真干活（6–8 步）。
- **全量测试 376 passed / 7 skipped / 0 回归**。
- 小尾巴：list-last `patch=0L` 但 resolved——Agent 用了 git_commit，`git diff HEAD` 看不到改动；
  resolve 由隐藏测试判定仍准确。SWE-bench 需要 patch，故待办：patch 抓取改为 `git diff <base_commit>`。

### 5.3 Token 拆解（实测）
每步固定开销 ~2.1k token = 12 工具 schema(~1348) + system 规则(~721)，**每步重发**；
小任务里 repo-map 仅 ~27。flash 上 13k token ≈ 可忽略。
**SWE-bench 才是 token/成本大头**（真实仓库 + 多步，单实例可能 3万–15万 token）。

---

## 6. 下一步（待指令）

- [x] 修过早终止 → resolve 0→3/3，测试 0 回归
- [ ] patch 抓取改为 `git diff <base_commit>`（兼容 Agent 自行 commit 的情况，SWE-bench 需要）
- [ ] C2 消融开关（enable_repo_map / enable_reflection …）+ C3 temperature=0
- [ ] 扩本地任务集（10+ 题，更难一点）跑出更稳的 resolve
- [ ] 上 SWE-bench Lite（需开 Docker；官方 harness 判分）

---

## 未提交说明
`eval/` 全部文件 + `llm/router.py` 改动尚未提交，等指令。
