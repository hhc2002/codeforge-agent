"""
context/repo_map.py

Repo-map：把整个 repo 的结构压缩成一段摘要字符串，注入 system prompt。

核心思路（简化版 Aider repo-map）：
1. 用 tree-sitter 扫描源码文件，提取函数/类定义
2. 用正则 fallback 处理 tree-sitter 不支持或未安装的语言
3. 按"重要性"排序：顶层定义 > 方法，文件越小越可能是核心文件
4. 按 token 预算截取，生成摘要字符串

## 多语言支持

tree-sitter 每种语言需要单独安装语言包：

    pip install tree-sitter-python       # Python（必装）
    pip install tree-sitter-javascript   # JavaScript
    pip install tree-sitter-typescript   # TypeScript
    pip install tree-sitter-go           # Go
    pip install tree-sitter-rust         # Rust
    pip install tree-sitter-java         # Java

未安装的语言自动降级为正则解析，不报错。
新增语言只需在 _LANG_REGISTRY 里加一行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 语言注册表
# 格式：文件扩展名 → (pip 包名, 模块属性名)
# 运行时按需 import，失败时静默跳过，降级为正则
# ---------------------------------------------------------------------------

_LANG_REGISTRY: dict[str, tuple[str, str]] = {
    ".py":  ("tree_sitter_python",     "language"),
    ".js":  ("tree_sitter_javascript", "language"),
    ".ts":  ("tree_sitter_typescript", "language_typescript"),
    ".tsx": ("tree_sitter_typescript", "language_tsx"),
    ".go":  ("tree_sitter_go",         "language"),
    ".rs":  ("tree_sitter_rust",       "language"),
    ".java":("tree_sitter_java",       "language"),
    ".cpp": ("tree_sitter_cpp",        "language"),
    ".c":   ("tree_sitter_c",          "language"),
    ".rb":  ("tree_sitter_ruby",       "language"),
}

# AST 节点类型 → symbol kind 映射（各语言通用名）
_FUNC_NODES: frozenset[str] = frozenset({
    "function_definition",       # Python, Go, C, C++
    "async_function_definition", # Python async def
    "function_declaration",      # JS, TS, Java
    "method_declaration",        # Java
    "method_definition",         # JS class method
    "function_item",             # Rust fn
    "arrow_function",            # JS arrow（跳过，通常是匿名的）
})
_CLASS_NODES: frozenset[str] = frozenset({
    "class_definition",   # Python
    "class_declaration",  # JS, TS, Java
    "struct_item",        # Rust struct
    "impl_item",          # Rust impl
    "interface_declaration",  # TS/Java
})

# 跳过的目录
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", "dist", "build",
})

# 正则 fallback：匹配常见语言的定义语句
_SYMBOL_RE = re.compile(
    r"^[ \t]*(def|class|function|func|fn|pub fn|async fn|async def"
    r"|public|private|protected|static)\s+(\w+)",
    re.MULTILINE,
)

# 单文件在 repo-map 里最多列几个顶层符号（防止巨型文件刷屏）
_MAX_SYMS_PER_FILE = 12


def _is_noise_name(name: str) -> bool:
    """对 map 无信息量的符号名：dispatch 用的裸 `_`、dunder。
    sympy 里大量 `def _(...)` 多分派函数，列出来纯噪声。"""
    return name == "_" or name.startswith("__")


# 从 issue/task 文本里抽"被提到的标识符"，用于任务相关性排序：
#   反引号包裹的 `foo_bar` / 点号路径取末段；snake_case；CamelCase。
_BACKTICK_RE = re.compile(r"`([A-Za-z_][\w.]*)`")
_SNAKE_RE = re.compile(r"\b([a-z_][a-z0-9_]{3,})\b")
_CAMEL_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]{3,})\b")
# snake_case 常见英文词，命中了也不算"标识符信号"，过滤掉减少噪声
_STOPWORDS = frozenset({
    "should", "would", "could", "which", "there", "where", "when", "this",
    "that", "with", "from", "have", "does", "doesn", "what", "your", "into",
    "return", "returns", "result", "error", "raise", "raises", "value", "values",
    "expected", "example", "following", "above", "below", "instead", "because",
})


def extract_query_idents(text: str) -> set[str]:
    """从任务描述里抽候选标识符（用于把相关文件顶到 map 顶部）。"""
    if not text:
        return set()
    idents: set[str] = set()
    for m in _BACKTICK_RE.findall(text):
        idents.add(m)
        idents.add(m.split(".")[-1])           # `a.b.c` 取末段
    idents.update(_CAMEL_RE.findall(text))
    idents.update(w for w in _SNAKE_RE.findall(text) if w not in _STOPWORDS)
    return {i for i in idents if len(i) >= 4}

# 已加载的 tree-sitter Language 对象缓存（避免重复 import）
_lang_cache: dict[str, object] = {}   # ext → Language or None


def _get_language(ext: str):
    """
    按文件扩展名获取 tree-sitter Language 对象。
    未安装时返回 None，调用方降级为正则。
    """
    if ext in _lang_cache:
        return _lang_cache[ext]

    entry = _LANG_REGISTRY.get(ext)
    if entry is None:
        _lang_cache[ext] = None
        return None

    module_name, attr_name = entry
    try:
        import importlib
        from tree_sitter import Language
        mod = importlib.import_module(module_name)
        lang_fn = getattr(mod, attr_name)
        lang = Language(lang_fn())
        _lang_cache[ext] = lang
        return lang
    except Exception:
        _lang_cache[ext] = None
        return None


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """一个提取出来的符号（函数或类定义）。"""
    name: str
    kind: str           # "function" | "class" | "method"
    line: int
    file: Path
    indent: int = 0
    signature: str = ""  # 源码里的定义行（如 "def foo(self, x):"），渲染用

    @property
    def is_toplevel(self) -> bool:
        return self.indent == 0


@dataclass
class FileInfo:
    """一个文件的元信息和符号列表。"""
    path: Path
    size: int
    symbols: list[Symbol] = field(default_factory=list)

    @property
    def rel_path(self) -> str:
        return str(self.path)

    def importance_score(self) -> float:
        # 旧实现 = 顶层符号数 − size/10000，奖励"符号多"→ 测试文件 / 机器生成的巨型
        # 文件（如 sympy 的 rubi 积分规则，几百个符号）霸榜，真源文件沉底。
        # 修正：① 符号数封顶（巨型文件不能靠数量取胜）；② 加重大小惩罚；
        #       ③ 测试文件大幅降权（要改 bug 的几乎都不是 test_*.py）。
        top_level = sum(1 for s in self.symbols
                        if s.is_toplevel and not _is_noise_name(s.name))
        score = min(top_level, 15) - self.size / 20_000
        p = self.rel_path.replace("\\", "/").lower()
        if "/test" in p or p.startswith("test") or "conftest" in p:
            score -= 100
        if "/bench" in p or "benchmark" in p:    # 基准测试也不是要改的源码
            score -= 100
        return score


# ---------------------------------------------------------------------------
# RepoMap
# ---------------------------------------------------------------------------

class RepoMap:
    """
    扫描 repo，生成摘要字符串。

    用法：
        rm = RepoMap(repo_path="/path/to/repo")
        summary = rm.build(budget=8000)
    """

    def __init__(self, repo_path: str | Path) -> None:
        self._root = Path(repo_path).resolve()

    def build(self, budget: int = 8000, query: str = "") -> str:
        files = self._scan()
        if not files:
            return "(empty repository)"

        idents = extract_query_idents(query)
        # 排序键：先按"任务相关性"（命中 issue 标识符的文件顶到最前），再按静态
        # 重要性兜底。query 为空时 relevance 恒 0，退化为纯重要性排序（向后兼容）。
        files.sort(
            key=lambda f: (self._relevance(f, idents), f.importance_score()),
            reverse=True,
        )

        lines: list[str] = []
        char_count = 0
        max_chars = budget * 4
        shown = 0

        for fi in files:
            block = self._format_file(fi, idents)
            # skip 不 break：装不下就跳过这个文件继续试更小的，
            # 杜绝"第一个超大文件就清空整张表"（旧 break 的 bug）。
            if char_count + len(block) > max_chars:
                continue
            lines.append(block)
            char_count += len(block)
            shown += 1

        omitted = len(files) - shown
        if omitted > 0:
            lines.append(f"... ({omitted} more files not shown)")

        return "".join(lines)   # 每个 block 自带换行

    def _scan(self) -> list[FileInfo]:
        results: list[FileInfo] = []
        for path in sorted(self._root.rglob("*")):
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size > 500_000:
                continue

            fi = FileInfo(path=path.relative_to(self._root), size=size)
            ext = path.suffix.lower()

            if ext in _LANG_REGISTRY or ext in {".py", ".js", ".ts", ".go", ".rs"}:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    fi.symbols = _extract_symbols(content, fi.path, ext)
                except OSError:
                    pass

            results.append(fi)
        return results

    def _relevance(self, fi: FileInfo, idents: set[str]) -> int:
        """文件与当前任务的相关性：符号名命中 issue 标识符（权重最高）+ 路径命中。"""
        if not idents:
            return 0
        names = {s.name for s in fi.symbols}
        score = len(names & idents) * 3
        parts = set(re.split(r"[/_.]", fi.rel_path.lower()))
        score += len(parts & {i.lower() for i in idents})
        return score

    def _format_file(self, fi: FileInfo, idents: set[str]) -> str:
        # 借鉴 Aider：显示真实**签名行**（含参数）而非裸名，保留类-方法嵌套。
        # 取舍（防回到旧的 60× 膨胀）：每文件**封顶 _MAX_SYMS_PER_FILE 个符号**，
        # 且 query 命中的符号**必显示、优先**（哪怕是方法）——保证 agent 直接看到
        # 该改的那个函数；其余按顶层补足。
        syms = [s for s in fi.symbols if not _is_noise_name(s.name)]
        if not syms:
            return f"{fi.rel_path}\n"

        matched = [s for s in syms if s.name in idents]      # 命中 issue 的符号
        toplevel = [s for s in syms if s.is_toplevel]
        picked: list[Symbol] = []
        seen: set[tuple[str, int]] = set()
        for s in matched + toplevel:                          # 命中优先，再补顶层
            key = (s.name, s.line)
            if key not in seen:
                seen.add(key)
                picked.append(s)
            if len(picked) >= _MAX_SYMS_PER_FILE:
                break
        picked.sort(key=lambda s: s.line)                     # 按行号还原结构顺序

        out = [f"{fi.rel_path}:"]
        for s in picked:
            indent = "    " if s.is_toplevel else "        "   # 方法多缩进一层
            out.append(indent + (s.signature or f"{s.kind} {s.name}"))
        omitted = len(syms) - len(picked)
        if omitted > 0:
            out.append(f"    ... (+{omitted} more)")
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 符号提取（对外暴露，供测试使用）
# ---------------------------------------------------------------------------

def _extract_symbols(content: str, filepath: Path, ext: str) -> list[Symbol]:
    """
    按扩展名选择解析方式：tree-sitter（如已安装）或正则 fallback。
    """
    lang = _get_language(ext)
    if lang is not None:
        syms = _extract_with_treesitter(content, filepath, lang)
    else:
        syms = _extract_symbols_regex(content, filepath)
    # 回填签名行：用行号取源码定义行（"def foo(self, x):" / "class Bar:"），
    # 渲染时显示真实签名而非裸名（借鉴 Aider）。截到 120 字符防超长。
    src_lines = content.splitlines()
    for s in syms:
        if 1 <= s.line <= len(src_lines):
            s.signature = src_lines[s.line - 1].strip()[:120]
    return syms


def _extract_with_treesitter(content: str, filepath: Path, lang) -> list[Symbol]:
    """用 tree-sitter 提取符号，失败时降级为正则。"""
    try:
        from tree_sitter import Parser
        parser = Parser(lang)
        tree = parser.parse(content.encode("utf-8", errors="replace"))
        return _walk_tree(tree.root_node, filepath)
    except Exception:
        return _extract_symbols_regex(content, filepath)


def _walk_tree(node, filepath: Path) -> list[Symbol]:
    """递归遍历 tree-sitter AST，提取函数和类定义。"""
    results: list[Symbol] = []
    ntype = node.type

    if ntype in _FUNC_NODES and ntype != "arrow_function":
        name_node = node.child_by_field_name("name")
        if name_node:
            indent = node.start_point[1]
            kind = "method" if indent > 0 else "function"
            results.append(Symbol(
                name=name_node.text.decode("utf-8", errors="replace"),
                kind=kind,
                line=node.start_point[0] + 1,
                file=filepath,
                indent=indent,
            ))
    elif ntype in _CLASS_NODES:
        name_node = node.child_by_field_name("name")
        if name_node:
            indent = node.start_point[1]
            results.append(Symbol(
                name=name_node.text.decode("utf-8", errors="replace"),
                kind="class",
                line=node.start_point[0] + 1,
                file=filepath,
                indent=indent,
            ))

    for child in node.children:
        results.extend(_walk_tree(child, filepath))

    return results


# 保留原函数名供测试 import
def _extract_python_symbols(content: str, filepath: Path) -> list[Symbol]:
    """兼容旧接口，测试文件用此名调用。"""
    return _extract_symbols(content, filepath, ".py")


def _extract_symbols_regex(content: str, filepath: Path) -> list[Symbol]:
    """正则 fallback，支持多语言。"""
    symbols: list[Symbol] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        m = _SYMBOL_RE.match(line)
        if not m:
            continue
        keyword = m.group(1)
        name = m.group(2)
        # 跳过 Java/JS 修饰符误匹配（public/private 后面跟的是类型，不是名字）
        if keyword in ("public", "private", "protected", "static"):
            continue
        indent = len(line) - len(line.lstrip())
        if keyword == "class":
            kind = "class"
        elif indent > 0:
            kind = "method"
        else:
            kind = "function"
        symbols.append(Symbol(
            name=name, kind=kind, line=lineno,
            file=filepath, indent=indent,
        ))
    return symbols