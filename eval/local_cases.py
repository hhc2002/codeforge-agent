"""
eval/local_cases.py

本地诊断集（local diagnostic eval）——不依赖 SWE-bench / Docker。

目标不是"严肃 benchmark"，而是**证明 harness 能稳定揭示 agent 行为**：
每道题刻意瞄准 agent 的某个能力（探索定位 / 测试反馈复盘 / 抗噪声），
并写明"关掉对应模块时预测会出现的失败信号"（expected_without_component），
这样 smoke 消融跑完可以拿结果去对预测——符合即组件确实在起作用，
不符合即 case 设计有问题或组件名不副实。

分桶（bucket）：
  sanity     单文件显式 bug，证明管道通（应在所有 condition 下都过）
  repo_map   多文件，问题描述**不点名** bug 文件，逼 agent 定位
  reflection 第一眼改法不完整 / 会引入回归，需测试反馈后再修
  noise      真实 bug 被一堆无关文件包围，制造定位/上下文噪声

provenance：
  source / source_id 记录每题出处。handwritten=手写；quixbugs=源自 QuixBugs。
  注意：从外部蒸馏来的小 repo **不等于**那个 benchmark 的官方分数，报告里别混淆。

每个 spec 自带 reference_fix（一处 str_replace），仅用于离线自检
（tests/test_diagnostic_cases.py 验证"buggy 必失败、打上 reference_fix 必通过"），
不会喂给 agent。
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from eval.harness import EvalCase


@dataclass
class CaseSpec:
    """一道诊断题的完整定义。"""
    case_id: str
    bucket: str
    files: dict[str, str]                 # 相对路径 -> 文件内容（含 buggy 源码 + 测试）
    verify_cmd: str
    problem_statement: str
    why_this_case: str
    expected_capability: str
    expected_without_component: str
    # reference_fix = (相对路径, old_string, new_string)，仅供离线自检
    reference_fix: tuple[str, str, str]
    source: str = "handwritten"
    source_id: str = ""
    max_steps: int = 15


# ===========================================================================
# sanity（2）—— 单文件显式 bug，管道通畅性检查
# ===========================================================================

_SANITY = [
    CaseSpec(
        case_id="sanity-arith-add",
        bucket="sanity",
        files={
            "mathutils.py": "def add(a, b):\n    return a - b  # bug\n",
            "test_mathutils.py": (
                "from mathutils import add\n\n\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
                "    assert add(-1, 1) == 0\n"
            ),
        },
        verify_cmd="python -m pytest test_mathutils.py -q",
        problem_statement=(
            "add() in mathutils.py returns wrong results; test_mathutils.py fails. "
            "Fix the bug."
        ),
        why_this_case="最简单的单文件单行 bug，确认 explore→edit→verify→finish 管道在该 backend 上能跑通。",
        expected_capability="基本的读文件、定位、改一行、跑测试。",
        expected_without_component="无（sanity 题应在所有 condition 下都 resolved）。",
        reference_fix=("mathutils.py", "return a - b  # bug", "return a + b"),
    ),
    CaseSpec(
        case_id="sanity-sieve",
        bucket="sanity",
        source="quixbugs",
        source_id="sieve",
        files={
            "sieve.py": (
                "def sieve(maxn):\n"
                "    primes = []\n"
                "    for n in range(2, maxn + 1):\n"
                "        if any(n % p > 0 for p in primes):  # bug: should be all(...)\n"
                "            primes.append(n)\n"
                "    return primes\n"
            ),
            "test_sieve.py": (
                "from sieve import sieve\n\n\n"
                "def test_sieve():\n"
                "    assert sieve(10) == [2, 3, 5, 7]\n"
                "    assert sieve(2) == [2]\n"
            ),
        },
        verify_cmd="python -m pytest test_sieve.py -q",
        problem_statement=(
            "sieve() in sieve.py is supposed to return all primes up to maxn but "
            "returns the wrong list; test_sieve.py fails. Fix the bug."
        ),
        why_this_case="QuixBugs 经典 any/all 误用，单行、输出错误（不挂死），给 sanity 桶一点外部出处。",
        expected_capability="读懂布尔量词语义并改对。",
        expected_without_component="无（sanity 题应在所有 condition 下都 resolved）。",
        reference_fix=(
            "sieve.py",
            "if any(n % p > 0 for p in primes):  # bug: should be all(...)",
            "if all(n % p > 0 for p in primes):",
        ),
    ),
]


# ===========================================================================
# repo_map（2）—— 多文件，问题描述不点名 bug 文件，逼 agent 自己定位
# ===========================================================================

_REPO_MAP = [
    CaseSpec(
        case_id="repomap-discount",
        bucket="repo_map",
        max_steps=18,
        files={
            "store/__init__.py": "",
            "store/models.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass\n"
                "class Product:\n"
                "    name: str\n"
                "    price: float\n"
            ),
            "store/pricing.py": (
                "def subtotal(items):\n"
                '    """items: list of (Product, qty)."""\n'
                "    return sum(p.price * q for p, q in items)\n"
            ),
            "store/discount.py": (
                "def apply_discount(total, pct):\n"
                "    # pct is a percentage, e.g. 10 means 10% off\n"
                "    return total + total * pct / 100  # bug: should subtract\n"
            ),
            "store/cart.py": (
                "from store.pricing import subtotal\n"
                "from store.discount import apply_discount\n\n\n"
                "def cart_total(items, discount_pct):\n"
                "    return apply_discount(subtotal(items), discount_pct)\n"
            ),
            "test_cart.py": (
                "from store.models import Product\n"
                "from store.cart import cart_total\n\n\n"
                "def test_discounted_total():\n"
                "    items = [(Product('a', 100.0), 2)]  # subtotal = 200\n"
                "    assert cart_total(items, 10) == 180.0  # 10% off\n"
            ),
        },
        verify_cmd="python -m pytest test_cart.py -q",
        problem_statement=(
            "test_cart.py fails: cart_total() returns more than the undiscounted "
            "total when a discount is applied (expected 180.0 for 10% off 200). "
            "Find and fix the bug."
        ),
        why_this_case="bug 藏在 5 文件包的 discount.py，描述只给失败现象不点名文件，必须先定位。",
        expected_capability="用 repo-map / 搜索在多文件中定位到 discount.apply_discount。",
        expected_without_component=(
            "no_repo_map → 少了仓库结构概览，要靠盲目 find/read 定位，"
            "steps_to_first_edit 明显变大，可能 max_steps_hit / 翻面。"
        ),
        reference_fix=(
            "store/discount.py",
            "return total + total * pct / 100  # bug: should subtract",
            "return total - total * pct / 100",
        ),
    ),
    CaseSpec(
        case_id="repomap-casefold",
        bucket="repo_map",
        max_steps=18,
        files={
            "textproc/__init__.py": "",
            "textproc/tokenize.py": (
                "def tokens(s):\n"
                "    return s.split()\n"
            ),
            "textproc/normalize.py": (
                "def normalize(word):\n"
                "    return word.upper()  # bug: should lowercase for case-insensitive counts\n"
            ),
            "textproc/counter.py": (
                "from collections import Counter\n"
                "from textproc.tokenize import tokens\n"
                "from textproc.normalize import normalize\n\n\n"
                "def word_counts(text):\n"
                "    return Counter(normalize(t) for t in tokens(text))\n"
            ),
            "test_counter.py": (
                "from textproc.counter import word_counts\n\n\n"
                "def test_case_insensitive():\n"
                "    c = word_counts('The the THE cat')\n"
                "    assert c['the'] == 3\n"
                "    assert c['cat'] == 1\n"
            ),
        },
        verify_cmd="python -m pytest test_counter.py -q",
        problem_statement=(
            "test_counter.py fails: word_counts() should count words "
            "case-insensitively (c['the'] should be 3) but doesn't. "
            "Find and fix the bug."
        ),
        why_this_case="bug 在 4 文件包深处的 normalize.py，描述不点名文件，考验跨文件定位。",
        expected_capability="顺着 counter→normalize 的调用链定位到真正出错的函数。",
        expected_without_component=(
            "no_repo_map → 无结构概览，定位 normalize 更慢，"
            "steps_to_first_edit 上升，可能 max_steps_hit。"
        ),
        reference_fix=(
            "textproc/normalize.py",
            "return word.upper()  # bug: should lowercase for case-insensitive counts",
            "return word.lower()",
        ),
    ),
]


# ===========================================================================
# reflection（2）—— 第一眼改法不完整 / 会引入回归，需测试反馈后再修
# ===========================================================================

_REFLECTION = [
    CaseSpec(
        case_id="reflect-clean-twostage",
        bucket="reflection",
        max_steps=18,
        files={
            "textclean.py": (
                "def clean(s):\n"
                '    """Return a cleaned version of s."""\n'
                "    return s.strip()  # incomplete\n"
            ),
            "test_textclean.py": (
                "from textclean import clean\n\n\n"
                "def test_clean():\n"
                "    assert clean('  Hello  ') == 'hello'\n"
                "    assert clean('A   B') == 'a b'\n"
            ),
        },
        verify_cmd="python -m pytest test_textclean.py -q",
        problem_statement=(
            "test_textclean.py has two failing assertions about clean(). "
            "Make both pass."
        ),
        why_this_case=(
            "最自然的第一步改法（加 .lower()）只过第一个 assert，第二个"
            "（折叠内部空白）仍失败——要靠测试反馈复盘后补第二处改动。"
        ),
        expected_capability="跑测试→读第二条失败→意识到第一次修不全→再改一次。",
        expected_without_component=(
            "no_reflection → 修完第一处后少了 [REFLECTION] 推动复查，"
            "更易过早 finish 或停步，留下第二个 assert 失败而翻面。"
        ),
        reference_fix=(
            "textclean.py",
            "    return s.strip()  # incomplete",
            "    return ' '.join(s.split()).lower()",
        ),
    ),
    CaseSpec(
        case_id="reflect-median-regression",
        bucket="reflection",
        max_steps=18,
        files={
            "stats.py": (
                "def median(nums):\n"
                "    nums = sorted(nums)\n"
                "    n = len(nums)\n"
                "    return nums[n // 2]  # bug: wrong for even-length lists\n"
            ),
            "test_stats.py": (
                "from stats import median\n\n\n"
                "def test_median():\n"
                "    assert median([3, 1, 2]) == 2        # odd\n"
                "    assert median([1, 2, 3, 4]) == 2.5   # even\n"
            ),
        },
        verify_cmd="python -m pytest test_stats.py -q",
        problem_statement=(
            "test_stats.py fails: median() is wrong for even-length lists. "
            "Make all tests pass without breaking the ones that already pass."
        ),
        why_this_case=(
            "只盯 even 的天真改法（恒取两个中间值平均）会把 odd 的 assert 弄挂，"
            "出现回归——需要测试反馈提示后兼顾两种奇偶。"
        ),
        expected_capability="改 even 后跑全测，发现 odd 回归，再调整到两种情况都对。",
        expected_without_component=(
            "no_reflection → 引入回归后缺少复盘提示，更可能交付半对的解"
            "（修好 even、挂掉 odd）而翻面。"
        ),
        reference_fix=(
            "stats.py",
            "    return nums[n // 2]  # bug: wrong for even-length lists",
            "    return (nums[(n - 1) // 2] + nums[n // 2]) / 2",
        ),
    ),
]


# ===========================================================================
# noise（1）—— 真实 bug 被一堆无关文件包围
# ===========================================================================

def _noise_files() -> dict[str, str]:
    """一堆语法正确但与 bug 无关的红鲱鱼文件，制造定位/上下文噪声。"""
    fillers = {
        "app/strings_util.py": "def shout(s):\n    return s.upper() + '!'\n",
        "app/list_util.py": "def dedupe(xs):\n    return list(dict.fromkeys(xs))\n",
        "app/math_util.py": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
        "app/dict_util.py": "def invert(d):\n    return {v: k for k, v in d.items()}\n",
        "app/time_util.py": "def to_minutes(seconds):\n    return seconds / 60\n",
        "app/path_util.py": "def basename(p):\n    return p.rsplit('/', 1)[-1]\n",
        "app/bool_util.py": "def xor(a, b):\n    return bool(a) != bool(b)\n",
        "app/range_util.py": "def inclusive(a, b):\n    return list(range(a, b + 1))\n",
    }
    return fillers


_NOISE = [
    CaseSpec(
        case_id="noise-temperature",
        bucket="noise",
        max_steps=18,
        files={
            "app/__init__.py": "",
            **_noise_files(),
            "app/temperature.py": (
                "def c_to_f(c):\n"
                "    return c * 5 / 9 + 32  # bug: ratio swapped, should be 9/5\n"
            ),
            "test_temperature.py": (
                "from app.temperature import c_to_f\n\n\n"
                "def test_c_to_f():\n"
                "    assert c_to_f(100) == 212\n"
                "    assert c_to_f(0) == 32\n"
            ),
        },
        verify_cmd="python -m pytest test_temperature.py -q",
        problem_statement=(
            "test_temperature.py fails: c_to_f(100) should be 212 but isn't. "
            "Find and fix the bug."
        ),
        why_this_case="真实 bug 只有一处，但被 8 个无关 util 文件包围，制造定位噪声与上下文膨胀。",
        expected_capability="在噪声文件中聚焦到 temperature.py，不被无关内容带偏。",
        expected_without_component=(
            "no_repo_map → 噪声里定位更难；no_token_budget → 无关内容更易挤占"
            "上下文。两者关掉时 steps / tokens 预计上升。"
        ),
        reference_fix=(
            "app/temperature.py",
            "    return c * 5 / 9 + 32  # bug: ratio swapped, should be 9/5",
            "    return c * 9 / 5 + 32",
        ),
    ),
]


# 全部诊断题，按桶排列
ALL_SPECS: list[CaseSpec] = _SANITY + _REPO_MAP + _REFLECTION + _NOISE


# ===========================================================================
# 构建：把 spec 落成隔离的 git 工作区，返回 EvalCase
# ===========================================================================

def _git_init(d: Path) -> None:
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.email=e@e.com", "-c", "user.name=e", "commit", "-qm", "buggy"]):
        subprocess.run(["git", *args], cwd=d, capture_output=True)


def _materialize(spec: CaseSpec) -> Path:
    """把 spec.files 写进一个新临时目录并 git init，返回该目录。"""
    d = Path(tempfile.mkdtemp(prefix=f"forge_{spec.case_id}_"))
    for rel, content in spec.files.items():
        target = d / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git_init(d)
    return d


def build_case(spec: CaseSpec) -> EvalCase:
    """把单个 spec 物化成 EvalCase（含 metadata）。"""
    d = _materialize(spec)
    return EvalCase(
        case_id=spec.case_id,
        repo_path=str(d),
        problem_statement=spec.problem_statement,
        verify_cmd=spec.verify_cmd,
        max_steps=spec.max_steps,
        bucket=spec.bucket,
        source=spec.source,
        source_id=spec.source_id,
        why_this_case=spec.why_this_case,
        expected_capability=spec.expected_capability,
        expected_without_component=spec.expected_without_component,
    )


def build_local_cases(
    n: int | None = None,
    buckets: list[str] | None = None,
) -> list[EvalCase]:
    """
    构建诊断集。

    Args:
        n:       只取前 n 题（None=全部）。保持与旧调用方（run_real）兼容。
        buckets: 只取这些桶（None=全部）。
    """
    specs = ALL_SPECS
    if buckets is not None:
        specs = [s for s in specs if s.bucket in buckets]
    if n is not None:
        specs = specs[:n]
    return [build_case(s) for s in specs]
