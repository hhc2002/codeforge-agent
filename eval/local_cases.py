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


# ===========================================================================
# 加难版（harder set）—— 提高首轮失败率，给 reflection / repo_map 消融造信号
#   现有 7 题在小样本上 100% resolved（天花板效应），condition 间差异被压平。
#   这批刻意：① reflection 桶——"第一眼改法"只过部分 assert / 引回归，逼测试反馈
#   后再改；② repo_map 桶——加深调用链，症状离 bug 2 跳，描述不点名文件；
#   ③ noise 桶——同名诱饵 / 红鲱鱼包围真实 bug。
#   reflection 指标用 on/off 配对翻盘看（救回 / 打挂），token 两边如实记、不预设代价。
# ===========================================================================

_HARD_REFLECTION = [
    CaseSpec(
        case_id="reflect-slugify",
        bucket="reflection",
        max_steps=20,
        files={
            "slugify.py": (
                "def slugify(s):\n"
                '    """Turn a title into a URL slug."""\n'
                "    return s.lower().replace(' ', '-')  # incomplete: only handles single spaces\n"
            ),
            "test_slugify.py": (
                "from slugify import slugify\n\n\n"
                "def test_slugify():\n"
                "    assert slugify('Hello World') == 'hello-world'\n"
                "    assert slugify('  Hello   World  ') == 'hello-world'\n"
                "    assert slugify('Hello, World!') == 'hello-world'\n"
                "    assert slugify('Already--Hyphenated') == 'already-hyphenated'\n"
            ),
        },
        verify_cmd="python -m pytest test_slugify.py -q",
        problem_statement=(
            "test_slugify.py has several failing assertions about slugify(): it must "
            "lowercase, collapse runs of whitespace, drop punctuation, and avoid "
            "repeated hyphens. Make all assertions pass."
        ),
        why_this_case=(
            "最自然的第一步（lower + 空格换连字符）只过第一个 assert，余下三个"
            "（折叠多空格 / 去标点 / 压缩连字符）逐个暴露，需多轮测试反馈才补全。"
        ),
        expected_capability="跑测试→逐条读失败→意识到单次改法不全→迭代到四种情况都过。",
        expected_without_component=(
            "no_reflection → 改完第一处就易 finish，留下后面几个 assert 失败而翻面。"
        ),
        reference_fix=(
            "slugify.py",
            "    return s.lower().replace(' ', '-')  # incomplete: only handles single spaces",
            "    return '-'.join(''.join(c if c.isalnum() else ' ' for c in s.lower()).split())",
        ),
    ),
    CaseSpec(
        case_id="reflect-truncate",
        bucket="reflection",
        max_steps=18,
        files={
            "truncate.py": (
                "def truncate(s, n):\n"
                '    """Shorten s to at most n characters; long strings end with an ellipsis."""\n'
                '    return s[:n] + "..."  # incomplete\n'
            ),
            "test_truncate.py": (
                "from truncate import truncate\n\n\n"
                "def test_truncate():\n"
                "    assert truncate('hello world', 8) == 'hello...'\n"
                "    assert truncate('hi', 8) == 'hi'\n"
                "    assert truncate('abcdefgh', 8) == 'abcdefgh'\n"
            ),
        },
        verify_cmd="python -m pytest test_truncate.py -q",
        problem_statement=(
            "test_truncate.py fails: truncate() must keep short strings unchanged and "
            "make truncated output (including the '...') fit within n characters. "
            "Make all assertions pass."
        ),
        why_this_case=(
            "天真改法（只补 len<=n 的早返回）仍过不了第一个 assert——因为省略号要算进 n，"
            "得切到 n-3，需再跑一轮测试才发现。"
        ),
        expected_capability="先补短串早返回，再从失败现象意识到省略号占位、调 slice 到 n-3。",
        expected_without_component=(
            "no_reflection → 补完早返回就易停步，留下第一个 assert（长度超界）失败而翻面。"
        ),
        reference_fix=(
            "truncate.py",
            '    return s[:n] + "..."  # incomplete',
            '    return s if len(s) <= n else s[: n - 3] + "..."',
        ),
    ),
    CaseSpec(
        case_id="reflect-average-empty",
        bucket="reflection",
        max_steps=18,
        files={
            "average.py": (
                "def average(nums):\n"
                '    """Average of nums, skipping None entries; empty input -> 0.0."""\n'
                "    return sum(nums) / len(nums)  # bug: crashes on None and on empty input\n"
            ),
            "test_average.py": (
                "from average import average\n\n\n"
                "def test_average():\n"
                "    assert average([1, 2, 3]) == 2.0\n"
                "    assert average([1, None, 3]) == 2.0\n"
                "    assert average([]) == 0.0\n"
                "    assert average([None, None]) == 0.0\n"
            ),
        },
        verify_cmd="python -m pytest test_average.py -q",
        problem_statement=(
            "test_average.py fails: average() should skip None entries and return 0.0 "
            "for empty (or all-None) input, never crash. Make all assertions pass."
        ),
        why_this_case=(
            "第一步（过滤 None）解决 TypeError，却在空列表 / 全 None 时变成 ZeroDivisionError，"
            "回归式新失败需测试反馈后再加空值守卫。"
        ),
        expected_capability="先滤 None 再跑测，发现空列表崩，补 empty→0.0 守卫。",
        expected_without_component=(
            "no_reflection → 滤完 None 就交付，留下空列表分支崩溃而翻面。"
        ),
        reference_fix=(
            "average.py",
            "    return sum(nums) / len(nums)  # bug: crashes on None and on empty input",
            "    return (lambda v: sum(v) / len(v) if v else 0.0)([x for x in nums if x is not None])",
        ),
    ),
    CaseSpec(
        case_id="reflect-parse-bool",
        bucket="reflection",
        max_steps=18,
        files={
            "parsebool.py": (
                "def to_bool(s):\n"
                '    """Parse a human-written truthy string into a bool."""\n'
                "    return s == 'true'  # incomplete: only the exact string 'true'\n"
            ),
            "test_parsebool.py": (
                "from parsebool import to_bool\n\n\n"
                "def test_to_bool():\n"
                "    assert to_bool('true') is True\n"
                "    assert to_bool('True') is True\n"
                "    assert to_bool('1') is True\n"
                "    assert to_bool('yes') is True\n"
                "    assert to_bool('no') is False\n"
                "    assert to_bool('0') is False\n"
                "    assert to_bool('') is False\n"
            ),
        },
        verify_cmd="python -m pytest test_parsebool.py -q",
        problem_statement=(
            "test_parsebool.py fails: to_bool() must accept several truthy spellings "
            "(case-insensitive 'true', '1', 'yes') and treat everything else as False. "
            "Make all assertions pass."
        ),
        why_this_case=(
            "多个可接受拼写（大小写 / '1' / 'yes'）让单次 == 比较必然不全，"
            "得逐条失败反馈后扩成集合归一化。"
        ),
        expected_capability="从一条条失败的拼写归纳出 strip+lower+成员判断。",
        expected_without_component=(
            "no_reflection → 改成覆盖 'True' 就停，剩下 '1'/'yes' 等分支失败而翻面。"
        ),
        reference_fix=(
            "parsebool.py",
            "    return s == 'true'  # incomplete: only the exact string 'true'",
            "    return s.strip().lower() in {'true', '1', 'yes', 'y', 'on'}",
        ),
    ),
    CaseSpec(
        case_id="reflect-ordinal-teens",
        bucket="reflection",
        max_steps=18,
        files={
            "ordinal.py": (
                "def ordinal(n):\n"
                '    """Return n with its English ordinal suffix, e.g. 1 -> 1st."""\n'
                "    return str(n) + 'th'  # incomplete: ignores st/nd/rd and the 11-13 rule\n"
            ),
            "test_ordinal.py": (
                "from ordinal import ordinal\n\n\n"
                "def test_ordinal():\n"
                "    assert ordinal(1) == '1st'\n"
                "    assert ordinal(2) == '2nd'\n"
                "    assert ordinal(3) == '3rd'\n"
                "    assert ordinal(4) == '4th'\n"
                "    assert ordinal(11) == '11th'\n"
                "    assert ordinal(12) == '12th'\n"
                "    assert ordinal(13) == '13th'\n"
                "    assert ordinal(21) == '21st'\n"
                "    assert ordinal(113) == '113th'\n"
            ),
        },
        verify_cmd="python -m pytest test_ordinal.py -q",
        problem_statement=(
            "test_ordinal.py fails: ordinal() must append the right English suffix "
            "(1st, 2nd, 3rd, 4th ...) including the 11th/12th/13th exception. "
            "Make all assertions pass."
        ),
        why_this_case=(
            "按末位数字改（1→st,2→nd,3→rd）能过 1/2/3/21，却在 11/12/13/113 上翻车，"
            "需测试反馈后补 teens 例外——经典两阶段。"
        ),
        expected_capability="先按末位补 st/nd/rd，再从 11-13 失败补 %100 的 teens 守卫。",
        expected_without_component=(
            "no_reflection → 补完末位规则就交付，留下 11/12/13 类 assert 失败而翻面。"
        ),
        reference_fix=(
            "ordinal.py",
            "    return str(n) + 'th'  # incomplete: ignores st/nd/rd and the 11-13 rule",
            "    return str(n) + ('th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'))",
        ),
    ),
]


_HARD_REPO_MAP = [
    CaseSpec(
        case_id="repomap-invoice-tax",
        bucket="repo_map",
        max_steps=22,
        files={
            "shop/__init__.py": "",
            "shop/catalog.py": (
                "PRICES = {'A': 10.0, 'B': 25.0}\n\n\n"
                "def unit_price(sku):\n"
                "    return PRICES[sku]\n"
            ),
            "shop/tax.py": (
                "def with_tax(amount, rate):\n"
                "    # rate is a fraction, e.g. 0.2 for 20%\n"
                "    return amount * rate  # bug: returns only the tax, not amount + tax\n"
            ),
            "shop/line_items.py": (
                "from shop.catalog import unit_price\n\n\n"
                "def line_total(sku, qty):\n"
                "    return unit_price(sku) * qty\n"
            ),
            "shop/invoice.py": (
                "from shop.line_items import line_total\n"
                "from shop.tax import with_tax\n\n\n"
                "def invoice_total(items, rate):\n"
                '    """items: list of (sku, qty)."""\n'
                "    sub = sum(line_total(sku, qty) for sku, qty in items)\n"
                "    return with_tax(sub, rate)\n"
            ),
            "test_invoice.py": (
                "from shop.invoice import invoice_total\n\n\n"
                "def test_invoice_total():\n"
                "    # 2xA(10)=20 + 1xB(25)=25 -> subtotal 45; +20% tax -> 54.0\n"
                "    assert invoice_total([('A', 2), ('B', 1)], 0.2) == 54.0\n"
            ),
        },
        verify_cmd="python -m pytest test_invoice.py -q",
        problem_statement=(
            "test_invoice.py fails: invoice_total([('A', 2), ('B', 1)], 0.2) should be "
            "54.0 (subtotal 45 plus 20% tax) but is far too small. Find and fix the bug."
        ),
        why_this_case="症状在最外层 invoice_total，真正的 bug 在 2 跳之外的 tax.with_tax，描述不点名文件。",
        expected_capability="顺 invoice→tax 调用链定位到 with_tax 只返回了税额。",
        expected_without_component=(
            "no_repo_map → 缺仓库结构概览，要盲翻 6 个文件才摸到 tax.py，"
            "steps_to_first_edit 上升，可能 max_steps_hit。"
        ),
        reference_fix=(
            "shop/tax.py",
            "    return amount * rate  # bug: returns only the tax, not amount + tax",
            "    return amount + amount * rate",
        ),
    ),
    CaseSpec(
        case_id="repomap-etl-clean",
        bucket="repo_map",
        max_steps=22,
        files={
            "etl/__init__.py": "",
            "etl/source.py": (
                "def raw_rows():\n"
                "    return [{'name': '  alice ', 'age': '30'},\n"
                "            {'name': 'BOB', 'age': '25'}]\n"
            ),
            "etl/clean.py": (
                "def clean_name(name):\n"
                "    return name.strip().upper()  # bug: should be title-case, not upper\n"
            ),
            "etl/cast.py": (
                "def to_int(s):\n"
                "    return int(s)\n"
            ),
            "etl/transform.py": (
                "from etl.clean import clean_name\n"
                "from etl.cast import to_int\n\n\n"
                "def transform(row):\n"
                "    return {'name': clean_name(row['name']), 'age': to_int(row['age'])}\n"
            ),
            "etl/pipeline.py": (
                "from etl.source import raw_rows\n"
                "from etl.transform import transform\n\n\n"
                "def run():\n"
                "    return [transform(r) for r in raw_rows()]\n"
            ),
            "test_pipeline.py": (
                "from etl.pipeline import run\n\n\n"
                "def test_run():\n"
                "    assert run() == [{'name': 'Alice', 'age': 30},\n"
                "                     {'name': 'Bob', 'age': 25}]\n"
            ),
        },
        verify_cmd="python -m pytest test_pipeline.py -q",
        problem_statement=(
            "test_pipeline.py fails: run() should produce names in title-case "
            "('Alice', 'Bob') but the casing comes out wrong. Find and fix the bug."
        ),
        why_this_case="管道输出名字大小写错，bug 藏在 pipeline→transform→clean 链末端的 clean.py。",
        expected_capability="顺数据流定位到 clean_name 用了 upper 而非 title。",
        expected_without_component=(
            "no_repo_map → 6 文件里缺结构概览，定位 clean.py 更慢，steps 上升。"
        ),
        reference_fix=(
            "etl/clean.py",
            "    return name.strip().upper()  # bug: should be title-case, not upper",
            "    return name.strip().title()",
        ),
    ),
    CaseSpec(
        case_id="repomap-acl-rank",
        bucket="repo_map",
        max_steps=20,
        files={
            "acl/__init__.py": "",
            "acl/roles.py": (
                "RANK = {'guest': 0, 'member': 1, 'editor': 1, 'admin': 3}"
                "  # bug: editor should outrank member\n\n\n"
                "def rank(role):\n"
                "    return RANK.get(role, 0)\n"
            ),
            "acl/policy.py": (
                "from acl.roles import rank\n\n\n"
                "def allowed(user_role, needed_role):\n"
                "    return rank(user_role) >= rank(needed_role)\n"
            ),
            "acl/handlers.py": (
                "from acl.policy import allowed\n\n\n"
                "def can_edit(user_role):\n"
                "    return allowed(user_role, 'editor')\n\n\n"
                "def can_admin(user_role):\n"
                "    return allowed(user_role, 'admin')\n"
            ),
            "test_acl.py": (
                "from acl.handlers import can_edit, can_admin\n\n\n"
                "def test_acl():\n"
                "    assert can_edit('editor') is True\n"
                "    assert can_edit('member') is False\n"
                "    assert can_admin('admin') is True\n"
                "    assert can_admin('editor') is False\n"
            ),
        },
        verify_cmd="python -m pytest test_acl.py -q",
        problem_statement=(
            "test_acl.py fails: can_edit('member') returns True, but a 'member' must "
            "not be allowed to edit (only 'editor' and above). Find and fix the bug."
        ),
        why_this_case="症状在 handlers.can_edit，真正的错在 2 跳外 roles.py 的 RANK 表（editor 权重写成与 member 相同）。",
        expected_capability="顺 handlers→policy→roles 链定位到 RANK 数据表，而非误改比较逻辑。",
        expected_without_component=(
            "no_repo_map → 容易只盯 policy 的比较运算符，错过真正出错的 roles.py 数据表。"
        ),
        reference_fix=(
            "acl/roles.py",
            "RANK = {'guest': 0, 'member': 1, 'editor': 1, 'admin': 3}  # bug: editor should outrank member",
            "RANK = {'guest': 0, 'member': 1, 'editor': 2, 'admin': 3}",
        ),
    ),
]


_HARD_NOISE = [
    CaseSpec(
        case_id="noise-parser-versions",
        bucket="noise",
        max_steps=20,
        files={
            "pkg/__init__.py": "",
            "pkg/parser.py": (
                "def parse_kv(s):\n"
                "    k, v = s.split('=', 1)\n"
                "    return k.strip(), v.strip()\n"
            ),
            "pkg/parser_legacy.py": (
                "def parse_pairs(text):\n"
                "    return [p for p in text.split(';') if p]\n"
            ),
            "pkg/parser_utils.py": (
                "def is_blank(line):\n"
                "    return not line.strip()\n"
            ),
            "pkg/parser_v2.py": (
                "def parse_csv_row(line):\n"
                "    return line.split(';')  # bug: CSV columns are comma-separated\n"
            ),
            "test_parser_v2.py": (
                "from pkg.parser_v2 import parse_csv_row\n\n\n"
                "def test_parse_csv_row():\n"
                "    assert parse_csv_row('a,b,c') == ['a', 'b', 'c']\n"
            ),
        },
        verify_cmd="python -m pytest test_parser_v2.py -q",
        problem_statement=(
            "test_parser_v2.py fails: parse_csv_row('a,b,c') should return "
            "['a', 'b', 'c'] but doesn't. Find and fix the bug."
        ),
        why_this_case="多个名字相近的 parser_*.py（parser / parser_legacy / parser_utils）做诱饵，搜 'parse' 命中一堆，需聚焦到真正被测的 parser_v2。",
        expected_capability="不被同名诱饵带偏，定位到 test 实际 import 的 parser_v2.parse_csv_row。",
        expected_without_component=(
            "no_repo_map → 同名文件多，靠盲搜更易翻错文件，steps 上升。"
        ),
        reference_fix=(
            "pkg/parser_v2.py",
            "    return line.split(';')  # bug: CSV columns are comma-separated",
            "    return line.split(',')",
        ),
    ),
    CaseSpec(
        case_id="noise-percent-of",
        bucket="noise",
        max_steps=20,
        files={
            "app/__init__.py": "",
            **_noise_files(),
            "app/percent_util.py": (
                "def pct_of(part, whole):\n"
                "    return part / whole  # bug: a percentage needs * 100\n"
            ),
            "test_percent.py": (
                "from app.percent_util import pct_of\n\n\n"
                "def test_pct_of():\n"
                "    assert pct_of(1, 4) == 25.0\n"
                "    assert pct_of(3, 4) == 75.0\n"
            ),
        },
        verify_cmd="python -m pytest test_percent.py -q",
        problem_statement=(
            "test_percent.py fails: pct_of(1, 4) should be 25.0 (a percentage) but "
            "isn't. Find and fix the bug."
        ),
        why_this_case="真实 bug 只一处，被 8 个 *_util.py 诱饵包围，搜 'util' 全命中，制造定位与上下文噪声。",
        expected_capability="在一堆 *_util 中聚焦到 percent_util.pct_of。",
        expected_without_component=(
            "no_repo_map → 噪声里定位更难；no_token_budget → 无关 util 更易挤占上下文。"
        ),
        reference_fix=(
            "app/percent_util.py",
            "    return part / whole  # bug: a percentage needs * 100",
            "    return part / whole * 100",
        ),
    ),
]


# 全部诊断题，按桶排列（基础 7 + 加难 10）
ALL_SPECS: list[CaseSpec] = (
    _SANITY + _REPO_MAP + _REFLECTION + _NOISE
    + _HARD_REFLECTION + _HARD_REPO_MAP + _HARD_NOISE
)


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
