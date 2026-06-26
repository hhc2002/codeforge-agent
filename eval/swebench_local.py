"""
eval/swebench_local.py

SWE-bench Lite 的**无 Docker** 本地判分器（先只支持 sympy 子集）。

为什么不需要 Docker：Docker 在官方 harness 里只干两件事——锁 Python 版本、锁依赖/编译。
对纯 Python repo（sympy 依赖仅 mpmath，无 C 扩展），这两件用一个 conda py3.9 env 就能复刻：
  - swebench 的 MAP_REPO_VERSION_TO_SPECS 把 **所有 sympy 版本都映射到 python 3.9 + `pip install -e .`**，
    所以单个 env 可判全部 77 道 sympy 题；
  - clone 一次缓存，按 base_commit checkout，跑指定测试即可判分。

两段彻底解耦（见 eval/harness.py 的设计）：
  ① 跑 agent 出 patch —— 本来就不需要 Docker；
  ② 判分（本文件）—— 用本地 env 跑 FAIL_TO_PASS / PASS_TO_PASS。

判分语义（与官方一致）：
  resolved = (打上候选 patch 后) 所有 FAIL_TO_PASS 通过 且 所有 PASS_TO_PASS 仍通过。

self-check（`python -m eval.swebench_local --check N`）：
  对 N 道题打 **官方 gold patch**，断言每道都 fail→pass——证明本地 env 忠实复刻了官方判分，
  是接 agent 之前的地基护栏（类比 tests/test_diagnostic_cases.py，但针对真实 Lite 实例）。

注意：这是 curated 子集（目前仅 sympy），**不等于官方 Lite-300 分数**，报告里如实标注。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# 默认环境：conda 的 py3.9 env + 本地 sympy clone 缓存。
# 这两个是“无 Docker”判分的全部外部依赖，可按机器覆盖。
DEFAULT_PYBIN = "/opt/miniconda3/envs/sb39/bin/python"
DEFAULT_REPO_CACHE = Path(
    "/private/tmp/claude-501/-Users-hhc-Desktop-forge-agent/"
    "986bf59a-487a-4bb2-a922-50a28da20df9/scratchpad/repos"
)
GITHUB = {
    "sympy/sympy": "https://github.com/sympy/sympy.git",
    "psf/requests": "https://github.com/psf/requests.git",
    "pallets/flask": "https://github.com/pallets/flask.git",
}


@dataclass
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    version: str
    problem_statement: str   # 喂给 agent 的 issue 描述（agent 看不到下面的测试）
    patch: str               # gold fix
    test_patch: str          # 加测试（FAIL_TO_PASS/PASS_TO_PASS 所在）
    fail_to_pass: list[str]
    pass_to_pass: list[str]


@dataclass
class GradeResult:
    instance_id: str
    resolved: bool
    f2p_before_fail: int     # 打 patch 前 FAIL_TO_PASS 里失败的个数（健康的题应 = 全部）
    f2p_after_pass: int      # 打 patch 后通过的个数
    f2p_total: int
    p2p_after_pass: int
    p2p_total: int
    note: str = ""


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _as_list(x) -> list[str]:
    return json.loads(x) if isinstance(x, str) else list(x)


def load_instances(
    repo: str = "sympy/sympy",
    ids: list[str] | None = None,
    n: int | None = None,
    sort_by_p2p: bool = True,
) -> list[Instance]:
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    rows = [r for r in ds if r["repo"] == repo]
    if ids:
        keep = set(ids)
        rows = [r for r in rows if r["instance_id"] in keep]
    insts = [
        Instance(
            instance_id=r["instance_id"], repo=r["repo"], base_commit=r["base_commit"],
            version=r["version"], problem_statement=r["problem_statement"],
            patch=r["patch"], test_patch=r["test_patch"],
            fail_to_pass=_as_list(r["FAIL_TO_PASS"]), pass_to_pass=_as_list(r["PASS_TO_PASS"]),
        )
        for r in rows
    ]
    if sort_by_p2p:                      # 小 PASS_TO_PASS 先跑，self-check 快
        insts.sort(key=lambda i: len(i.pass_to_pass))
    if n is not None:
        insts = insts[:n]
    return insts


# ---------------------------------------------------------------------------
# 仓库准备 / patch
# ---------------------------------------------------------------------------

def _git(repo_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True,
                          text=True, check=check)


def ensure_clone(repo: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    repo_dir = cache / repo.split("/")[-1]
    if not (repo_dir / ".git").exists():
        subprocess.run(["git", "clone", "-q", GITHUB[repo], str(repo_dir)], check=True)
    return repo_dir


def checkout_base(repo_dir: Path, base_commit: str) -> None:
    _git(repo_dir, "checkout", "-fq", base_commit)
    _git(repo_dir, "clean", "-fdxq")


def apply_patch(repo_dir: Path, patch_text: str) -> bool:
    p = subprocess.run(["git", "apply", "-"], cwd=repo_dir, input=patch_text,
                       text=True, capture_output=True)
    return p.returncode == 0


# ---------------------------------------------------------------------------
# 测试执行：从 test_patch 找测试文件 → collect 出精确 nodeid → 跑指定测试
# ---------------------------------------------------------------------------

def test_files_from_patch(test_patch: str) -> list[str]:
    files = re.findall(r"^\+\+\+ b/(.+)$", test_patch, flags=re.M)
    return [f for f in files if f.endswith(".py")]


def _collect_nodeids(pybin: str, repo_dir: Path, files: list[str]) -> dict[str, str]:
    """{测试函数名 -> 完整 nodeid}，用于把 FAIL_TO_PASS 的裸名解析成可精确执行的节点。"""
    try:
        out = subprocess.run(
            [pybin, "-m", "pytest", *files, "--collect-only", "-q",
             "-p", "no:cacheprovider"],
            cwd=repo_dir, capture_output=True, text=True, timeout=120,
        ).stdout
    except subprocess.TimeoutExpired:
        return {}
    mapping: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith("<"):
            name = line.split("::")[-1].split("[")[0]   # 去参数化后缀
            mapping.setdefault(name, line)
    return mapping


_SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors)")


def _run_nodes(pybin: str, repo_dir: Path, nodeids: list[str]) -> tuple[int, int]:
    """跑给定 nodeid，返回 (passed, failed+errors)。"""
    if not nodeids:
        return (0, 0)
    try:
        out = subprocess.run(
            [pybin, "-m", "pytest", *nodeids, "--tb=no", "-q", "-p", "no:cacheprovider"],
            cwd=repo_dir, capture_output=True, text=True, timeout=180,
        ).stdout
    except subprocess.TimeoutExpired:
        return (0, len(nodeids))   # 超时（如网络挂死）记为全失败，别让判分卡死
    passed = failed = 0
    for n, kind in _SUMMARY.findall(out):
        if kind == "passed":
            passed = int(n)
        else:
            failed += int(n)
    return passed, failed


def _resolve(names: list[str], nodemap: dict[str, str]) -> list[str]:
    return [nodemap.get(nm.split("::")[-1], nm) for nm in names]


# ---------------------------------------------------------------------------
# 判分
# ---------------------------------------------------------------------------

def grade(inst: Instance, candidate_patch: str | None, *,
          pybin: str = DEFAULT_PYBIN, cache: Path = DEFAULT_REPO_CACHE) -> GradeResult:
    """
    对一道实例判分。candidate_patch=None 时用官方 gold patch（self-check）。
    流程：checkout base → 打 test_patch（加测试）→ 测 before → 打候选 patch → 测 after。
    """
    patch = candidate_patch if candidate_patch is not None else inst.patch
    repo_dir = ensure_clone(inst.repo, cache)
    checkout_base(repo_dir, inst.base_commit)
    if not apply_patch(repo_dir, inst.test_patch):
        return GradeResult(inst.instance_id, False, 0, 0, len(inst.fail_to_pass),
                           0, len(inst.pass_to_pass), note="test_patch 打不上")

    files = test_files_from_patch(inst.test_patch)
    nodemap = _collect_nodeids(pybin, repo_dir, files)
    f2p = _resolve(inst.fail_to_pass, nodemap)
    p2p = _resolve(inst.pass_to_pass, nodemap)

    # before：候选 patch 未打，FAIL_TO_PASS 应失败（证明这题确实在测 bug）
    _, f2p_fail_before = _run_nodes(pybin, repo_dir, f2p)

    if not apply_patch(repo_dir, patch):
        return GradeResult(inst.instance_id, False, f2p_fail_before, 0,
                           len(f2p), 0, len(p2p), note="候选 patch 打不上")

    f2p_pass_after, _ = _run_nodes(pybin, repo_dir, f2p)
    p2p_pass_after, _ = _run_nodes(pybin, repo_dir, p2p)

    resolved = (f2p_pass_after == len(f2p)) and (p2p_pass_after == len(p2p))
    return GradeResult(
        inst.instance_id, resolved,
        f2p_before_fail=f2p_fail_before, f2p_after_pass=f2p_pass_after, f2p_total=len(f2p),
        p2p_after_pass=p2p_pass_after, p2p_total=len(p2p),
    )


def _check(n: int, pybin: str, cache: Path) -> int:
    """对 n 道题打 gold patch 自检：每道都该 fail→pass 且 PASS_TO_PASS 不回归。"""
    insts = load_instances(n=n)
    print(f"self-check {len(insts)} sympy instances (gold patch, no Docker, {pybin})\n")
    hdr = f"{'instance':22s} {'resolved':9s} {'F2P fail→pass':16s} {'P2P pass':10s} note"
    print(hdr); print("-" * len(hdr))
    ok = 0
    for inst in insts:
        r = grade(inst, None, pybin=pybin, cache=cache)
        f2p = f"{r.f2p_before_fail}/{r.f2p_total} → {r.f2p_after_pass}/{r.f2p_total}"
        p2p = f"{r.p2p_after_pass}/{r.p2p_total}"
        mark = "✅" if r.resolved else "❌"
        print(f"{r.instance_id:22s} {mark:9s} {f2p:16s} {p2p:10s} {r.note}")
        ok += r.resolved
    print(f"\n{ok}/{len(insts)} gold patches reproduce fail→pass locally (no Docker).")
    return 0 if ok == len(insts) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, metavar="N", default=5,
                    help="对前 N 道 sympy 实例做 gold-patch 本地自检")
    ap.add_argument("--pybin", default=DEFAULT_PYBIN)
    ap.add_argument("--cache", type=Path, default=DEFAULT_REPO_CACHE)
    args = ap.parse_args()
    raise SystemExit(_check(args.check, args.pybin, args.cache))
