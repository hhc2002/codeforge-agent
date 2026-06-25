"""
tests/test_diagnostic_cases.py

诊断集自检：保证每道题"开箱即坏、打上 reference_fix 即好"，且 metadata 齐全。
这是数据集的护栏——case 一旦写错（buggy 版本其实能过，或 reference_fix 修不好），
消融结果就全无意义，所以在 CI 里钉死。
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from eval.local_cases import ALL_SPECS, CaseSpec, build_case

_ALLOWED_BUCKETS = {"sanity", "repo_map", "reflection", "noise"}


def _run_verify(repo_path: str, verify_cmd: str) -> subprocess.CompletedProcess:
    cmd = verify_cmd.replace("python", sys.executable, 1)
    return subprocess.run(
        cmd, shell=True, cwd=repo_path,
        capture_output=True, text=True, timeout=60,
    )


def _apply_reference_fix(repo_path: str, spec: CaseSpec) -> None:
    from pathlib import Path
    rel, old, new = spec.reference_fix
    path = Path(repo_path) / rel
    content = path.read_text(encoding="utf-8")
    assert content.count(old) == 1, (
        f"{spec.case_id}: reference_fix old_string 在 {rel} 中应恰好出现 1 次，"
        f"实际 {content.count(old)} 次"
    )
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


@pytest.mark.parametrize("spec", ALL_SPECS, ids=[s.case_id for s in ALL_SPECS])
class TestDiagnosticSpec:
    def test_buggy_fails_then_fix_resolves(self, spec):
        case = build_case(spec)
        # 1. buggy 版本必须失败（否则这道题根本测不出修复能力）
        before = _run_verify(case.repo_path, case.verify_cmd)
        assert before.returncode != 0, (
            f"{spec.case_id}: buggy 版本竟然通过了 verify_cmd，case 失效。\n"
            f"{before.stdout}\n{before.stderr}"
        )
        # 2. 打上 reference_fix 后必须通过（证明这道题确实可解、fix 正确）
        _apply_reference_fix(case.repo_path, spec)
        after = _run_verify(case.repo_path, case.verify_cmd)
        assert after.returncode == 0, (
            f"{spec.case_id}: 打上 reference_fix 后仍失败，fix 不正确。\n"
            f"{after.stdout}\n{after.stderr}"
        )

    def test_metadata_complete(self, spec):
        assert spec.bucket in _ALLOWED_BUCKETS
        for fld in ("problem_statement", "verify_cmd", "why_this_case",
                     "expected_capability", "expected_without_component"):
            assert getattr(spec, fld).strip(), f"{spec.case_id}: 缺 {fld}"


def test_case_ids_unique():
    ids = [s.case_id for s in ALL_SPECS]
    assert len(ids) == len(set(ids))


def test_bucket_coverage():
    buckets = {s.bucket for s in ALL_SPECS}
    # repo_map / reflection 是本轮能真正被消融驱动的两个桶，必须有题
    assert {"sanity", "repo_map", "reflection"} <= buckets
