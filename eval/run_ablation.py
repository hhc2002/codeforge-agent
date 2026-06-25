"""
eval/run_ablation.py

Smoke 消融：在本地诊断集上，对同一批 case 跑多个 condition（每个只关一个开关），
逐题配对看翻面，按 bucket 分组出表。

目标不是统计显著（这点题做不到），而是**验证 harness 能否揭示 agent 行为**：
关掉某模块后，预期翻面的桶（如 no_repo_map 影响 repo_map 桶）是否真的翻面。

跑法：
    # (a) 零 API 管道自检：每题用 MockBackend 直接打 reference_fix，
    #     应在所有 condition 下都 ✓，证明 runner / 判分 / 开关注入全通。
    .venv/bin/python -m eval.run_ablation --mock

    # (b) 真实模型，temperature=0（压采样噪声）：
    GEMINI_API_KEY=... PATH="$PWD/.venv/bin:$PATH" \
        .venv/bin/python -m eval.run_ablation \
            --provider gemini --model gemini-2.5-flash --temperature 0

复盘建议：smoke 阶段默认 repeats=1。对照表里**任何翻面的 case，建议
用 --repeats 3 复跑确认不是采样噪声**，再采信（不预付 3×，只对存疑项付）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from agent.task import Action, ActionType, ToolCall
from eval.harness import RunRecord, aggregate, run_case
from eval.local_cases import ALL_SPECS, CaseSpec, build_case
from llm.base import MockBackend


# condition -> AgentConfig 覆盖（每个只关一个开关）
CONDITIONS: dict[str, dict] = {
    "full":              {},
    "no_repo_map":       {"enable_repo_map": False},
    "no_reflection":     {"enable_reflection": False},
    "no_loop_detection": {"enable_loop_detection": False},
    "no_token_budget":   {"enable_token_budget": False},
}

_COND_SHORT = {
    "full": "full",
    "no_repo_map": "-repomap",
    "no_reflection": "-reflect",
    "no_loop_detection": "-loop",
    "no_token_budget": "-budget",
}


def _mock_backend_for(spec: CaseSpec) -> MockBackend:
    """脚本：直接 file_edit 打上 reference_fix，再 finish。用于零 API 管道自检。"""
    rel, old, new = spec.reference_fix
    script = [
        Action(
            action_type=ActionType.TOOL_CALL,
            thought="apply the known reference fix",
            tool_call=ToolCall(name="file_edit",
                               params={"path": rel, "old_string": old, "new_string": new}),
        ),
        Action(action_type=ActionType.FINISH, thought="done", message="applied reference fix"),
    ]
    return MockBackend(script)


def _run_cell(spec, cond_overrides, *, mock, real_backend, log_dir, repeats):
    """跑一个 (case, condition) 单元 repeats 次，返回代表性 RunRecord + 翻面统计。"""
    resolved_count = 0
    last: RunRecord | None = None
    for _ in range(repeats):
        case = build_case(spec)  # 每次全新工作区，避免跨 run 污染
        backend = _mock_backend_for(spec) if mock else real_backend
        rec = run_case(case, backend, log_dir=log_dir, config_overrides=cond_overrides)
        resolved_count += int(rec.resolved)
        last = rec
    resolved = resolved_count * 2 >= repeats  # 多数判定
    flaky = 0 < resolved_count < repeats
    return last, resolved, resolved_count, flaky


def _cell_str(resolved: bool, status: str, flaky: bool) -> str:
    mark = "✓" if resolved else "✗"
    if flaky:
        mark += "~"
    if not resolved and status not in ("success",):
        # 失败时附状态缩写，便于看是 max_steps 还是 gave_up
        abbr = {"max_steps": "MX", "gave_up": "GU", "failed": "ER"}.get(status, status[:2])
        return f"{mark}({abbr})"
    return mark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="零 API 管道自检（reference_fix 直填）")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="采样温度，默认 0（压噪声、可复现）")
    ap.add_argument("--repeats", type=int, default=1, help="每个 cell 跑几次（多数判定）")
    ap.add_argument("--buckets", nargs="*", default=None, help="只跑这些桶")
    ap.add_argument("--conditions", nargs="*", default=None, help="只跑这些 condition")
    ap.add_argument("--log-dir", default="./logs/ablation")
    args = ap.parse_args()

    specs = [s for s in ALL_SPECS if args.buckets is None or s.bucket in args.buckets]
    conditions = {k: v for k, v in CONDITIONS.items()
                  if args.conditions is None or k in args.conditions}
    if "full" not in conditions:
        conditions = {"full": {}, **conditions}  # full 永远作为配对基线

    real_backend = None
    if not args.mock:
        from llm.router import create_backend
        real_backend = create_backend(
            provider=args.provider, model=args.model,
            max_tokens=4096, temperature=args.temperature,
        )

    mode = "MOCK (reference_fix)" if args.mock else f"{args.provider}/{args.model} temp={args.temperature}"
    print(f"\n>>> Ablation smoke | {mode} | repeats={args.repeats} | "
          f"{len(specs)} cases × {len(conditions)} conditions\n")

    # results[cond][case_id] = {resolved, status, steps, s2e, tokens, flaky, record}
    results: dict[str, dict[str, dict]] = {c: {} for c in conditions}
    for cond, overrides in conditions.items():
        for spec in specs:
            rec, resolved, rcount, flaky = _run_cell(
                spec, overrides, mock=args.mock, real_backend=real_backend,
                log_dir=args.log_dir, repeats=args.repeats,
            )
            results[cond][spec.case_id] = {
                "resolved": resolved, "status": rec.status, "steps": rec.steps,
                "steps_to_first_edit": rec.steps_to_first_edit,
                "tokens": rec.total_tokens, "patch_lines": rec.patch_lines,
                "resolved_count": rcount, "flaky": flaky,
            }
            print(f"  [{cond:<17}] {spec.case_id:<26} -> {_cell_str(resolved, rec.status, flaky)}")

    _print_matrix(specs, conditions, results)
    _print_flips(specs, conditions, results)
    _print_secondary(conditions, results)
    out = _dump(args, specs, conditions, results)
    print(f"\n结果已写入 {out}")
    return 0


def _print_matrix(specs, conditions, results) -> None:
    cond_keys = list(conditions)
    headers = [_COND_SHORT[c] for c in cond_keys]
    print("\n" + "=" * 78)
    print("配对对照表（行=case，列=condition；✓=resolved，~=repeats 内翻动）")
    print("-" * 78)
    print(f"{'bucket':<10} {'case_id':<26} " + " ".join(f"{h:<10}" for h in headers))
    last_bucket = None
    for spec in specs:
        bucket = spec.bucket if spec.bucket != last_bucket else ""
        last_bucket = spec.bucket
        cells = []
        for c in cond_keys:
            r = results[c][spec.case_id]
            cells.append(f"{_cell_str(r['resolved'], r['status'], r['flaky']):<10}")
        print(f"{bucket:<10} {spec.case_id:<26} " + " ".join(cells))


def _print_flips(specs, conditions, results) -> None:
    print("\n" + "-" * 78)
    print("翻面 vs full（每个 condition 关掉后，哪些 case 的 resolved 变了）")
    full = results["full"]
    any_flip = False
    for cond in conditions:
        if cond == "full":
            continue
        flips = []
        for spec in specs:
            f = full[spec.case_id]["resolved"]
            c = results[cond][spec.case_id]["resolved"]
            if f != c:
                flips.append(f"{spec.case_id}({'✓→✗' if f and not c else '✗→✓'})")
        if flips:
            any_flip = True
            print(f"  {cond:<17}: " + ", ".join(flips))
        else:
            print(f"  {cond:<17}: （无翻面）")
    if not any_flip:
        print("  ⚠ 没有任何翻面：要么 case 太简单顶到天花板，要么组件在这些题上没被用到——"
              "看次级指标（steps/tokens）是否有差异。")


def _print_secondary(conditions, results) -> None:
    print("\n" + "-" * 78)
    print("次级指标（抗天花板：即使 resolved 不变，steps/tokens 也可能暴露组件作用）")
    print(f"{'condition':<17} {'resolve':<9} {'avg_steps':<10} {'avg_1st_edit':<13} {'avg_tokens':<11}")
    for cond in conditions:
        recs = [
            RunRecord(
                case_id=cid, resolved=r["resolved"], status=r["status"], steps=r["steps"],
                steps_to_first_edit=r["steps_to_first_edit"], total_tokens=r["tokens"],
                patch_lines=r["patch_lines"],
            )
            for cid, r in results[cond].items()
        ]
        agg = aggregate(recs)
        s2e = agg["avg_steps_to_first_edit"]
        print(f"{cond:<17} {agg['resolve_rate']:<9.1%} {agg['avg_steps']:<10} "
              f"{str(s2e):<13} {agg['avg_tokens']:<11}")


def _dump(args, specs, conditions, results) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ablation-{ts}.json"
    payload = {
        "mode": "mock" if args.mock else f"{args.provider}/{args.model}",
        "temperature": None if args.mock else args.temperature,
        "repeats": args.repeats,
        "conditions": list(conditions),
        "cases": [
            {"case_id": s.case_id, "bucket": s.bucket, "source": s.source,
             "source_id": s.source_id, "expected_without_component": s.expected_without_component}
            for s in specs
        ],
        "results": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    sys.exit(main())
