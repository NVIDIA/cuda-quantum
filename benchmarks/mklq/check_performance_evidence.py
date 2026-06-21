#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Check sanitized MKL-Q benchmark summaries for public evidence quality."""

import argparse
import json
import math
from pathlib import Path
from typing import Any


CHECK_SCHEMA_VERSION = "mklq-performance-evidence-check-v1"
SUMMARY_SCHEMA_VERSION = "mklq-benchmark-summary-v1"
DEFAULT_EVIDENCE_KIND = "clean_local_benchmark_evidence"
DEFAULT_REPORT_PATTERN = "local-clean-cpu-*.summary.json"
DEFAULT_RATIO_GROUP = "clean_worktree_cross_target_ratio"
DEFAULT_CANDIDATE_ELAPSED_GROUP = "mklq_cpu_elapsed_seconds_median"
DEFAULT_REFERENCE_TARGET = "qpp-cpu"
DEFAULT_CANDIDATE_TARGET = "mklq-cpu"
DEFAULT_REQUIRED_RATIOS = (
    "y_state_q20",
    "cy_state_q20",
    "cz_state_q20",
    "qft_like_state_q20",
    "seeded_clifford_state_q20",
    "sample_full_register_q20_1024_shots",
    "sample_full_register_q20_65536_shots",
    "sample_partial_register_q20_1024_shots",
    "sample_partial_register_q20_65536_shots",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float '{value}'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected positive float, got {parsed}")
    return parsed


def finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def int_at_least(value: Any, minimum: int) -> bool:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return False
    return parsed >= minimum


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def all_status_rows_ok(status_rows: Any) -> bool:
    if not isinstance(status_rows, dict) or not status_rows:
        return False
    return all(key == "ok" and isinstance(value, int) and value > 0
               for key, value in status_rows.items())


def ratio_key(reference_target: str, candidate_target: str, label: str) -> str:
    return (f"{reference_target.replace('-', '_')}_over_"
            f"{candidate_target.replace('-', '_')}_{label}")


def list_summaries(reports: Path, pattern: str,
                   summary_ids: set[str]) -> list[Path]:
    paths = sorted(reports.glob(pattern))
    if not summary_ids:
        return paths
    filtered: list[Path] = []
    for path in paths:
        try:
            summary = load_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            filtered.append(path)
            continue
        if str(summary.get("summary_id")) in summary_ids:
            filtered.append(path)
    return filtered


def check_summary(summary: dict[str, Any], *,
                  required_ratios: list[str],
                  min_speedup: float,
                  ratio_group: str,
                  candidate_elapsed_group: str,
                  reference_target: str,
                  candidate_target: str) -> dict[str, Any]:
    failures: list[str] = []

    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        failures.append("unexpected schema_version")
    if summary.get("evidence_kind") != DEFAULT_EVIDENCE_KIND:
        failures.append("unexpected evidence_kind")

    interpretation = summary.get("interpretation")
    if not isinstance(interpretation, dict):
        failures.append("missing interpretation object")
        interpretation = {}
    if interpretation.get("clean_worktree") is not True:
        failures.append("interpretation.clean_worktree is not true")
    if interpretation.get("raw_json_files_are_ignored") is not True:
        failures.append("raw JSON files are not marked ignored")
    performance_scope = str(interpretation.get("performance_claim_scope", ""))
    if "cross-machine" not in performance_scope or "not" not in performance_scope:
        failures.append("performance scope does not reject cross-machine claims")

    git = summary.get("git")
    if not isinstance(git, dict):
        failures.append("missing git provenance")
        git = {}
    if git.get("dirty") is not False:
        failures.append("git.dirty is not false")
    if not git.get("commit"):
        failures.append("missing git commit")

    config = summary.get("config")
    if not isinstance(config, dict):
        failures.append("missing config object")
        config = {}
    targets = config.get("targets")
    target_values = {str(target) for target in targets} if isinstance(
        targets, list) else set()
    if not {reference_target, candidate_target}.issubset(target_values):
        failures.append("config.targets does not include reference and candidate")
    if config.get("isolate_rows") is not True:
        failures.append("config.isolate_rows is not true")
    if not int_at_least(config.get("repeats"), 2):
        failures.append("config.repeats is below 2")
    if not int_at_least(config.get("warmups"), 1):
        failures.append("config.warmups is below 1")

    raw_results = summary.get("raw_results")
    if not isinstance(raw_results, list) or not raw_results:
        failures.append("missing raw_results")
        raw_results = []
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, dict):
            failures.append(f"raw_results[{index}] is not an object")
            continue
        path = str(raw.get("path", ""))
        if not path.startswith("benchmarks/mklq/results/"):
            failures.append(f"raw_results[{index}] path is outside ignored results")
        if raw.get("tracked") is not False:
            failures.append(f"raw_results[{index}] is not marked untracked")
        sha256 = raw.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            failures.append(f"raw_results[{index}] has invalid sha256")
        if not all_status_rows_ok(raw.get("status_rows")):
            failures.append(f"raw_results[{index}] has non-ok or missing rows")

    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        failures.append("missing rows")
    else:
        non_ok = [
            row for row in rows
            if isinstance(row, dict) and row.get("status") != "ok"
        ]
        if non_ok:
            failures.append("rows contain non-ok benchmark status")

    comparison = summary.get("comparison")
    if not isinstance(comparison, dict):
        failures.append("missing comparison object")
        comparison = {}
    ratios = comparison.get(ratio_group)
    if not isinstance(ratios, dict):
        failures.append(f"missing comparison.{ratio_group}")
        ratios = {}
    candidate_elapsed = comparison.get(candidate_elapsed_group)
    if not isinstance(candidate_elapsed, dict):
        failures.append(f"missing comparison.{candidate_elapsed_group}")
        candidate_elapsed = {}

    checked_ratios: dict[str, float] = {}
    for label in required_ratios:
        key = ratio_key(reference_target, candidate_target, label)
        value = ratios.get(key)
        if not finite_number(value):
            failures.append(f"missing finite ratio {key}")
            continue
        ratio = float(value)
        checked_ratios[key] = ratio
        if ratio < min_speedup:
            failures.append(
                f"{key}={ratio:.6g} is below min_speedup={min_speedup:g}")
        elapsed = candidate_elapsed.get(label)
        if not finite_number(elapsed) or float(elapsed) <= 0:
            failures.append(f"missing positive candidate elapsed {label}")

    status = "passed" if not failures else "failed"
    return {
        "status": status,
        "summary_id": summary.get("summary_id"),
        "failures": failures,
        "checked_ratio_count": len(checked_ratios),
        "min_checked_speedup": min(checked_ratios.values())
        if checked_ratios else None,
        "max_checked_speedup": max(checked_ratios.values())
        if checked_ratios else None,
    }


def build_report(root: Path, reports: Path, pattern: str,
                 summary_ids: set[str], required_ratios: list[str],
                 min_speedup: float, ratio_group: str,
                 candidate_elapsed_group: str, reference_target: str,
                 candidate_target: str) -> dict[str, Any]:
    paths = list_summaries(reports, pattern, summary_ids)
    checked: list[dict[str, Any]] = []
    for path in paths:
        try:
            summary = load_json(path)
            result = check_summary(
                summary,
                required_ratios=required_ratios,
                min_speedup=min_speedup,
                ratio_group=ratio_group,
                candidate_elapsed_group=candidate_elapsed_group,
                reference_target=reference_target,
                candidate_target=candidate_target,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            result = {
                "status": "failed",
                "summary_id": None,
                "failures": [f"{type(exc).__name__}: {exc}"],
                "checked_ratio_count": 0,
                "min_checked_speedup": None,
                "max_checked_speedup": None,
            }
        result["path"] = command_path(root, path)
        checked.append(result)

    passed_count = sum(1 for item in checked if item["status"] == "passed")
    failed_count = sum(1 for item in checked if item["status"] == "failed")
    status = "passed" if passed_count > 0 and failed_count == 0 else "failed"
    if not checked:
        status = "failed"

    return {
        "schema_version": CHECK_SCHEMA_VERSION,
        "config": {
            "reports": command_path(root, reports),
            "pattern": pattern,
            "summary_ids": sorted(summary_ids),
            "required_ratios": required_ratios,
            "min_speedup": min_speedup,
            "ratio_group": ratio_group,
            "candidate_elapsed_group": candidate_elapsed_group,
            "reference_target": reference_target,
            "candidate_target": candidate_target,
        },
        "summaries": checked,
        "summary": {
            "status": status,
            "passed": passed_count,
            "failed": failed_count,
            "checked": len(checked),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check sanitized MKL-Q benchmark summaries.")
    parser.add_argument("--reports",
                        type=Path,
                        default=Path("benchmarks/mklq/reports"),
                        help="Directory containing sanitized summary JSON.")
    parser.add_argument("--pattern",
                        default=DEFAULT_REPORT_PATTERN,
                        help="Glob pattern under --reports.")
    parser.add_argument("--summary-id",
                        action="append",
                        default=[],
                        help="Specific summary_id to check. May be repeated.")
    parser.add_argument("--required-ratios",
                        default=",".join(DEFAULT_REQUIRED_RATIOS),
                        help="Comma-separated required ratio labels.")
    parser.add_argument("--min-speedup",
                        type=positive_float,
                        default=10.0,
                        help="Minimum qpp-cpu over mklq-cpu ratio.")
    parser.add_argument("--ratio-group",
                        default=DEFAULT_RATIO_GROUP,
                        help="Comparison group containing cross-target ratios.")
    parser.add_argument("--candidate-elapsed-group",
                        default=DEFAULT_CANDIDATE_ELAPSED_GROUP,
                        help="Comparison group containing candidate medians.")
    parser.add_argument("--reference-target",
                        default=DEFAULT_REFERENCE_TARGET,
                        help="Reference target name.")
    parser.add_argument("--candidate-target",
                        default=DEFAULT_CANDIDATE_TARGET,
                        help="Candidate target name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    reports = args.reports if args.reports.is_absolute() else root / args.reports
    report = build_report(root=root,
                          reports=reports,
                          pattern=args.pattern,
                          summary_ids=set(args.summary_id),
                          required_ratios=parse_csv(args.required_ratios),
                          min_speedup=args.min_speedup,
                          ratio_group=args.ratio_group,
                          candidate_elapsed_group=args.candidate_elapsed_group,
                          reference_target=args.reference_target,
                          candidate_target=args.candidate_target)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
