#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Check experimental MKL-Q Metal benchmark evidence boundaries."""

import argparse
import json
from pathlib import Path
from typing import Any


CHECK_SCHEMA_VERSION = "mklq-metal-evidence-check-v1"
SUMMARY_SCHEMA_VERSION = "mklq-benchmark-summary-v1"
DEFAULT_EVIDENCE_KIND = "local_tuning_evidence"
DEFAULT_REPORT_PATTERN = "*.summary.json"
METAL_TARGET = "mklq-metal"
FORBIDDEN_METAL_CLAIMS = (
    "default-ready",
    "release-ready",
    "production-ready",
    "full metal-native",
    "full-metal-native",
    "fully metal-native",
    "fully-metal-native",
    "clean release evidence",
    "release certification",
)
REQUIRED_PATH_SCOPE_HINTS = (
    "mixed-path",
    "resident",
    "host",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


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


def rows_for_target(rows: Any, target: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [
        row for row in rows
        if isinstance(row, dict) and row.get("target") == target
    ]


def target_values(summary: dict[str, Any]) -> set[str]:
    config = summary.get("config")
    if not isinstance(config, dict):
        return set()
    targets = config.get("targets")
    if not isinstance(targets, list):
        return set()
    return {str(target) for target in targets}


def is_metal_summary(summary: dict[str, Any]) -> bool:
    summary_id = str(summary.get("summary_id", ""))
    return (METAL_TARGET in target_values(summary)
            or bool(rows_for_target(summary.get("rows"), METAL_TARGET))
            or "metal" in summary_id.lower())


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


def forbidden_claim_failures(values: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for field, value in values.items():
        normalized = value.lower()
        for phrase in FORBIDDEN_METAL_CLAIMS:
            if phrase in normalized:
                failures.append(
                    f"{field} contains forbidden Metal claim {phrase!r}")
    return failures


def check_summary(summary: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []

    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        failures.append("unexpected schema_version")
    if summary.get("evidence_kind") != DEFAULT_EVIDENCE_KIND:
        failures.append("unexpected evidence_kind")

    config_targets = target_values(summary)
    if METAL_TARGET not in config_targets:
        failures.append(f"config.targets does not include {METAL_TARGET}")

    interpretation = summary.get("interpretation")
    if not isinstance(interpretation, dict):
        failures.append("missing interpretation object")
        interpretation = {}
    if interpretation.get("do_not_treat_as_clean_release_provenance") is not True:
        failures.append(
            "interpretation.do_not_treat_as_clean_release_provenance is not true"
        )
    if interpretation.get("raw_json_files_are_ignored") is not True:
        failures.append("raw JSON files are not marked ignored")

    performance_scope = str(interpretation.get("performance_claim_scope", ""))
    if not performance_scope:
        failures.append("missing interpretation.performance_claim_scope")
    elif "local" not in performance_scope.lower():
        failures.append("performance scope does not state local-only evidence")

    metal_scope = str(interpretation.get("metal_path_scope", ""))
    if not metal_scope:
        failures.append("missing interpretation.metal_path_scope")
    elif not any(hint in metal_scope.lower()
                 for hint in REQUIRED_PATH_SCOPE_HINTS):
        failures.append("metal path scope does not state mixed, resident, or host path")

    summary_text = str(interpretation.get("summary", ""))
    failures.extend(
        forbidden_claim_failures({
            "interpretation.summary": summary_text,
            "interpretation.performance_claim_scope": performance_scope,
            "interpretation.metal_path_scope": metal_scope,
        }))

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
    metal_rows = rows_for_target(rows, METAL_TARGET)
    if not metal_rows:
        failures.append(f"missing {METAL_TARGET} rows")
    non_ok = [row for row in metal_rows if row.get("status") != "ok"]
    if non_ok:
        failures.append(f"{METAL_TARGET} rows contain non-ok benchmark status")

    status = "passed" if not failures else "failed"
    return {
        "status": status,
        "summary_id": summary.get("summary_id"),
        "failures": failures,
        "metal_row_count": len(metal_rows),
        "raw_result_count": len(raw_results),
    }


def build_report(root: Path, reports: Path, pattern: str,
                 summary_ids: set[str]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    for path in list_summaries(reports, pattern, summary_ids):
        try:
            summary = load_json(path)
            if not is_metal_summary(summary):
                continue
            result = check_summary(summary)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            result = {
                "status": "failed",
                "summary_id": None,
                "failures": [f"{type(exc).__name__}: {exc}"],
                "metal_row_count": 0,
                "raw_result_count": 0,
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
            "target": METAL_TARGET,
            "evidence_kind": DEFAULT_EVIDENCE_KIND,
            "forbidden_metal_claims": list(FORBIDDEN_METAL_CLAIMS),
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
        description="Check experimental MKL-Q Metal benchmark evidence.")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    reports = args.reports if args.reports.is_absolute() else root / args.reports
    report = build_report(root=root,
                          reports=reports,
                          pattern=args.pattern,
                          summary_ids=set(args.summary_id))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
