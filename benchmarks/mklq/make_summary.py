#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Create sanitized MKL-Q benchmark summaries from ignored raw reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


RAW_SCHEMA_VERSION = "mklq-benchmark-v1"
SUMMARY_SCHEMA_VERSION = "mklq-benchmark-summary-v1"
DEFAULT_EVIDENCE_KIND = "clean_local_benchmark_evidence"
DEFAULT_PERFORMANCE_SCOPE = (
    "local benchmark evidence only; not cross-machine performance "
    "certification")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path} is not a JSON object")
    schema_version = report.get("schema_version")
    if schema_version != RAW_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version={schema_version!r}, expected "
            f"{RAW_SCHEMA_VERSION!r}")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path} does not contain a results list")
    return report


def status_counts(report: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in report.get("results", []):
        if isinstance(row, dict):
            counts[str(row.get("status", "unknown"))] += 1
    return dict(sorted(counts.items()))


def require_same(label: str, values: list[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise ValueError(f"raw reports have mismatched {label}: {values!r}")
    return first


def first_seen(values: list[Any]) -> list[Any]:
    seen: list[Any] = []
    for value in values:
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if candidate not in seen:
                seen.append(candidate)
    return seen


def sorted_numbers(values: list[Any]) -> list[Any]:
    merged = first_seen(values)
    if all(isinstance(value, (int, float)) for value in merged):
        return sorted(merged)
    return merged


def raw_git(report: dict[str, Any]) -> dict[str, Any]:
    provenance = report.get("provenance", {})
    if not isinstance(provenance, dict):
        return {}
    git = provenance.get("git", {})
    return git if isinstance(git, dict) else {}


def raw_environment(report: dict[str, Any]) -> dict[str, Any]:
    provenance = report.get("provenance", {})
    if not isinstance(provenance, dict):
        return {}
    environment = provenance.get("environment", {})
    return environment if isinstance(environment, dict) else {}


def raw_config(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("config", {})
    return config if isinstance(config, dict) else {}


def raw_runtime(report: dict[str, Any]) -> dict[str, Any]:
    runtime = report.get("runtime")
    if isinstance(runtime, dict):
        return runtime
    for row in report.get("results", []):
        if not isinstance(row, dict):
            continue
        isolated = row.get("isolated_process", {})
        if isinstance(isolated, dict) and isinstance(isolated.get("runtime"),
                                                    dict):
            return isolated["runtime"]
    return {}


def validate_reports(reports: list[dict[str, Any]],
                     allow_dirty: bool,
                     allow_errors: bool) -> None:
    if not reports:
        raise ValueError("expected at least one raw report")

    dirty_values = [bool(raw_git(report).get("dirty")) for report in reports]
    if any(dirty_values) and not allow_dirty:
        raise ValueError(
            "raw report records a dirty git worktree; pass --allow-dirty only "
            "for explicit tuning evidence")

    bad_statuses: list[dict[str, int]] = []
    for report in reports:
        counts = status_counts(report)
        non_ok = {key: value for key, value in counts.items() if key != "ok"}
        if non_ok:
            bad_statuses.append(non_ok)
    if bad_statuses and not allow_errors:
        raise ValueError(
            f"raw report contains non-ok benchmark rows: {bad_statuses!r}")

    require_same("git commit", [raw_git(report).get("commit") for report in reports])
    require_same("git dirty flag", dirty_values)


def bounded_row(row: dict[str, Any]) -> dict[str, Any]:
    public = {
        "target": row.get("target"),
        "case": row.get("case"),
        "qubits": row.get("qubits"),
        "shots": row.get("shots"),
        "status": row.get("status"),
        "estimated_state_bytes": row.get("estimated_state_bytes"),
        "repeats": row.get("repeats"),
        "warmups": row.get("warmups"),
    }
    if row.get("error"):
        public["error"] = row["error"]

    metrics = row.get("metrics", {})
    if isinstance(metrics, dict):
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                public[key] = value
    return {key: value for key, value in public.items() if value is not None}


def collect_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        for row in report.get("results", []):
            if isinstance(row, dict):
                rows.append(bounded_row(row))
    return rows


def metric_key(row: dict[str, Any], include_shots: bool) -> str:
    case = str(row["case"]).replace("-", "_")
    key = f"{case}_q{row['qubits']}"
    if include_shots:
        key += f"_{row['shots']}_shots"
    return key


def should_include_shots(row: dict[str, Any],
                         rows: list[dict[str, Any]]) -> bool:
    case = row.get("case")
    qubits = row.get("qubits")
    shots = {
        candidate.get("shots")
        for candidate in rows
        if candidate.get("case") == case and candidate.get("qubits") == qubits
    }
    return str(case).startswith("sample-") or len(shots) > 1


def elapsed_median(row: dict[str, Any]) -> float | None:
    value = row.get("elapsed_seconds_median")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def build_comparison(rows: list[dict[str, Any]],
                     reference_target: str,
                     candidate_target: str,
                     ratio_group: str | None) -> dict[str, Any]:
    by_key = {
        (row.get("target"), row.get("case"), row.get("qubits"), row.get("shots")):
            row
        for row in rows
    }
    ratios: dict[str, float] = {}
    candidate_elapsed: dict[str, float] = {}

    for row in rows:
        if row.get("target") != candidate_target:
            continue
        candidate_value = elapsed_median(row)
        if candidate_value is None:
            continue
        include_shots = should_include_shots(row, rows)
        label = metric_key(row, include_shots)
        candidate_elapsed[label] = candidate_value
        reference = by_key.get(
            (reference_target, row.get("case"), row.get("qubits"),
             row.get("shots")))
        if reference is None:
            continue
        reference_value = elapsed_median(reference)
        if reference_value is None or candidate_value == 0:
            continue
        ratio = reference_value / candidate_value
        if math.isfinite(ratio):
            ratios[
                f"{reference_target.replace('-', '_')}_over_"
                f"{candidate_target.replace('-', '_')}_{label}"] = ratio

    comparison: dict[str, Any] = {}
    if ratios:
        comparison[ratio_group or "cross_target_elapsed_ratio"] = dict(
            sorted(ratios.items()))
    if candidate_elapsed:
        comparison[
            f"{candidate_target.replace('-', '_')}_elapsed_seconds_median"] = (
                dict(sorted(candidate_elapsed.items())))
    return comparison


def build_config(reports: list[dict[str, Any]]) -> dict[str, Any]:
    configs = [raw_config(report) for report in reports]
    config: dict[str, Any] = {
        "targets": first_seen([item for cfg in configs for item in [cfg.get("targets")]]),
        "cases": first_seen([item for cfg in configs for item in [cfg.get("cases")]]),
        "qubits": sorted_numbers(
            [item for cfg in configs for item in [cfg.get("qubits")]]),
        "shot_counts": sorted_numbers([
            item for cfg in configs
            for item in [cfg.get("shot_counts", [cfg.get("shots")])]
        ]),
        "repeats": require_same("repeats",
                                 [cfg.get("repeats") for cfg in configs]),
        "warmups": require_same("warmups",
                                 [cfg.get("warmups") for cfg in configs]),
        "layers": require_same("layers", [cfg.get("layers") for cfg in configs]),
        "isolate_rows": require_same(
            "isolate_rows", [cfg.get("isolate_rows") for cfg in configs]),
        "commands": [cfg.get("command") for cfg in configs if cfg.get("command")],
    }
    if config["shot_counts"]:
        config["shots"] = config["shot_counts"][0]
    return config


def build_summary(raw_paths: list[Path],
                  summary_id: str,
                  evidence_kind: str,
                  reference_target: str,
                  candidate_target: str,
                  ratio_group: str | None,
                  performance_scope: str,
                  summary_text: str,
                  runtime_note: str | None = None,
                  allow_dirty: bool = False,
                  allow_errors: bool = False) -> dict[str, Any]:
    reports = [load_raw_report(path) for path in raw_paths]
    validate_reports(reports, allow_dirty=allow_dirty, allow_errors=allow_errors)
    rows = collect_rows(reports)
    first = reports[0]
    runtime = raw_runtime(first)
    git = raw_git(first)

    interpretation = {
        "clean_worktree": not bool(git.get("dirty")),
        "raw_json_files_are_ignored": True,
        "performance_claim_scope": performance_scope,
        "summary": summary_text,
    }
    if runtime_note:
        interpretation["runtime_build_note"] = runtime_note

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "evidence_kind": evidence_kind,
        "summary_id": summary_id,
        "raw_results": [{
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "status_rows": status_counts(report),
            "tracked": False,
        } for path, report in zip(raw_paths, reports)],
        "machine": first.get("machine", {}),
        "environment": raw_environment(first),
        "git": git,
        "runtime": {
            "cudaq_module_file": runtime.get("cudaq_module_file"),
            "cudaq_version": runtime.get("cudaq_version"),
            "module_from_build_tree": runtime.get("module_from_build_tree"),
            "python_prefix": runtime.get("python_prefix"),
        },
        "config": build_config(reports),
        "rows": rows,
        "comparison": build_comparison(rows, reference_target,
                                       candidate_target, ratio_group),
        "interpretation": interpretation,
    }


def write_json(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sanitized MKL-Q benchmark summary JSON.")
    parser.add_argument("--raw",
                        action="append",
                        type=Path,
                        required=True,
                        help="Raw benchmark JSON path. May be repeated.")
    parser.add_argument("--summary-id", required=True, help="Summary id.")
    parser.add_argument("--evidence-kind",
                        default=DEFAULT_EVIDENCE_KIND,
                        help="Summary evidence_kind value.")
    parser.add_argument("--reference-target",
                        default="qpp-cpu",
                        help="Reference target for elapsed ratios.")
    parser.add_argument("--candidate-target",
                        default="mklq-cpu",
                        help="Candidate target for elapsed ratios.")
    parser.add_argument("--ratio-group",
                        default=None,
                        help="Optional comparison group name for ratios.")
    parser.add_argument("--performance-scope",
                        default=DEFAULT_PERFORMANCE_SCOPE,
                        help="Interpretation performance_claim_scope text.")
    parser.add_argument("--summary-text",
                        required=True,
                        help="Short human summary for interpretation.summary.")
    parser.add_argument("--runtime-note",
                        default=None,
                        help="Optional interpretation.runtime_build_note text.")
    parser.add_argument("--output",
                        type=Path,
                        help="Output summary JSON path. Defaults to stdout.")
    parser.add_argument("--allow-dirty",
                        action="store_true",
                        help="Allow raw reports whose provenance is dirty.")
    parser.add_argument("--allow-errors",
                        action="store_true",
                        help="Allow non-ok raw benchmark rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(raw_paths=args.raw,
                            summary_id=args.summary_id,
                            evidence_kind=args.evidence_kind,
                            reference_target=args.reference_target,
                            candidate_target=args.candidate_target,
                            ratio_group=args.ratio_group,
                            performance_scope=args.performance_scope,
                            summary_text=args.summary_text,
                            runtime_note=args.runtime_note,
                            allow_dirty=args.allow_dirty,
                            allow_errors=args.allow_errors)
    write_json(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
