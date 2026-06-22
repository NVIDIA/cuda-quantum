#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Render bounded MKL-Q Metal runtime counter evidence for public docs."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mklq-metal-runtime-counter-summary-v1"
COUNTER_REPORT_SCHEMA_VERSION = "mklq-metal-runtime-counter-probe-v1"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "reports"

CATEGORY_DESCRIPTIONS = {
    "resident_gate": "Resident Metal gate/update counter tests",
    "probability_sampling": "Resident probability fill and sampling counter tests",
    "measurement_reset": "Measurement, collapse, and reset counter tests",
    "fallback_boundary": "Unsupported-gate fallback and reupload boundary tests",
    "runtime_device": "Runtime/device boundary counter tests",
    "other": "Unclassified runtime counter tests",
}

CATEGORY_RULES = (
    ("fallback_boundary", ("Unsupported", "Fallback", "Reupload")),
    ("measurement_reset", ("Measure", "Measurement", "Collapse", "Reset")),
    ("probability_sampling",
     ("Probability", "Probabilities", "Sample", "Samples", "Sampling")),
    ("resident_gate", ("Resident", "Gate", "BuiltIn", "Controlled",
                       "MultiControl", "Phase", "Rx", "Ry", "Rz")),
    ("runtime_device", ("Runtime", "Device", "TargetRange", "Detects",
                        "Rejects", "Applies")),
)


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def load_counter_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path} is not a JSON object")
    schema_version = report.get("schema_version")
    if schema_version != COUNTER_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version={schema_version!r}, expected "
            f"{COUNTER_REPORT_SCHEMA_VERSION!r}")
    return report


def discover_counter_reports(reports: list[Path],
                             pattern: str = "*.counter.json") -> list[Path]:
    paths: list[Path] = []
    for report_path in reports:
        if report_path.is_dir():
            paths.extend(sorted(report_path.glob(pattern)))
            continue
        paths.append(report_path)

    unique: dict[str, Path] = {}
    for path in paths:
        unique[path.resolve().as_posix()] = path
    return [unique[key] for key in sorted(unique)]


def test_suffix(test_name: str) -> str:
    return test_name.rsplit(".", 1)[-1]


def categorize_test(test_name: str) -> str:
    suffix = test_suffix(test_name)
    for category, tokens in CATEGORY_RULES:
        if any(token in suffix for token in tokens):
            return category
    return "other"


def boundary_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {
            "runtime_counter_evidence": False,
            "release_signoff": False,
            "all_metal_execution_proof": False,
            "raw_logs_truncated": False,
        }

    boundaries = [
        report.get("boundary", {})
        for report in reports
        if isinstance(report.get("boundary"), dict)
    ]
    if len(boundaries) != len(reports):
        return {
            "runtime_counter_evidence": False,
            "release_signoff": False,
            "all_metal_execution_proof": False,
            "raw_logs_truncated": False,
        }
    return {
        "runtime_counter_evidence": all(
            bool(boundary.get("runtime_counter_evidence"))
            for boundary in boundaries),
        "release_signoff": any(
            bool(boundary.get("release_signoff")) for boundary in boundaries),
        "all_metal_execution_proof": any(
            bool(boundary.get("all_metal_execution_proof"))
            for boundary in boundaries),
        "raw_logs_truncated": all(
            bool(boundary.get("raw_logs_truncated", True))
            for boundary in boundaries),
    }


def build_report_digest(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    boundary = report.get("boundary", {})
    if not isinstance(boundary, dict):
        boundary = {}

    return {
        "path": path.as_posix(),
        "created_at_utc": report.get("created_at_utc", "-"),
        "status": summary.get("status", "unknown"),
        "expected": int(summary.get("expected", 0)),
        "selected": int(summary.get("selected", 0)),
        "missing": int(summary.get("missing", 0)),
        "passed": int(summary.get("passed", 0)),
        "failed": int(summary.get("failed", 0)),
        "runtime_counter_evidence": bool(
            boundary.get("runtime_counter_evidence")),
        "release_signoff": bool(boundary.get("release_signoff")),
        "all_metal_execution_proof": bool(
            boundary.get("all_metal_execution_proof")),
    }


def build_summary(paths: list[Path]) -> dict[str, Any]:
    loaded = [(path, load_counter_report(path)) for path in sorted(paths)]
    reports = [report for _, report in loaded]

    totals = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    category_tests: dict[str, list[str]] = defaultdict(list)

    for _, report in loaded:
        report_summary = report.get("summary", {})
        if isinstance(report_summary, dict):
            for key in ("expected", "selected", "missing", "passed",
                        "failed"):
                totals[key] += int(report_summary.get(key, 0))
        for test in report.get("tests", []):
            if not isinstance(test, dict):
                continue
            name = str(test.get("name", "<unknown>"))
            status = str(test.get("status", "unknown"))
            category = categorize_test(name)
            category_counts[category][status] += 1
            category_tests[category].append(name)

    categories = []
    for category in sorted(CATEGORY_DESCRIPTIONS):
        counts = category_counts.get(category, Counter())
        if not counts:
            continue
        tests = sorted(set(category_tests.get(category, [])))
        categories.append({
            "category": category,
            "description": CATEGORY_DESCRIPTIONS[category],
            "passed": counts.get("passed", 0),
            "failed": counts.get("failed", 0),
            "unknown": sum(
                value for key, value in counts.items()
                if key not in {"passed", "failed"}),
            "tests": tests,
        })

    failed_count = totals["failed"]
    missing_count = totals["missing"]
    boundary = boundary_summary(reports)
    boundary_ok = (
        bool(boundary["runtime_counter_evidence"])
        and not bool(boundary["release_signoff"])
        and not bool(boundary["all_metal_execution_proof"])
        and bool(boundary["raw_logs_truncated"]))
    status = (
        "passed" if loaded and failed_count == 0 and missing_count == 0
        and boundary_ok else "failed")

    return {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": COUNTER_REPORT_SCHEMA_VERSION,
        "summary": {
            "status": status,
            "report_count": len(loaded),
            "expected": totals["expected"],
            "selected": totals["selected"],
            "missing": missing_count,
            "passed": totals["passed"],
            "failed": failed_count,
        },
        "boundary": boundary,
        "reports": [
            build_report_digest(path, report) for path, report in loaded
        ],
        "categories": categories,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    totals = summary.get("summary", {})
    boundary = summary.get("boundary", {})
    lines = [
        "# MKL-Q Metal Runtime Counter Summary",
        "",
        "This file is generated from bounded `.counter.json` reports under "
        "`benchmarks/mklq/reports/`.",
        "",
        "Caveat: this is runtime counter evidence from selected build-tree "
        "ctest cases. It is not release sign-off, not a benchmark result, "
        "and not proof that every operation stayed on Metal.",
        "",
        "## Aggregate",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in ("status", "report_count", "expected", "selected", "missing",
                "passed", "failed"):
        lines.append(f"| `{key}` | {markdown_escape(totals.get(key, '-'))} |")

    lines.extend([
        "",
        "## Evidence Boundary",
        "",
        "| Boundary | Value |",
        "| --- | --- |",
    ])
    for key in ("runtime_counter_evidence", "release_signoff",
                "all_metal_execution_proof", "raw_logs_truncated"):
        lines.append(
            f"| `{key}` | {markdown_escape(boundary.get(key, '-'))} |")

    lines.extend([
        "",
        "## Counter Coverage Categories",
        "",
        "| Category | Passed | Failed | Other | Description |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for category in summary.get("categories", []):
        lines.append(
            "| {category} | {passed} | {failed} | {unknown} | {description} |".
            format(category=markdown_escape(category.get("category", "-")),
                   passed=markdown_escape(category.get("passed", 0)),
                   failed=markdown_escape(category.get("failed", 0)),
                   unknown=markdown_escape(category.get("unknown", 0)),
                   description=markdown_escape(
                       category.get("description", "-"))))

    lines.extend([
        "",
        "## Counter Tests",
        "",
        "| Category | Test |",
        "| --- | --- |",
    ])
    for category in summary.get("categories", []):
        category_name = str(category.get("category", "-"))
        tests = category.get("tests", [])
        if not tests:
            continue
        for test_name in tests:
            lines.append(
                f"| {markdown_escape(category_name)} | "
                f"`{markdown_escape(test_name)}` |")

    lines.extend([
        "",
        "## Reports",
        "",
        "| Report | Created | Status | Expected | Selected | Missing | Passed | "
        "Failed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for report in summary.get("reports", []):
        lines.append(
            "| {path} | {created} | {status} | {expected} | {selected} | "
            "{missing} | {passed} | {failed} |".format(
                path=markdown_escape(report.get("path", "-")),
                created=markdown_escape(report.get("created_at_utc", "-")),
                status=markdown_escape(report.get("status", "-")),
                expected=markdown_escape(report.get("expected", "-")),
                selected=markdown_escape(report.get("selected", "-")),
                missing=markdown_escape(report.get("missing", "-")),
                passed=markdown_escape(report.get("passed", "-")),
                failed=markdown_escape(report.get("failed", "-")),
            ))

    lines.extend([
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python3 benchmarks/mklq/summarize_metal_runtime_counters.py \\",
        "  --reports benchmarks/mklq/reports \\",
        "  --output docs/mklq/metal-runtime-counters.md",
        "```",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize bounded MKL-Q Metal runtime counter reports for "
            "public documentation."))
    parser.add_argument("--reports",
                        nargs="+",
                        type=Path,
                        default=[DEFAULT_REPORTS_DIR],
                        help=(
                            "Report file(s) or directory/directories to read. "
                            "Directories are searched for *.counter.json."))
    parser.add_argument("--pattern",
                        default="*.counter.json",
                        help="Glob used when a --reports path is a directory.")
    parser.add_argument("--format",
                        choices=("markdown", "json"),
                        default="markdown")
    parser.add_argument("--output",
                        type=Path,
                        help="Write output to this path instead of stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    paths = discover_counter_reports(args.reports, args.pattern)
    summary = build_summary(paths)
    if args.format == "json":
        rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    else:
        rendered = render_markdown(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if summary["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
