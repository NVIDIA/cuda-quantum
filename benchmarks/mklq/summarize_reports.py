#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA_VERSION = "mklq-benchmark-summary-v1"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def markdown_escape(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def format_sequence(values: object) -> str:
    if values is None:
        return "-"
    if isinstance(values, list):
        return ", ".join(str(value) for value in values) if values else "-"
    return str(values)


def format_machine(machine: dict[str, Any]) -> str:
    cpu = machine.get("cpu_brand") or machine.get("processor") or "unknown CPU"
    cores = machine.get("logical_cores")
    memory = machine.get("memory_bytes")
    macos = machine.get("macos_version")

    parts = [str(cpu)]
    if cores:
        parts.append(f"{cores} logical cores")
    if isinstance(memory, int):
        parts.append(f"{memory / (1024 ** 3):.0f} GiB RAM")
    if macos:
        parts.append(f"macOS {macos}")
    return ", ".join(parts)


def format_run_shape(config: dict[str, Any]) -> str:
    parts: list[str] = []
    if "shot_counts" in config:
        parts.append(f"shot_counts={format_sequence(config['shot_counts'])}")
    elif "shots" in config:
        parts.append(f"shots={config['shots']}")
    for key in ("repeats", "warmups", "layers"):
        if key in config:
            parts.append(f"{key}={config[key]}")
    if config.get("isolate_rows"):
        parts.append("isolate_rows=true")
    return "; ".join(parts) if parts else "-"


def row_status_counts(rows: object) -> dict[str, int]:
    if not isinstance(rows, list):
        return {}
    counts: Counter[str] = Counter()
    for row in rows:
        if isinstance(row, dict):
            counts[str(row.get("status", "unknown"))] += 1
    return dict(sorted(counts.items()))


def format_status_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items()) or "-"


def format_raw_results(raw_results: object) -> str:
    if not isinstance(raw_results, list) or not raw_results:
        return "-"

    labels: list[str] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        path = raw_result.get("path", "<unknown>")
        sha = str(raw_result.get("sha256", ""))
        suffix = f" sha256={sha[:12]}" if sha else ""
        labels.append(f"{path}{suffix}")
    return "; ".join(labels) if labels else "-"


def load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"{path} is not a JSON object")
    schema_version = summary.get("schema_version")
    if schema_version != SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version={schema_version!r}, expected "
            f"{SUMMARY_SCHEMA_VERSION!r}")
    return summary


def digest_summary(path: Path) -> dict[str, Any]:
    summary = load_summary(path)
    config = summary.get("config", {})
    if not isinstance(config, dict):
        config = {}
    rows = summary.get("rows", [])

    return {
        "summary_id": summary.get("summary_id", path.stem),
        "path": path.as_posix(),
        "evidence_kind": summary.get("evidence_kind", "-"),
        "machine": format_machine(summary.get("machine", {})),
        "targets": format_sequence(config.get("targets")),
        "cases": format_sequence(config.get("cases")),
        "qubits": format_sequence(config.get("qubits")),
        "run_shape": format_run_shape(config),
        "row_status_counts": row_status_counts(rows),
        "raw_results": format_raw_results(summary.get("raw_results")),
        "comparison": summary.get("comparison", {}),
        "interpretation": summary.get("interpretation", {}),
    }


def load_digests(paths: list[Path]) -> list[dict[str, Any]]:
    digests = [digest_summary(path) for path in paths]
    return sorted(digests, key=lambda item: str(item["summary_id"]))


def iter_numeric_comparison_metrics(
        value: object,
        prefix: str = "") -> list[tuple[str, float]]:
    if isinstance(value, dict):
        metrics: list[tuple[str, float]] = []
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.extend(
                iter_numeric_comparison_metrics(value[key], child_prefix))
        return metrics
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [(prefix, float(value))]
    return []


def format_metric_value(metric_name: str, value: float) -> str:
    if metric_name.endswith("seconds") or "_seconds" in metric_name:
        return f"{value:.6g} s"
    if "ratio" in metric_name:
        return f"{value:.2f}x"
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def comparison_signals(
        digests: list[dict[str, Any]]) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    for digest in digests:
        for metric_name, value in iter_numeric_comparison_metrics(
                digest.get("comparison", {})):
            signals.append({
                "summary_id": str(digest["summary_id"]),
                "metric": metric_name,
                "value": format_metric_value(metric_name, value),
            })
    return signals


def render_markdown(digests: list[dict[str, Any]]) -> str:
    lines = [
        "# MKL-Q Benchmark Evidence",
        "",
        "This file is generated from sanitized benchmark summaries under "
        "`benchmarks/mklq/reports/`.",
        "",
        "Caveat: these entries are local benchmark evidence from development "
        "or release-prep runs. Interpret each entry through its "
        "`evidence_kind` and `interpretation` fields; none is a cross-machine "
        "performance certification.",
        "",
        "## Evidence Inventory",
        "",
        "| Summary ID | Kind | Machine | Targets | Cases | Qubits | Run shape | "
        "Rows | Raw evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for digest in digests:
        lines.append(
            "| {summary_id} | {evidence_kind} | {machine} | {targets} | "
            "{cases} | {qubits} | {run_shape} | {rows} | {raw} |".format(
                summary_id=markdown_escape(digest["summary_id"]),
                evidence_kind=markdown_escape(digest["evidence_kind"]),
                machine=markdown_escape(digest["machine"]),
                targets=markdown_escape(digest["targets"]),
                cases=markdown_escape(digest["cases"]),
                qubits=markdown_escape(digest["qubits"]),
                run_shape=markdown_escape(digest["run_shape"]),
                rows=markdown_escape(
                    format_status_counts(digest["row_status_counts"])),
                raw=markdown_escape(digest["raw_results"]),
            ))

    lines.extend([
        "",
        "## Comparison Signals",
        "",
        "The values below are copied from each summary's bounded `comparison` "
        "object. Keep their original local-run context when citing them.",
        "",
    ])

    signals = comparison_signals(digests)
    if signals:
        lines.extend([
            "| Summary ID | Metric | Value |",
            "| --- | --- | --- |",
        ])
        for signal in signals:
            lines.append(
                "| {summary_id} | `{metric}` | {value} |".format(
                    summary_id=markdown_escape(signal["summary_id"]),
                    metric=markdown_escape(signal["metric"]),
                    value=markdown_escape(signal["value"]),
                ))
    else:
        lines.append("No numeric comparison signals are recorded.")

    lines.extend([
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python3 benchmarks/mklq/summarize_reports.py \\",
        "  --reports benchmarks/mklq/reports \\",
        "  --format markdown \\",
        "  --output docs/mklq/benchmark-evidence.md",
        "```",
    ])
    return "\n".join(lines)


def build_payload(digests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "mklq-benchmark-evidence-index-v1",
        "source_schema_version": SUMMARY_SCHEMA_VERSION,
        "reports": digests,
        "comparison_signals": comparison_signals(digests),
        "caveat": (
            "These entries are local benchmark evidence from development or "
            "release-prep runs, not cross-machine performance "
            "certification."),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize sanitized MKL-Q benchmark report JSON files.")
    parser.add_argument("--reports",
                        type=Path,
                        default=DEFAULT_REPORTS_DIR,
                        help="Directory containing *.summary.json reports.")
    parser.add_argument("--format",
                        choices=("markdown", "json"),
                        default="markdown",
                        help="Output format.")
    parser.add_argument("--output",
                        type=Path,
                        help="Optional output file. Defaults to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.reports.glob("*.summary.json"))
    if not paths:
        raise SystemExit(f"no *.summary.json files found under {args.reports}")

    digests = load_digests(paths)
    if args.format == "json":
        rendered = json.dumps(build_payload(digests), indent=2, sort_keys=True)
    else:
        rendered = render_markdown(digests)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
