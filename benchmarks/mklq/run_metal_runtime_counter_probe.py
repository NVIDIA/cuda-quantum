#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Run build-tree Metal runtime counter tests and emit bounded evidence JSON."""

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mklq-metal-runtime-counter-probe-v1"
EVIDENCE_KIND = "local_runtime_counter_probe"

COUNTER_TEST_SUFFIXES = (
    "MetalRuntimeKeepsResidentStateAcrossGateSequence",
    "MetalRuntimeFillsResidentProbabilitiesWithoutStateReadback",
    "MetalRuntimeComputesAndCollapsesResidentQubitProbability",
    "SimulatorKeepsSupportedGateSequenceResidentUntilReadback",
    "SimulatorKeepsYAndControlledYResidentUntilReadback",
    "SimulatorKeepsBuiltInYAndControlledYResidentUntilReadback",
    "SimulatorSamplesResidentDenseStateWithoutReadback",
    "SimulatorSamplesLargeResidentPartialRegisterThroughFullProbability",
    "SimulatorSamplesSmallResidentPartialRegisterThroughMarginalProbability",
    "SimulatorMeasuresAndResetsResidentStateWithoutReadback",
    "SimulatorResetsResidentNonzeroTargetWithoutReadback",
    "SimulatorSamplesDenseFullRegisterThroughMetalProbabilityFill",
)

TEST_PREFIX = "mklq_metal_MKLQMetalTester."
TEST_LINE_RE = re.compile(r"Test #\d+:\s+(\S+)")
TAIL_CHARS = 1200


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_output(cwd: Path, command: list[str]) -> str:
    return subprocess.check_output(command,
                                   cwd=cwd,
                                   text=True,
                                   stderr=subprocess.STDOUT)


def run_command(cwd: Path, command: list[str]) -> dict[str, Any]:
    start = time.perf_counter()
    result = subprocess.run(command,
                            cwd=cwd,
                            capture_output=True,
                            text=True)
    return {
        "returncode": result.returncode,
        "elapsed_seconds": time.perf_counter() - start,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def output_tail(text: str | None, limit: int = TAIL_CHARS) -> str:
    return (text or "")[-limit:]


def command_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def select_counter_tests(ctest_listing: str) -> list[str]:
    selected: list[str] = []
    suffixes = set(COUNTER_TEST_SUFFIXES)
    for line in ctest_listing.splitlines():
        match = TEST_LINE_RE.search(line)
        if not match:
            continue
        test_name = match.group(1)
        if not test_name.startswith(TEST_PREFIX):
            continue
        suffix = test_name.removeprefix(TEST_PREFIX)
        if suffix in suffixes:
            selected.append(test_name)
    return selected


def build_report(repo_root: Path, build_dir: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    build_dir = build_dir.resolve()
    listing_command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "-N",
        "-R",
        "mklq_metal_MKLQMetalTester",
    ]
    listing = command_output(repo_root, listing_command)
    selected = select_counter_tests(listing)

    run_result: dict[str, Any] | None = None
    if selected:
        run_command_args = [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-R",
            "|".join(selected),
            "--output-on-failure",
        ]
        run_result = run_command(repo_root, run_command_args)
        aggregate_passed = run_result["returncode"] == 0
    else:
        run_command_args = []
        aggregate_passed = False

    tests: list[dict[str, Any]] = []
    for test_name in selected:
        item: dict[str, Any] = {
            "name": test_name,
            "status": "passed" if aggregate_passed else "failed",
            "counter_source": "MetalStateVectorExecutor runtime counters",
        }
        if run_result and not aggregate_passed:
            item["failure_excerpt"] = {
                "stdout_tail": output_tail(run_result.get("stdout")),
                "stderr_tail": output_tail(run_result.get("stderr")),
            }
        tests.append(item)

    passed_count = sum(1 for item in tests if item["status"] == "passed")
    failed_count = sum(1 for item in tests if item["status"] == "failed")
    status = "passed" if selected and failed_count == 0 else "failed"

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_kind": EVIDENCE_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "source": {
            "repo_root": repo_root.as_posix(),
            "build_dir": command_path(repo_root, build_dir),
            "listing_command": listing_command,
            "probe_command": run_command_args,
        },
        "summary": {
            "status": status,
            "selected": len(selected),
            "passed": passed_count,
            "failed": failed_count,
        },
        "boundary": {
            "runtime_counter_evidence": True,
            "runtime_counter_source": (
                "build-tree ctest cases that assert MetalStateVectorExecutor "
                "counters"),
            "release_signoff": False,
            "all_metal_execution_proof": False,
            "raw_logs_truncated": True,
        },
        "tests": tests,
    }
    if run_result is not None:
        execution: dict[str, Any] = {"returncode": run_result["returncode"]}
        if "elapsed_seconds" in run_result:
            execution["elapsed_seconds"] = run_result["elapsed_seconds"]
        report["execution"] = execution
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MKL-Q Metal runtime counter ctests and write bounded JSON "
            "evidence."))
    parser.add_argument("--repo-root",
                        type=Path,
                        default=repo_root(),
                        help="Repository root. Defaults to this checkout.")
    parser.add_argument("--build-dir",
                        type=Path,
                        default=Path("build-python"),
                        help="CMake build directory containing ctest metadata.")
    parser.add_argument("--output",
                        type=Path,
                        help="Optional JSON output path.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve()
    build_dir = args.build_dir
    if not build_dir.is_absolute():
        build_dir = root / build_dir
    report = build_report(repo_root=root, build_dir=build_dir)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if report["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
