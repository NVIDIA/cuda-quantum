#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Run the MKL-Q local correctness gate and write a JSON summary."""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mklq-correctness-gate-v1"
TARGET_CONFIG_REGEX = (
    r"(mklq_(cpu|metal)_MKLQ|backend_target_setter_check|TargetConfigTester)")
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_TAIL_CHARS = 8000


@dataclass(frozen=True)
class CorrectnessGateConfig:
    repo_root: Path
    pythonpath: str
    nvqpp: Path
    build_dir: Path
    output: Path
    stamp: str
    python_executable: str
    timeout_seconds: int
    tail_chars: int
    skip_python: bool
    skip_nvqpp: bool
    skip_ctest: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer '{value}'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {parsed}")
    return parsed


def command_path(config: CorrectnessGateConfig, path: Path) -> str:
    try:
        return path.relative_to(config.repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def fixed_env(config: CorrectnessGateConfig) -> dict[str, str]:
    return {
        "PYTHONPATH": config.pythonpath,
        "CUDAQ_NVQPP": str(config.nvqpp),
    }


def output_default(stamp: str) -> Path:
    return Path("benchmarks/mklq/results") / (
        f"local-correctness-gate-{stamp}.json")


def step_plan(config: CorrectnessGateConfig) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []

    if not config.skip_python:
        steps.append({
            "name":
                "python_target_smoke",
            "command": [
                config.python_executable,
                "-m",
                "pytest",
                "python/tests/backends/test_mklq_python_api.py",
                "python/tests/backends/test_mklq_cpu_correctness_fixtures.py",
                "python/tests/backends/test_mklq_metal_correctness_fixtures.py",
                "python/tests/builder/test_mklq_targets.py",
                "-q",
            ],
            "env":
                fixed_env(config),
        })

    if not config.skip_nvqpp:
        steps.append({
            "name":
                "nvqpp_smoke",
            "command": [
                config.python_executable,
                "-m",
                "pytest",
                "python/tests/backends/test_mklq_nvqpp_smoke.py",
                "-q",
            ],
            "env":
                fixed_env(config),
        })

    if not config.skip_ctest:
        steps.append({
            "name":
                "target_config_ctest",
            "command": [
                "ctest",
                "--test-dir",
                command_path(config, config.build_dir),
                "-R",
                TARGET_CONFIG_REGEX,
                "--output-on-failure",
            ],
            "env":
                fixed_env(config),
        })

    return steps


def skipped_steps(config: CorrectnessGateConfig) -> list[str]:
    skipped: list[str] = []
    if config.skip_python:
        skipped.append("python_target_smoke")
    if config.skip_nvqpp:
        skipped.append("nvqpp_smoke")
    if config.skip_ctest:
        skipped.append("target_config_ctest")
    return skipped


def git_output(root: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args],
                                       cwd=root,
                                       stderr=subprocess.DEVNULL,
                                       text=True).rstrip("\n")
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_snapshot(root: Path) -> dict[str, Any]:
    status = git_output(root, ["status", "--short"]).splitlines()
    return {
        "root": git_output(root, ["rev-parse", "--show-toplevel"]),
        "branch": git_output(root, ["branch", "--show-current"]),
        "commit": git_output(root, ["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def build_plan(config: CorrectnessGateConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "stamp": config.stamp,
            "repo_root": config.repo_root.as_posix(),
            "pythonpath": config.pythonpath,
            "nvqpp": str(config.nvqpp),
            "build_dir": config.build_dir.as_posix(),
            "output": config.output.as_posix(),
            "python_executable": config.python_executable,
            "timeout_seconds": config.timeout_seconds,
            "tail_chars": config.tail_chars,
        },
        "environment": fixed_env(config),
        "skipped_steps": skipped_steps(config),
        "steps": step_plan(config),
    }


def output_tail(value: str | bytes | None, tail_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-tail_chars:]


def merged_env(step_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(step_env)
    return env


def run_step(step: dict[str, Any], config: CorrectnessGateConfig) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = subprocess.run(step["command"],
                                cwd=config.repo_root,
                                env=merged_env(step["env"]),
                                capture_output=True,
                                text=True,
                                timeout=config.timeout_seconds)
        elapsed = time.perf_counter() - start
        return {
            "name": step["name"],
            "command": step["command"],
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "stdout_tail": output_tail(result.stdout, config.tail_chars),
            "stderr_tail": output_tail(result.stderr, config.tail_chars),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return {
            "name": step["name"],
            "command": step["command"],
            "status": "failed",
            "returncode": None,
            "elapsed_seconds": elapsed,
            "timeout_seconds": config.timeout_seconds,
            "stdout_tail": output_tail(exc.stdout, config.tail_chars),
            "stderr_tail": output_tail(exc.stderr, config.tail_chars),
        }


def summarize(step_results: list[dict[str, Any]],
              skipped: list[str]) -> dict[str, Any]:
    passed = sum(1 for result in step_results if result["status"] == "passed")
    failed = sum(1 for result in step_results if result["status"] == "failed")
    return {
        "status": "passed" if failed == 0 else "failed",
        "passed": passed,
        "failed": failed,
        "skipped": len(skipped),
    }


def run_gate(config: CorrectnessGateConfig,
             plan_only: bool = False) -> dict[str, Any]:
    plan = build_plan(config)
    if plan_only:
        return plan

    step_results = [run_step(step, config) for step in plan["steps"]]
    report = dict(plan)
    report["git"] = git_snapshot(config.repo_root)
    report["machine"] = {
        "platform": sys.platform,
        "python_version": sys.version,
    }
    report["steps"] = step_results
    report["summary"] = summarize(step_results, plan["skipped_steps"])
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(report, indent=2, sort_keys=True) +
                             "\n",
                             encoding="utf-8")
    return report


def make_config(args: argparse.Namespace) -> CorrectnessGateConfig:
    root = repo_root()
    stamp = args.stamp or date.today().isoformat()
    install_prefix = args.install_prefix.expanduser()
    pythonpath = args.pythonpath or str(install_prefix)
    nvqpp = args.nvqpp or install_prefix / "bin" / "nvq++"
    output = args.output or output_default(stamp)
    return CorrectnessGateConfig(
        repo_root=root,
        pythonpath=pythonpath,
        nvqpp=resolve_path(root, nvqpp.expanduser()),
        build_dir=resolve_path(root, args.build_dir.expanduser()),
        output=resolve_path(root, output.expanduser()),
        stamp=stamp,
        python_executable=args.python_executable,
        timeout_seconds=args.timeout_seconds,
        tail_chars=args.tail_chars,
        skip_python=args.skip_python,
        skip_nvqpp=args.skip_nvqpp,
        skip_ctest=args.skip_ctest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MKL-Q local correctness gate.")
    parser.add_argument(
        "--install-prefix",
        type=Path,
        default=Path.home() / ".cudaq-mklq",
        help="Installed CUDA-Q/MKL-Q prefix used for PYTHONPATH and nvq++.")
    parser.add_argument("--pythonpath",
                        help="Override PYTHONPATH for the Python smoke tests.")
    parser.add_argument("--nvqpp",
                        type=Path,
                        help="Override the nvq++ path for the smoke test.")
    parser.add_argument("--build-dir",
                        type=Path,
                        default=Path("build-python"),
                        help="CMake build directory for TargetConfig ctest.")
    parser.add_argument("--output",
                        type=Path,
                        help="JSON output path. Defaults under ignored results/.")
    parser.add_argument("--stamp",
                        help="Date or label for the default output filename.")
    parser.add_argument("--python-executable",
                        default=sys.executable,
                        help="Python executable used for pytest steps.")
    parser.add_argument("--timeout-seconds",
                        type=positive_int,
                        default=DEFAULT_TIMEOUT_SECONDS,
                        help="Per-step subprocess timeout.")
    parser.add_argument("--tail-chars",
                        type=positive_int,
                        default=DEFAULT_TAIL_CHARS,
                        help="Characters of stdout/stderr retained per step.")
    parser.add_argument("--skip-python",
                        action="store_true",
                        help="Skip Python target smoke tests.")
    parser.add_argument("--skip-nvqpp",
                        action="store_true",
                        help="Skip nvq++ smoke tests.")
    parser.add_argument("--skip-ctest",
                        action="store_true",
                        help="Skip TargetConfig ctest.")
    parser.add_argument("--plan-only",
                        action="store_true",
                        help="Print the planned gate as JSON without running it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = make_config(args)
    report = run_gate(config, plan_only=args.plan_only)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if args.plan_only or report["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
