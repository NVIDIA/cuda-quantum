#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Run MKL-Q public repository health checks."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "mklq-public-healthcheck-v1"
DEFAULT_TAIL_CHARS = 8000
TRACKED_ARTIFACT_PATTERN = re.compile(
    r"(^|/)(__pycache__|\.pytest_cache)(/|$)|"
    r"\.pyc$|\.DS_Store$|^build(-python)?/|"
    r"^benchmarks/mklq/results/|^docs/superpowers/|"
    r"^(dist|wheelhouse)/|\.(whl|dmg|pkg|zip)$|\.tar\.gz$")

BENCHMARK_HELPERS = (
    "benchmarks/mklq/bench_mklq_targets.py",
    "benchmarks/mklq/bench_probability_kernels.py",
    "benchmarks/mklq/make_summary.py",
    "benchmarks/mklq/run_clean_cpu_benchmark.py",
    "benchmarks/mklq/run_correctness_gate.py",
    "benchmarks/mklq/run_public_healthcheck.py",
    "benchmarks/mklq/summarize_reports.py",
)

PUBLIC_MARKDOWN_FILES = (
    "README.md",
    "benchmarks/mklq/README.md",
)


@dataclass(frozen=True)
class HealthcheckConfig:
    repo_root: Path
    install_prefix: Path
    build_dir: Path
    python_executable: str
    pythonpath: str
    nvqpp: Path
    stamp: str
    output: Path
    jobs: int
    timeout_seconds: int
    tail_chars: int
    require_clean: bool
    full: bool
    include_harness_tests: bool
    refresh_clean_cpu_benchmark: bool
    plan_only: bool


@dataclass(frozen=True)
class Step:
    name: str
    description: str
    runner: Callable[[HealthcheckConfig], dict[str, Any]]
    command: list[str] | None = None


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


def command_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def output_default(stamp: str) -> Path:
    return Path("benchmarks/mklq/results") / f"public-healthcheck-{stamp}.json"


def output_tail(value: str | bytes | None, tail_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-tail_chars:]


def command_output(root: Path, args: list[str]) -> str:
    return subprocess.check_output(args, cwd=root, text=True).rstrip("\n")


def run_command(config: HealthcheckConfig,
                command: list[str],
                env_overlay: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    if env_overlay:
        env.update(env_overlay)
    start = time.perf_counter()
    result = subprocess.run(command,
                            cwd=config.repo_root,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=config.timeout_seconds)
    elapsed = time.perf_counter() - start
    return {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": output_tail(result.stdout, config.tail_chars),
        "stderr_tail": output_tail(result.stderr, config.tail_chars),
    }


def passed(details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": "passed", "details": details or {}}


def failed(message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"status": "failed", "message": message}
    if details:
        payload["details"] = details
    return payload


def run_git_repository_check(config: HealthcheckConfig) -> dict[str, Any]:
    status = command_output(config.repo_root, ["git", "status", "--short", "--branch"])
    remotes = command_output(config.repo_root, ["git", "remote", "-v"]).splitlines()
    shallow = command_output(config.repo_root,
                             ["git", "rev-parse", "--is-shallow-repository"])
    workflows = command_output(config.repo_root,
                               ["git", "ls-files", ".github/workflows"]).splitlines()

    failures: list[str] = []
    if shallow.strip() != "false":
        failures.append("repository is shallow")
    if workflows != [".github/workflows/mklq-public-hygiene.yml"]:
        failures.append("unexpected tracked GitHub workflow set")
    if config.require_clean:
        dirty_lines = [line for line in status.splitlines() if not line.startswith("##")]
        if dirty_lines:
            failures.append("working tree is dirty")
    if not any(line.startswith("origin\thttps://github.com/wuls968/MKL-Q.git")
               for line in remotes):
        failures.append("origin does not point to https://github.com/wuls968/MKL-Q.git")
    if not any(line.startswith("upstream\thttps://github.com/NVIDIA/cuda-quantum.git")
               for line in remotes):
        failures.append("upstream does not point to NVIDIA/cuda-quantum")

    details = {
        "status_short_branch": status.splitlines(),
        "remotes": remotes,
        "is_shallow": shallow.strip(),
        "workflows": workflows,
    }
    return failed("; ".join(failures), details) if failures else passed(details)


def run_tracked_artifact_check(config: HealthcheckConfig) -> dict[str, Any]:
    tracked = command_output(config.repo_root, ["git", "ls-files"]).splitlines()
    bad = [path for path in tracked if TRACKED_ARTIFACT_PATTERN.search(path)]
    details = {"tracked_file_count": len(tracked), "bad_paths": bad}
    return failed("generated, local, release, or agent-internal files are tracked",
                  details) if bad else passed(details)


def public_metadata_requirements() -> list[tuple[str, str]]:
    return [
        ("README.md", "MKL-Q"),
        ("README.md", "mklq-cpu"),
        ("README.md", "mklq-metal"),
        ("README.md", "source-only"),
        ("README.md", "cudaq"),
        ("docs/mklq/known-limitations.md", "mklq-cpu"),
        ("docs/mklq/known-limitations.md", "mklq-metal"),
        ("docs/mklq/architecture.md", "Public Compatibility Boundary"),
        ("docs/mklq/testing-matrix.md", "Gate Summary"),
        ("docs/mklq/testing-matrix.md", "Capability Coverage"),
        ("docs/mklq/upstream-sync.md", "Post-merge Gates"),
        ("docs/mklq/release-policy.md", "source-only"),
        ("docs/mklq/public-release-checklist.md", "GitHub Verification"),
        ("docs/mklq/developer-workflow.md", "Public Hygiene"),
        ("docs/mklq/maintainer-runbook.md", "Routine Health Check"),
        ("docs/mklq/issue-labels.md", "Label Taxonomy"),
        ("docs/mklq/branch-protection.md", "Source-only repository checks"),
        ("docs/mklq/public-readiness.md", "Public Readiness"),
        ("docs/mklq/validation.md", "not a release certification"),
        ("docs/mklq/benchmark-evidence.md", "cross-machine performance certification"),
        (".github/pull_request_template.md", "Compatibility Boundary"),
        (".github/pull_request_template.md", "Benchmark Evidence"),
        (".github/labels.yml", "backend:cpu"),
        (".github/labels.yml", "backend:metal"),
        (".github/ISSUE_TEMPLATE/bug_report.yaml", "needs-repro"),
    ]


def banned_tokens() -> list[str]:
    return [
        "cuda-quantum" + "@nvidia.com",
        "github.com/NVIDIA/cuda-quantum" + "/actions",
        "nv-" + "slack",
        "ops-" + "bot",
        "copy-pr-" + "bot",
    ]


def public_metadata_paths(root: Path) -> list[Path]:
    paths = [
        root / "README.md",
        root / "CITATION.cff",
        root / "Contributing.md",
        root / "SECURITY.md",
        root / ".github" / "pull_request_template.md",
        root / "benchmarks" / "mklq" / "README.md",
    ]
    paths.extend((root / "docs" / "mklq").glob("*.md"))
    paths.extend(path for path in (root / ".github").rglob("*") if path.is_file())
    return sorted(set(paths))


def run_public_metadata_check(config: HealthcheckConfig) -> dict[str, Any]:
    missing: list[str] = []
    for relative_path, token in public_metadata_requirements():
        path = config.repo_root / relative_path
        if not path.exists():
            missing.append(f"{relative_path}: file is missing")
            continue
        if token not in path.read_text(encoding="utf-8", errors="replace"):
            missing.append(f"{relative_path}: missing {token!r}")

    banned_failures: list[str] = []
    for path in public_metadata_paths(config.repo_root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in banned_tokens():
            if token in text:
                banned_failures.append(
                    f"{command_path(config.repo_root, path)}: contains {token}")

    details = {
        "keyword_failures": missing,
        "banned_token_failures": banned_failures,
        "scanned_file_count": len(public_metadata_paths(config.repo_root)),
    }
    failures = missing + banned_failures
    return failed("public metadata check failed", details) if failures else passed(details)


def run_benchmark_summary_parse(config: HealthcheckConfig) -> dict[str, Any]:
    report_dir = config.repo_root / "benchmarks" / "mklq" / "reports"
    summaries = sorted(report_dir.glob("*.summary.json"))
    if not summaries:
        return failed("no sanitized MKL-Q benchmark summaries found")
    parsed: list[str] = []
    for path in summaries:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        parsed.append(command_path(config.repo_root, path))
    return passed({"summary_count": len(parsed), "summaries": parsed})


def run_py_compile(config: HealthcheckConfig) -> dict[str, Any]:
    command = [config.python_executable, "-m", "py_compile", *BENCHMARK_HELPERS]
    result = run_command(config, command)
    if result["returncode"] != 0:
        return failed("benchmark helper py_compile failed", result)
    return passed(result)


def markdown_files(root: Path) -> list[Path]:
    files = [root / item for item in PUBLIC_MARKDOWN_FILES]
    files.extend((root / "docs" / "mklq").glob("*.md"))
    return sorted(files)


def run_markdown_link_check(config: HealthcheckConfig) -> dict[str, Any]:
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    missing: list[str] = []
    files = markdown_files(config.repo_root)
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for target in link_re.findall(text):
            if "://" in target or target.startswith("#") or target.startswith("mailto:"):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            resolved = (path.parent / target_path).resolve()
            if not resolved.exists():
                missing.append(f"{command_path(config.repo_root, path)}: missing {target}")
    details = {
        "checked_file_count": len(files),
        "missing_links": missing,
    }
    return failed("local markdown link check failed", details) if missing else passed(details)


def run_benchmark_evidence_check(config: HealthcheckConfig) -> dict[str, Any]:
    summarize = config.repo_root / "benchmarks" / "mklq" / "summarize_reports.py"
    expected = config.repo_root / "docs" / "mklq" / "benchmark-evidence.md"
    with tempfile.TemporaryDirectory(prefix="mklq-healthcheck-") as tmpdir:
        generated = Path(tmpdir) / "benchmark-evidence.md"
        command = [
            config.python_executable,
            str(summarize),
            "--reports",
            "benchmarks/mklq/reports",
            "--format",
            "markdown",
            "--output",
            str(generated),
        ]
        result = run_command(config, command)
        if result["returncode"] != 0:
            return failed("benchmark evidence regeneration failed", result)
        if expected.read_text(encoding="utf-8") != generated.read_text(encoding="utf-8"):
            details = dict(result)
            details["expected"] = command_path(config.repo_root, expected)
            return failed("tracked benchmark evidence differs from regenerated output",
                          details)
    return passed({"expected": command_path(config.repo_root, expected)})


def run_harness_tests(config: HealthcheckConfig) -> dict[str, Any]:
    command = [
        config.python_executable,
        "-m",
        "pytest",
        "python/tests/backends/test_mklq_benchmark_harness.py",
        "-q",
    ]
    result = run_command(config, command)
    return passed(result) if result["returncode"] == 0 else failed(
        "benchmark harness tests failed", result)


def run_install_build(config: HealthcheckConfig) -> dict[str, Any]:
    command = [
        "cmake",
        "--build",
        command_path(config.repo_root, config.build_dir),
        "--target",
        "install",
        "-j",
        str(config.jobs),
    ]
    result = run_command(config, command)
    return passed(result) if result["returncode"] == 0 else failed(
        "install-prefix build failed", result)


def run_correctness_gate(config: HealthcheckConfig) -> dict[str, Any]:
    script = config.repo_root / "benchmarks" / "mklq" / "run_correctness_gate.py"
    command = [
        config.python_executable,
        str(script),
        "--install-prefix",
        str(config.install_prefix),
        "--pythonpath",
        config.pythonpath,
        "--nvqpp",
        str(config.nvqpp),
        "--build-dir",
        command_path(config.repo_root, config.build_dir),
    ]
    result = run_command(config, command)
    return passed(result) if result["returncode"] == 0 else failed(
        "one-command correctness gate failed", result)


def run_clean_cpu_benchmark(config: HealthcheckConfig) -> dict[str, Any]:
    script = config.repo_root / "benchmarks" / "mklq" / "run_clean_cpu_benchmark.py"
    command = [
        config.python_executable,
        str(script),
        "--pythonpath",
        config.pythonpath,
        "--stamp",
        config.stamp,
    ]
    result = run_command(config, command)
    return passed(result) if result["returncode"] == 0 else failed(
        "clean CPU benchmark gate failed", result)


def build_steps(config: HealthcheckConfig) -> list[Step]:
    steps = [
        Step("git_repository", "Check remotes, shallow state, and workflow set.",
             run_git_repository_check),
        Step("tracked_artifacts", "Reject generated, local, release, and agent files.",
             run_tracked_artifact_check),
        Step("public_metadata", "Check public metadata keywords and banned tokens.",
             run_public_metadata_check),
        Step("benchmark_summary_parse", "Parse sanitized benchmark summary JSON.",
             run_benchmark_summary_parse),
        Step("benchmark_helper_py_compile", "Compile public benchmark helper scripts.",
             run_py_compile),
        Step("markdown_links", "Check local markdown links in public MKL-Q docs.",
             run_markdown_link_check),
        Step("benchmark_evidence_regeneration",
             "Regenerate benchmark evidence markdown to a temp file and compare.",
             run_benchmark_evidence_check),
    ]
    if config.include_harness_tests:
        steps.append(
            Step("benchmark_harness_tests", "Run benchmark harness pytest coverage.",
                 run_harness_tests))
    if config.full:
        steps.append(
            Step("install_prefix_build", "Build and install MKL-Q to the prefix.",
                 run_install_build))
        steps.append(
            Step("correctness_gate",
                 "Run Python target, nvq++, and TargetConfig correctness gates.",
                 run_correctness_gate))
    if config.refresh_clean_cpu_benchmark:
        steps.append(
            Step("clean_cpu_benchmark",
                 "Refresh clean qpp-cpu versus mklq-cpu benchmark evidence.",
                 run_clean_cpu_benchmark))
    return steps


def step_plan(steps: list[Step]) -> list[dict[str, Any]]:
    return [{
        "name": step.name,
        "description": step.description,
        "command": step.command,
    } for step in steps]


def summarize(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    passed_count = sum(1 for result in step_results if result["status"] == "passed")
    failed_count = sum(1 for result in step_results if result["status"] == "failed")
    return {
        "status": "passed" if failed_count == 0 else "failed",
        "passed": passed_count,
        "failed": failed_count,
    }


def run_healthcheck(config: HealthcheckConfig) -> dict[str, Any]:
    steps = build_steps(config)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "repo_root": config.repo_root.as_posix(),
            "install_prefix": config.install_prefix.as_posix(),
            "build_dir": config.build_dir.as_posix(),
            "python_executable": config.python_executable,
            "pythonpath": config.pythonpath,
            "nvqpp": config.nvqpp.as_posix(),
            "stamp": config.stamp,
            "output": config.output.as_posix(),
            "jobs": config.jobs,
            "timeout_seconds": config.timeout_seconds,
            "require_clean": config.require_clean,
            "full": config.full,
            "include_harness_tests": config.include_harness_tests,
            "refresh_clean_cpu_benchmark": config.refresh_clean_cpu_benchmark,
        },
        "steps": step_plan(steps),
    }
    if config.plan_only:
        report["summary"] = {
            "status": "planned",
            "planned": len(steps),
            "failed": 0,
        }
        return report

    step_results: list[dict[str, Any]] = []
    for step in steps:
        start = time.perf_counter()
        try:
            result = step.runner(config)
        except Exception as exc:  # noqa: BLE001 - report all maintainer failures.
            result = failed(f"{type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - start
        step_results.append({
            "name": step.name,
            "description": step.description,
            "status": result["status"],
            "elapsed_seconds": elapsed,
            **{key: value for key, value in result.items() if key != "status"},
        })
    report["steps"] = step_results
    report["summary"] = summarize(step_results)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return report


def make_config(args: argparse.Namespace) -> HealthcheckConfig:
    root = repo_root()
    stamp = args.stamp or date.today().isoformat()
    install_prefix = args.install_prefix.expanduser()
    pythonpath = args.pythonpath or str(install_prefix)
    nvqpp = args.nvqpp or install_prefix / "bin" / "nvq++"
    output = args.output or output_default(stamp)
    return HealthcheckConfig(
        repo_root=root,
        install_prefix=resolve_path(root, install_prefix),
        build_dir=resolve_path(root, args.build_dir.expanduser()),
        python_executable=args.python_executable,
        pythonpath=pythonpath,
        nvqpp=resolve_path(root, nvqpp.expanduser()),
        stamp=stamp,
        output=resolve_path(root, output.expanduser()),
        jobs=args.jobs,
        timeout_seconds=args.timeout_seconds,
        tail_chars=args.tail_chars,
        require_clean=args.require_clean,
        full=args.full,
        include_harness_tests=not args.skip_harness_tests,
        refresh_clean_cpu_benchmark=args.refresh_clean_cpu_benchmark,
        plan_only=args.plan_only,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MKL-Q public repository health checks.")
    parser.add_argument("--install-prefix",
                        type=Path,
                        default=Path.home() / ".cudaq-mklq",
                        help="Installed CUDA-Q/MKL-Q prefix.")
    parser.add_argument("--pythonpath",
                        help="Override PYTHONPATH for installed MKL-Q checks.")
    parser.add_argument("--nvqpp",
                        type=Path,
                        help="Override the nvq++ path for correctness checks.")
    parser.add_argument("--build-dir",
                        type=Path,
                        default=Path("build-python"),
                        help="CMake build directory.")
    parser.add_argument("--python-executable",
                        default=sys.executable,
                        help="Python executable used for Python checks.")
    parser.add_argument("--stamp",
                        help="Date or label for healthcheck and benchmark outputs.")
    parser.add_argument("--output",
                        type=Path,
                        help="JSON output path. Defaults under ignored results/.")
    parser.add_argument("--jobs",
                        type=positive_int,
                        default=6,
                        help="Parallel build jobs used by --full.")
    parser.add_argument("--timeout-seconds",
                        type=positive_int,
                        default=900,
                        help="Per-step subprocess timeout.")
    parser.add_argument("--tail-chars",
                        type=positive_int,
                        default=DEFAULT_TAIL_CHARS,
                        help="Characters of stdout/stderr retained per subprocess.")
    parser.add_argument("--require-clean",
                        action="store_true",
                        help="Fail if tracked files are modified or untracked files are present.")
    parser.add_argument("--full",
                        action="store_true",
                        help="Also run install build and correctness gate.")
    parser.add_argument("--skip-harness-tests",
                        action="store_true",
                        help="Skip benchmark harness pytest coverage.")
    parser.add_argument("--refresh-clean-cpu-benchmark",
                        action="store_true",
                        help="Refresh clean CPU benchmark evidence. This may update tracked summary/docs files.")
    parser.add_argument("--plan-only",
                        action="store_true",
                        help="Print the healthcheck plan as JSON without running checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = make_config(args)
    report = run_healthcheck(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["status"] in {"passed", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
