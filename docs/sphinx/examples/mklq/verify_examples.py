#!/usr/bin/env python3
"""Run the public MKL-Q example smoke tests."""

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
from typing import Any


SCHEMA_VERSION = "mklq-example-verification-v1"
DEFAULT_TARGETS = ("mklq-cpu", "mklq-metal")
DEFAULT_EXAMPLES = (
    "bell",
    "ghz",
    "parametric",
    "phase_kickback",
    "clifford_chain",
)
EXPECTED_BITSTRINGS = {
    "bell": {"00", "11"},
    "ghz": {"000", "111"},
    "parametric": {"111"},
    "phase_kickback": {"11"},
    "clifford_chain": {"1111"},
}
COUNT_RE = re.compile(r"([01]+)\s*:\s*\d+")


@dataclass(frozen=True)
class ExampleConfig:
    repo_root: Path
    install_prefix: Path
    pythonpath: str
    nvqpp: Path
    python_executable: str
    targets: list[str]
    examples: list[str]
    shots: int
    output: Path
    timeout_seconds: int
    plan_only: bool
    skip_python: bool
    skip_cpp: bool


def repo_root() -> Path:
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                       cwd=Path.cwd(),
                                       stderr=subprocess.DEVNULL,
                                       text=True).strip()
        if root:
            return Path(root)
    except (OSError, subprocess.CalledProcessError):
        pass
    return Path(__file__).resolve().parents[4]


def output_default(stamp: str) -> Path:
    return Path("benchmarks/mklq/results") / f"example-smoke-{stamp}.json"


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer '{value}'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {parsed}")
    return parsed


def split_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one value")
    return items


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def command_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def output_tail(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def example_source(config: ExampleConfig, language: str, name: str) -> Path:
    extension = "py" if language == "python" else "cpp"
    return config.repo_root / "examples" / "mklq" / language / f"{name}.{extension}"


def target_label(target: str) -> str:
    return target.replace("-", "_")


def parse_bitstrings(stdout: str) -> set[str]:
    return set(COUNT_RE.findall(stdout))


def validate_counts(example: str, stdout: str) -> dict[str, Any]:
    observed = parse_bitstrings(stdout)
    expected = EXPECTED_BITSTRINGS[example]
    unexpected = sorted(observed - expected)
    return {
        "expected_bitstrings": sorted(expected),
        "observed_bitstrings": sorted(observed),
        "unexpected_bitstrings": unexpected,
        "counts_ok": bool(observed) and not unexpected,
    }


def run_process(config: ExampleConfig,
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
        "stdout_tail": output_tail(result.stdout),
        "stderr_tail": output_tail(result.stderr),
    }


def python_step(config: ExampleConfig, example: str, target: str) -> dict[str, Any]:
    source = example_source(config, "python", example)
    command = [
        config.python_executable,
        command_path(config.repo_root, source),
        "--target",
        target,
        "--shots",
        str(config.shots),
    ]
    result = run_process(config, command, {"PYTHONPATH": config.pythonpath})
    validation = validate_counts(example, result["stdout_tail"])
    status = "passed" if result["returncode"] == 0 and validation["counts_ok"] else "failed"
    return {
        "name": f"python_{example}_{target_label(target)}",
        "kind": "python",
        "example": example,
        "target": target,
        "source": command_path(config.repo_root, source),
        "status": status,
        **result,
        **validation,
    }


def cpp_steps(config: ExampleConfig, example: str, target: str,
              tmpdir: Path) -> list[dict[str, Any]]:
    source = example_source(config, "cpp", example)
    executable = tmpdir / f"{example}_{target_label(target)}"
    compile_command = [
        str(config.nvqpp),
        "--target",
        target,
        command_path(config.repo_root, source),
        "-o",
        str(executable),
    ]
    compile_result = run_process(config, compile_command)
    compile_status = "passed" if compile_result["returncode"] == 0 and executable.exists(
    ) else "failed"
    compile_step = {
        "name": f"cpp_compile_{example}_{target_label(target)}",
        "kind": "cpp_compile",
        "example": example,
        "target": target,
        "source": command_path(config.repo_root, source),
        "executable": str(executable),
        "status": compile_status,
        **compile_result,
    }
    if compile_status != "passed":
        return [compile_step]

    run_command = [str(executable), str(config.shots)]
    run_result = run_process(config, run_command)
    validation = validate_counts(example, run_result["stdout_tail"])
    run_status = "passed" if run_result["returncode"] == 0 and validation[
        "counts_ok"] else "failed"
    run_step = {
        "name": f"cpp_run_{example}_{target_label(target)}",
        "kind": "cpp_run",
        "example": example,
        "target": target,
        "source": command_path(config.repo_root, source),
        "status": run_status,
        **run_result,
        **validation,
    }
    return [compile_step, run_step]


def build_plan(config: ExampleConfig) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for example in config.examples:
        for target in config.targets:
            if not config.skip_python:
                steps.append({
                    "name": f"python_{example}_{target_label(target)}",
                    "kind": "python",
                    "example": example,
                    "target": target,
                    "source": command_path(config.repo_root,
                                           example_source(config, "python", example)),
                })
            if not config.skip_cpp:
                steps.append({
                    "name": f"cpp_compile_{example}_{target_label(target)}",
                    "kind": "cpp_compile",
                    "example": example,
                    "target": target,
                    "source": command_path(config.repo_root,
                                           example_source(config, "cpp", example)),
                })
                steps.append({
                    "name": f"cpp_run_{example}_{target_label(target)}",
                    "kind": "cpp_run",
                    "example": example,
                    "target": target,
                    "source": command_path(config.repo_root,
                                           example_source(config, "cpp", example)),
                })
    return steps


def summarize(steps: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for step in steps if step["status"] == "passed")
    failed = sum(1 for step in steps if step["status"] == "failed")
    return {
        "status": "passed" if failed == 0 else "failed",
        "passed": passed,
        "failed": failed,
    }


def run_examples(config: ExampleConfig) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "repo_root": config.repo_root.as_posix(),
            "install_prefix": config.install_prefix.as_posix(),
            "pythonpath": config.pythonpath,
            "nvqpp": config.nvqpp.as_posix(),
            "python_executable": config.python_executable,
            "targets": config.targets,
            "examples": config.examples,
            "shots": config.shots,
            "output": config.output.as_posix(),
            "timeout_seconds": config.timeout_seconds,
            "skip_python": config.skip_python,
            "skip_cpp": config.skip_cpp,
        },
        "steps": build_plan(config),
    }
    if config.plan_only:
        report["summary"] = {
            "status": "planned",
            "planned": len(report["steps"]),
            "failed": 0,
        }
        return report

    steps: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mklq-examples-") as tmp:
        tmpdir = Path(tmp)
        for example in config.examples:
            for target in config.targets:
                if not config.skip_python:
                    steps.append(python_step(config, example, target))
                if not config.skip_cpp:
                    steps.extend(cpp_steps(config, example, target, tmpdir))
    report["steps"] = steps
    report["summary"] = summarize(steps)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return report


def make_config(args: argparse.Namespace) -> ExampleConfig:
    root = repo_root()
    stamp = args.stamp or date.today().isoformat()
    install_prefix = args.install_prefix.expanduser()
    pythonpath = args.pythonpath or str(install_prefix)
    nvqpp = args.nvqpp or install_prefix / "bin" / "nvq++"
    output = args.output or output_default(stamp)
    targets = args.targets or list(DEFAULT_TARGETS)
    examples = args.examples or list(DEFAULT_EXAMPLES)
    unknown_targets = sorted(set(targets) - set(DEFAULT_TARGETS))
    unknown_examples = sorted(set(examples) - set(DEFAULT_EXAMPLES))
    if unknown_targets:
        raise SystemExit(f"unsupported target(s): {', '.join(unknown_targets)}")
    if unknown_examples:
        raise SystemExit(f"unsupported example(s): {', '.join(unknown_examples)}")
    return ExampleConfig(
        repo_root=root,
        install_prefix=resolve_path(root, install_prefix),
        pythonpath=pythonpath,
        nvqpp=resolve_path(root, nvqpp),
        python_executable=args.python_executable,
        targets=targets,
        examples=examples,
        shots=args.shots,
        output=resolve_path(root, output.expanduser()),
        timeout_seconds=args.timeout_seconds,
        plan_only=args.plan_only,
        skip_python=args.skip_python,
        skip_cpp=args.skip_cpp,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MKL-Q public example smoke tests.")
    parser.add_argument("--install-prefix",
                        type=Path,
                        default=Path.home() / ".cudaq-mklq",
                        help="Installed CUDA-Q/MKL-Q prefix.")
    parser.add_argument("--pythonpath",
                        help="Override PYTHONPATH for Python examples.")
    parser.add_argument("--nvqpp",
                        type=Path,
                        help="Override nvq++ for C++ examples.")
    parser.add_argument("--python-executable",
                        default=sys.executable,
                        help="Python executable used for Python examples.")
    parser.add_argument("--targets",
                        type=split_csv,
                        help="Comma-separated targets. Defaults to mklq-cpu,mklq-metal.")
    parser.add_argument("--examples",
                        type=split_csv,
                        help="Comma-separated examples. Defaults to bell,ghz.")
    parser.add_argument("--shots",
                        type=positive_int,
                        default=20,
                        help="Sample count for each example run.")
    parser.add_argument("--stamp",
                        help="Date or label for the default output filename.")
    parser.add_argument("--output",
                        type=Path,
                        help="JSON output path. Defaults under ignored results/.")
    parser.add_argument("--timeout-seconds",
                        type=positive_int,
                        default=120,
                        help="Per-command timeout.")
    parser.add_argument("--skip-python",
                        action="store_true",
                        help="Skip Python examples.")
    parser.add_argument("--skip-cpp",
                        action="store_true",
                        help="Skip C++ examples.")
    parser.add_argument("--plan-only",
                        action="store_true",
                        help="Print the planned example checks without running them.")
    return parser.parse_args()


def main() -> int:
    config = make_config(parse_args())
    report = run_examples(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["status"] in {"passed", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
