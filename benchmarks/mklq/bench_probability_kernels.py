#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Benchmark standalone probability-vector kernels for MKL-Q tuning."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "mklq-probability-benchmark-v1"
DEFAULT_VARIANTS = (
    "scalar-norm",
    "scalar-split",
    "accelerate-interleaved",
    "openmp-split",
    "accelerate-vdsp",
)
DEFAULT_QUBITS = (15, 16, 17, 18, 19, 20)
ENV_KEYS = (
    "CXX",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_DYNAMIC",
    "VECLIB_MAXIMUM_THREADS",
)


def parse_csv(value: str) -> list[str]:
  items = [item.strip() for item in value.split(",") if item.strip()]
  if not items:
    raise argparse.ArgumentTypeError("expected at least one comma-separated item")
  return items


def parse_qubits(value: str) -> list[int]:
  qubits: list[int] = []
  for item in parse_csv(value):
    try:
      count = int(item)
    except ValueError as exc:
      raise argparse.ArgumentTypeError(
          f"invalid qubit count '{item}'") from exc
    if count < 1 or count >= 63:
      raise argparse.ArgumentTypeError(
          f"qubit count must be in [1, 62], got {count}")
    qubits.append(count)
  return qubits


def positive_int(value: str) -> int:
  try:
    parsed = int(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f"invalid integer '{value}'") from exc
  if parsed < 1:
    raise argparse.ArgumentTypeError(f"expected positive integer, got {parsed}")
  return parsed


def non_negative_int(value: str) -> int:
  try:
    parsed = int(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f"invalid integer '{value}'") from exc
  if parsed < 0:
    raise argparse.ArgumentTypeError(
        f"expected non-negative integer, got {parsed}")
  return parsed


def command_output(args: list[str]) -> str:
  try:
    return subprocess.check_output(args,
                                   stderr=subprocess.DEVNULL,
                                   text=True,
                                   timeout=2).strip()
  except Exception:
    return ""


def machine_snapshot() -> dict[str, Any]:
  cpu_brand = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
  core_count = command_output(["sysctl", "-n", "hw.ncpu"])
  memsize = command_output(["sysctl", "-n", "hw.memsize"])
  macos_version = command_output(["sw_vers", "-productVersion"])

  return {
      "platform": platform.platform(),
      "machine": platform.machine(),
      "processor": platform.processor(),
      "python": sys.version.split()[0],
      "python_executable": sys.executable,
      "cpu_brand": cpu_brand,
      "logical_cores": int(core_count) if core_count.isdigit() else None,
      "memory_bytes": int(memsize) if memsize.isdigit() else None,
      "macos_version": macos_version or None,
  }


def git_snapshot() -> dict[str, Any]:
  status = command_output(["git", "status", "--short"])
  return {
      "root": command_output(["git", "rev-parse", "--show-toplevel"]) or None,
      "branch": command_output(["git", "branch", "--show-current"]) or None,
      "commit": command_output(["git", "rev-parse", "HEAD"]) or None,
      "dirty": bool(status),
      "status_short": status.splitlines() if status else [],
  }


def environment_snapshot() -> dict[str, str | None]:
  return {key: os.environ.get(key) for key in ENV_KEYS}


def result_template(variant: str, qubits: int,
                    repeats: int) -> dict[str, Any]:
  return {
      "variant": variant,
      "qubits": qubits,
      "dimension": 1 << qubits,
      "repeats": repeats,
      "status": "planned",
      "metrics": {},
  }


def choose_compiler() -> str:
  candidates = []
  env_cxx = os.environ.get("CXX")
  if env_cxx:
    candidates.append(env_cxx)
  candidates.extend((
      "/opt/homebrew/opt/llvm/bin/clang++",
      "/usr/bin/clang++",
      shutil.which("clang++"),
      shutil.which("c++"),
  ))
  for candidate in candidates:
    if candidate and Path(candidate).exists():
      compiler = str(candidate)
      if compiler_can_build(compiler):
        return compiler
  raise RuntimeError("could not locate a C++ compiler")


def sdk_compile_flags() -> list[str]:
  if sys.platform != "darwin":
    return []
  sdk_path = command_output(["xcrun", "--show-sdk-path"])
  return ["-isysroot", sdk_path] if sdk_path else []


def compiler_can_build(compiler: str) -> bool:
  with tempfile.TemporaryDirectory(prefix="mklq-cxx-check-") as tmpdir:
    source = Path(tmpdir) / "check.cpp"
    binary = Path(tmpdir) / "check"
    source.write_text("#include <algorithm>\nint main(){return 0;}\n",
                      encoding="utf-8")
    result = subprocess.run([
        compiler,
        "-std=c++20",
        *sdk_compile_flags(),
        str(source),
        "-o",
        str(binary),
    ],
                            capture_output=True,
                            text=True)
    return result.returncode == 0


def compiler_supports_openmp(compiler: str) -> bool:
  with tempfile.TemporaryDirectory(prefix="mklq-openmp-check-") as tmpdir:
    source = Path(tmpdir) / "check.cpp"
    binary = Path(tmpdir) / "check"
    source.write_text("#include <omp.h>\nint main(){return omp_get_max_threads()<1;}\n",
                      encoding="utf-8")
    result = subprocess.run([
        compiler,
        "-std=c++20",
        "-O2",
        "-fopenmp",
        *sdk_compile_flags(),
        str(source),
        "-o",
        str(binary),
    ],
                            capture_output=True,
                            text=True)
    return result.returncode == 0


def openmp_link_flags(compiler: str) -> list[str]:
  candidates = [
      Path(compiler).resolve().parents[1] / "lib" / "libomp.dylib",
      Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
  ]
  for candidate in candidates:
    if candidate.exists():
      return [f"-Wl,-rpath,{candidate.parent}"]
  return []


def compile_binary(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
  repo_root = Path(__file__).resolve().parents[2]
  source = repo_root / "benchmarks" / "mklq" / "probability_kernels.cpp"
  compiler = choose_compiler()
  binary = Path(args.binary) if args.binary else Path(tempfile.mkdtemp(
      prefix="mklq-probability-bench-")) / "probability_kernels"

  command = [
      compiler,
      "-std=c++20",
  ]

  openmp_enabled = compiler_supports_openmp(compiler)
  if openmp_enabled:
    command.append("-fopenmp")

  command.extend([
      "-O3",
      "-DNDEBUG",
      *sdk_compile_flags(),
      str(source),
      "-o",
      str(binary),
  ])
  if openmp_enabled:
    command.extend(openmp_link_flags(compiler))

  accelerate_enabled = sys.platform == "darwin"
  if accelerate_enabled:
    command.extend(["-DMKLQ_HAS_ACCELERATE=1", "-framework", "Accelerate"])

  result = subprocess.run(command, capture_output=True, text=True)
  metadata = {
      "compiler": compiler,
      "command": command,
      "returncode": result.returncode,
      "stdout": result.stdout.strip(),
      "stderr": result.stderr.strip(),
      "openmp_enabled": openmp_enabled,
      "accelerate_enabled": accelerate_enabled,
      "binary": str(binary),
  }
  if result.returncode != 0:
    raise RuntimeError("probability benchmark compile failed: " +
                       result.stderr.strip())
  return binary, metadata


def build_report(args: argparse.Namespace) -> dict[str, Any]:
  report = {
      "schema_version": SCHEMA_VERSION,
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "machine": machine_snapshot(),
      "provenance": {
          "cwd": str(Path.cwd()),
          "git": git_snapshot(),
          "environment": environment_snapshot(),
      },
      "config": {
          "variants": args.variants,
          "qubits": args.qubits,
          "repeats": args.repeats,
          "warmups": args.warmups,
          "dry_run": args.dry_run,
          "command": sys.argv,
      },
      "results": [],
  }

  if args.dry_run:
    for variant in args.variants:
      for qubits in args.qubits:
        report["results"].append(
            result_template(variant, qubits, args.repeats))
    return report

  binary, compile_metadata = compile_binary(args)
  report["compile"] = compile_metadata
  command = [
      str(binary),
      "--variants",
      ",".join(args.variants),
      "--qubits",
      ",".join(str(qubit) for qubit in args.qubits),
      "--repeats",
      str(args.repeats),
      "--warmups",
      str(args.warmups),
      "--seed",
      str(args.seed),
  ]
  result = subprocess.run(command, capture_output=True, text=True)
  report["execution"] = {
      "command": command,
      "returncode": result.returncode,
      "stdout": result.stdout.strip(),
      "stderr": result.stderr.strip(),
  }
  if result.returncode != 0:
    raise RuntimeError("probability benchmark execution failed: " +
                       result.stderr.strip())

  child = json.loads(result.stdout)
  rows = child.get("results")
  if not isinstance(rows, list):
    raise RuntimeError("probability benchmark did not emit results")
  for row in rows:
    row["repeats"] = args.repeats
  report["results"] = rows
  return report


def write_report(report: dict[str, Any], output: str | None) -> None:
  payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if output:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
  else:
    print(payload, end="")


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Benchmark probability-vector kernels for MKL-Q tuning.")
  parser.add_argument("--variants",
                      type=parse_csv,
                      default=list(DEFAULT_VARIANTS),
                      help="Comma-separated kernel variants.")
  parser.add_argument("--qubits",
                      type=parse_qubits,
                      default=list(DEFAULT_QUBITS),
                      help="Comma-separated qubit counts.")
  parser.add_argument("--repeats",
                      type=positive_int,
                      default=5,
                      help="Measured repeats per variant/qubit row.")
  parser.add_argument("--warmups",
                      type=non_negative_int,
                      default=2,
                      help="Warmup executions before measured repeats.")
  parser.add_argument("--seed",
                      type=non_negative_int,
                      default=13,
                      help="Deterministic synthetic state seed.")
  parser.add_argument("--binary",
                      help="Optional output path for the compiled benchmark.")
  parser.add_argument("--output",
                      help="JSON output path. Defaults to stdout.")
  parser.add_argument("--dry-run",
                      action="store_true",
                      help="Write the benchmark plan without compiling C++.")
  parser.add_argument("--allow-errors",
                      action="store_true",
                      help="Return 0 even when one or more rows are unsupported.")
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = make_parser()
  args = parser.parse_args(argv)
  report = build_report(args)
  write_report(report, args.output)
  if (not args.dry_run and not args.allow_errors and any(
      row.get("status") != "ok" for row in report["results"])):
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
