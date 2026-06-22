#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Benchmark MKL-Q Apple Silicon simulator targets.

This script intentionally records raw measurements and machine metadata. It
does not make performance claims; compare JSON reports after reviewing the
machine, build, and command metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = "mklq-benchmark-v1"
APPLE_SILICON_TARGETS = ("qpp-cpu", "mklq-cpu", "mklq-metal")
PORTABLE_DEFAULT_TARGETS = ("qpp-cpu",)
TARGET_NOTES = {
    "mklq-metal":
        "Experimental mixed-path target: mklq_metal uses resident fp32 Metal single-target/two-target/three-target/probability-fill kernels, cost-gated resident full-register and marginal probability kernels for sampling, a measured-qubit probability-reduction kernel, and a resident measurement-collapse path with MKL-Q fp64 CPU-oracle fallback for unsupported paths; sample draw/count accumulation remains host-side, not full Metal GPU backend evidence.",
}
METAL_EVIDENCE_BOUNDARY = (
    "benchmark harness static case-map label only; not a runtime counter, "
    "release sign-off, or proof that every operation stayed on Metal")
METAL_SINGLE_GATE_SCOPE = (
    "resident fp32 Metal single-target gate update followed by host readback "
    "for cudaq.get_state")
METAL_CONTROLLED_GATE_SCOPE = (
    "resident fp32 Metal controlled gate update followed by host readback for "
    "cudaq.get_state")
METAL_TWO_QUBIT_SCOPE = (
    "resident fp32 Metal two-target gate update followed by host readback for "
    "cudaq.get_state")
METAL_THREE_QUBIT_SCOPE = (
    "resident fp32 Metal three-target gate update followed by host readback for "
    "cudaq.get_state")
METAL_COMPOSITE_SCOPE = (
    "experimental mklq-metal mixed-path composite state-vector update followed "
    "by host readback for cudaq.get_state")
METAL_SAMPLING_SCOPE = (
    "mixed-path Metal probability fill with host-side sample draw/count "
    "accumulation")
METAL_SPARSE_SAMPLING_SCOPE = (
    "mixed-path sparse or deterministic sampling with host-side count "
    "accumulation")
METAL_PATH_CASES = {
    "gate-state": ("mklq_metal_mixed_gate_state_host_readback",
                   METAL_COMPOSITE_SCOPE),
    "single-qubit-state":
        ("mklq_metal_resident_single_gate_state_host_readback",
         METAL_SINGLE_GATE_SCOPE),
    "h-state": ("mklq_metal_resident_single_gate_state_host_readback",
                METAL_SINGLE_GATE_SCOPE),
    "y-state": ("mklq_metal_resident_single_gate_state_host_readback",
                METAL_SINGLE_GATE_SCOPE),
    "rx-state": ("mklq_metal_resident_single_gate_state_host_readback",
                 METAL_SINGLE_GATE_SCOPE),
    "ry-state": ("mklq_metal_resident_single_gate_state_host_readback",
                 METAL_SINGLE_GATE_SCOPE),
    "rz-state": ("mklq_metal_resident_single_gate_state_host_readback",
                 METAL_SINGLE_GATE_SCOPE),
    "controlled-state":
        ("mklq_metal_mixed_controlled_gate_state_host_readback",
         METAL_CONTROLLED_GATE_SCOPE),
    "ch-state": ("mklq_metal_resident_controlled_gate_state_host_readback",
                 METAL_CONTROLLED_GATE_SCOPE),
    "cy-state": ("mklq_metal_resident_controlled_gate_state_host_readback",
                 METAL_CONTROLLED_GATE_SCOPE),
    "crx-state": ("mklq_metal_resident_controlled_gate_state_host_readback",
                  METAL_CONTROLLED_GATE_SCOPE),
    "cry-state": ("mklq_metal_resident_controlled_gate_state_host_readback",
                  METAL_CONTROLLED_GATE_SCOPE),
    "crz-state": ("mklq_metal_resident_controlled_gate_state_host_readback",
                  METAL_CONTROLLED_GATE_SCOPE),
    "cz-state": ("mklq_metal_resident_controlled_gate_state_host_readback",
                 METAL_CONTROLLED_GATE_SCOPE),
    "two-qubit-state": ("mklq_metal_resident_two_gate_state_host_readback",
                        METAL_TWO_QUBIT_SCOPE),
    "three-qubit-state": ("mklq_metal_resident_three_gate_state_host_readback",
                          METAL_THREE_QUBIT_SCOPE),
    "qft-like-state": ("mklq_metal_mixed_composite_state_host_readback",
                       METAL_COMPOSITE_SCOPE),
    "seeded-clifford-state":
        ("mklq_metal_mixed_composite_state_host_readback",
         METAL_COMPOSITE_SCOPE),
    "sample-basis": ("mklq_metal_sparse_sampling_host_counts",
                     METAL_SPARSE_SAMPLING_SCOPE),
    "sample-ghz": ("mklq_metal_sparse_sampling_host_counts",
                   METAL_SPARSE_SAMPLING_SCOPE),
    "sample-full-register": ("mklq_metal_mixed_sampling_host_counts",
                             METAL_SAMPLING_SCOPE),
    "sample-partial-register": ("mklq_metal_mixed_sampling_host_counts",
                                METAL_SAMPLING_SCOPE),
}


def default_targets_for_platform(system: str | None = None,
                                 machine: str | None = None) -> list[str]:
  system = system if system is not None else platform.system()
  machine = machine if machine is not None else platform.machine()
  if system == "Darwin" and machine in {"arm64", "aarch64"}:
    return list(APPLE_SILICON_TARGETS)
  return list(PORTABLE_DEFAULT_TARGETS)


def metal_path_metrics(target: str, case: str) -> dict[str, Any]:
  if target != "mklq-metal":
    return {}
  label, scope = METAL_PATH_CASES.get(
      case, ("mklq_metal_experimental_mixed_path", METAL_COMPOSITE_SCOPE))
  return {
      "metal_path_label": label,
      "metal_path_scope": scope,
      "metal_path_label_source": "benchmark_harness_static_case_map",
      "metal_evidence_boundary": METAL_EVIDENCE_BOUNDARY,
      "metal_full_native": False,
      "metal_runtime_counter": False,
  }


SINGLE_GATE_STATE_CASES = {
    "h-state": ("h", "h_gate_count", "h_gate_state_throughput_per_second"),
    "y-state": ("y", "y_gate_count", "y_gate_state_throughput_per_second"),
    "rx-state": ("rx", "rx_gate_count", "rx_gate_state_throughput_per_second"),
    "ry-state": ("ry", "ry_gate_count", "ry_gate_state_throughput_per_second"),
    "rz-state": ("rz", "rz_gate_count", "rz_gate_state_throughput_per_second"),
}
CONTROLLED_GATE_STATE_CASES = {
    "ch-state": ("ch", "ch_gate_count", "ch_gate_state_throughput_per_second"),
    "cy-state": ("cy", "cy_gate_count", "cy_gate_state_throughput_per_second"),
    "crx-state": ("crx", "crx_gate_count",
                  "crx_gate_state_throughput_per_second"),
    "cry-state": ("cry", "cry_gate_count",
                  "cry_gate_state_throughput_per_second"),
    "crz-state": ("crz", "crz_gate_count",
                  "crz_gate_state_throughput_per_second"),
}
DEFAULT_CASES = (
    "gate-state",
    "sample-basis",
    "sample-ghz",
    "sample-full-register",
    "sample-partial-register",
    "single-qubit-state",
    "h-state",
    "y-state",
    "rx-state",
    "ry-state",
    "rz-state",
    "controlled-state",
    "ch-state",
    "cy-state",
    "crx-state",
    "cry-state",
    "crz-state",
    "cz-state",
    "two-qubit-state",
    "three-qubit-state",
    "qft-like-state",
    "seeded-clifford-state",
)
DEFAULT_QUBITS = (4, 8, 12)
SEEDED_CLIFFORD_SEED = 17
THREE_QUBIT_OPERATION_NAME = "mklq_bench_flip_all_3"
PROVENANCE_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_DYNAMIC",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "DYLD_LIBRARY_PATH",
    "PYTHONPATH",
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


def parse_shot_counts(value: str) -> list[int]:
  return [positive_int(item) for item in parse_csv(value)]


def validate_cases(cases: list[str]) -> list[str]:
  allowed = set(DEFAULT_CASES)
  unknown = sorted(set(cases) - allowed)
  if unknown:
    raise argparse.ArgumentTypeError(
        f"unknown benchmark case(s): {', '.join(unknown)}")
  return cases


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
  return {key: os.environ.get(key) for key in PROVENANCE_ENV_KEYS}


def runtime_snapshot(cudaq: Any) -> dict[str, Any]:
  module_file = getattr(cudaq, "__file__", None)
  return {
      "cudaq_module_file": module_file,
      "cudaq_version": getattr(cudaq, "__version__", None),
      "python_prefix": sys.prefix,
      "python_base_prefix": sys.base_prefix,
      "sys_path_head": sys.path[:5],
      "module_from_build_tree": "build-python" in module_file
      if module_file else None,
  }


def estimated_state_bytes(qubits: int) -> int:
  return 16 * (1 << qubits)


def process_max_rss_bytes() -> int:
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if sys.platform == "darwin":
    return int(rss)
  return int(rss) * 1024


def build_gate_state_kernel(cudaq: Any, qubits: int,
                            layers: int) -> tuple[Any, int]:
  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  gate_count = 0

  for layer in range(layers):
    theta = 0.125 + layer * 0.001
    for index in range(qubits):
      kernel.h(q[index])
      kernel.rx(theta, q[index])
      kernel.rz(-theta, q[index])
      gate_count += 3
    for index in range(qubits - 1):
      kernel.cx(q[index], q[index + 1])
      gate_count += 1

  return kernel, gate_count


def build_single_qubit_state_kernel(cudaq: Any, qubits: int,
                                    layers: int) -> tuple[Any, int]:
  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  gate_count = 0

  for layer in range(layers):
    theta = 0.125 + layer * 0.001
    for index in range(qubits):
      kernel.h(q[index])
      kernel.rx(theta, q[index])
      kernel.ry(0.5 * theta, q[index])
      kernel.rz(-theta, q[index])
      gate_count += 4

  return kernel, gate_count


def build_single_gate_state_kernel(
    cudaq: Any, qubits: int,
    layers: int, gate_name: str) -> tuple[Any, int, int, int]:
  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  state_prep_gate_count = 0
  single_gate_count = 0

  for index in range(qubits):
    theta = 0.061 + 0.001 * index
    kernel.ry(theta, q[index])
    kernel.rz(-0.5 * theta, q[index])
    state_prep_gate_count += 2

  for layer in range(layers):
    theta = 0.125 + layer * 0.001
    for index in range(qubits):
      if gate_name == "h":
        kernel.h(q[index])
      elif gate_name == "y":
        kernel.y(q[index])
      elif gate_name == "rx":
        kernel.rx(theta, q[index])
      elif gate_name == "ry":
        kernel.ry(theta, q[index])
      elif gate_name == "rz":
        kernel.rz(theta, q[index])
      else:
        raise ValueError(f"unsupported single-gate benchmark: {gate_name}")
      single_gate_count += 1

  return (kernel, state_prep_gate_count + single_gate_count,
          state_prep_gate_count, single_gate_count)


def build_controlled_state_kernel(cudaq: Any, qubits: int,
                                  layers: int) -> tuple[Any, int]:
  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  gate_count = 0

  for index in range(qubits):
    theta = 0.043 + 0.0017 * index
    kernel.ry(theta, q[index])
    if index % 2:
      kernel.rz(-0.5 * theta, q[index])
      gate_count += 1
    gate_count += 1

  for layer in range(layers):
    theta = 0.125 + layer * 0.001
    for index in range(qubits - 1):
      kernel.cx(q[index], q[index + 1])
      kernel.cz(q[index + 1], q[index])
      kernel.crx(theta, q[index], q[index + 1])
      gate_count += 3

  return kernel, gate_count


def build_controlled_gate_state_kernel(
    cudaq: Any, qubits: int,
    layers: int, gate_name: str) -> tuple[Any, int, int, int]:
  if qubits < 2:
    raise ValueError("controlled-gate benchmarks require at least 2 qubits")

  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  state_prep_gate_count = 0
  controlled_gate_count = 0

  for index in range(qubits):
    theta = 0.043 + 0.0017 * index
    kernel.ry(theta, q[index])
    kernel.rz(-0.5 * theta, q[index])
    state_prep_gate_count += 2

  for layer in range(layers):
    theta = 0.125 + layer * 0.001
    for index in range(qubits - 1):
      if gate_name == "ch":
        kernel.ch(q[index], q[index + 1])
      elif gate_name == "cy":
        kernel.cy(q[index], q[index + 1])
      elif gate_name == "crx":
        kernel.crx(theta, q[index], q[index + 1])
      elif gate_name == "cry":
        kernel.cry(theta, q[index], q[index + 1])
      elif gate_name == "crz":
        kernel.crz(theta, q[index], q[index + 1])
      else:
        raise ValueError(f"unsupported controlled-gate benchmark: {gate_name}")
      controlled_gate_count += 1

  return (kernel, state_prep_gate_count + controlled_gate_count,
          state_prep_gate_count, controlled_gate_count)


def build_cz_state_kernel(cudaq: Any, qubits: int,
                          layers: int) -> tuple[Any, int, int, int]:
  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  state_prep_gate_count = 0
  cz_gate_count = 0

  for index in range(qubits):
    theta = 0.043 + 0.0017 * index
    kernel.ry(theta, q[index])
    state_prep_gate_count += 1

  for layer in range(layers):
    for index in range(qubits - 1):
      if layer % 2:
        kernel.cz(q[index + 1], q[index])
      else:
        kernel.cz(q[index], q[index + 1])
      cz_gate_count += 1

  return (kernel, state_prep_gate_count + cz_gate_count, state_prep_gate_count,
          cz_gate_count)


def build_two_qubit_state_kernel(cudaq: Any, qubits: int,
                                 layers: int) -> tuple[Any, int]:
  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  gate_count = 0

  for index in range(qubits):
    theta = 0.043 + 0.0017 * index
    kernel.ry(theta, q[index])
    if index % 2:
      kernel.rz(-0.5 * theta, q[index])
      gate_count += 1
    gate_count += 1

  for layer in range(layers):
    for index in range(0, qubits - 1, 2):
      kernel.swap(q[index], q[index + 1])
      gate_count += 1
    for index in range(1, qubits - 1, 2):
      kernel.swap(q[index], q[index + 1])
      gate_count += 1

  return kernel, gate_count


def three_qubit_flip_all_matrix() -> list[list[int]]:
  return [[1 if column == 7 - row else 0 for column in range(8)]
          for row in range(8)]


def build_three_qubit_state_kernel(
    cudaq: Any, qubits: int,
    layers: int) -> tuple[Any, int, int, int]:
  if qubits < 3:
    raise ValueError("three-qubit benchmarks require at least 3 qubits")
  if not hasattr(cudaq, "register_operation"):
    raise RuntimeError("cudaq.register_operation is required")

  cudaq.register_operation(THREE_QUBIT_OPERATION_NAME,
                           three_qubit_flip_all_matrix())

  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  state_prep_gate_count = 0
  three_qubit_gate_count = 0

  for index in range(qubits):
    theta = 0.043 + 0.0017 * index
    kernel.ry(theta, q[index])
    kernel.rz(-0.5 * theta, q[index])
    state_prep_gate_count += 2

  for layer in range(layers):
    if layer % 2:
      windows = range(qubits - 3, -1, -1)
    else:
      windows = range(0, qubits - 2)
    for index in windows:
      kernel.__getattr__(THREE_QUBIT_OPERATION_NAME)(q[index], q[index + 1],
                                                     q[index + 2])
      three_qubit_gate_count += 1

  gate_count = state_prep_gate_count + three_qubit_gate_count
  return (kernel, gate_count, state_prep_gate_count, three_qubit_gate_count)


def build_qft_like_state_kernel(
    cudaq: Any, qubits: int, layers: int) -> tuple[Any, int, int, int, int, int]:
  if qubits < 2:
    raise ValueError("qft-like benchmarks require at least 2 qubits")

  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  state_prep_gate_count = 0
  h_gate_count = 0
  crz_gate_count = 0
  swap_gate_count = 0

  kernel.x(q[0])
  kernel.x(q[qubits - 1])
  state_prep_gate_count += 2

  for _ in range(layers):
    for target in range(qubits):
      kernel.h(q[target])
      h_gate_count += 1
      for control in range(target + 1, qubits):
        angle = math.pi / float(1 << (control - target + 1))
        kernel.crz(angle, q[control], q[target])
        crz_gate_count += 1

    for index in range(qubits // 2):
      kernel.swap(q[index], q[qubits - index - 1])
      swap_gate_count += 1

  qft_like_gate_count = h_gate_count + crz_gate_count + swap_gate_count
  return (kernel, state_prep_gate_count + qft_like_gate_count,
          state_prep_gate_count, h_gate_count, crz_gate_count, swap_gate_count)


def build_seeded_clifford_state_kernel(
    cudaq: Any,
    qubits: int,
    layers: int,
    seed: int = SEEDED_CLIFFORD_SEED) -> tuple[Any, int, int, int]:
  if qubits < 3:
    raise ValueError("seeded Clifford benchmarks require at least 3 qubits")

  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  single_gate_count = 0
  two_qubit_gate_count = 0

  for layer in range(layers):
    for step in range(qubits):
      sequence_index = layer * qubits + step
      target = (seed + layer + step) % qubits
      selector = (seed + 7 * sequence_index) % 6
      if selector == 0:
        kernel.h(q[target])
      elif selector == 1:
        kernel.s(q[target])
      elif selector == 2:
        kernel.sdg(q[target])
      elif selector == 3:
        kernel.x(q[target])
      elif selector == 4:
        kernel.y(q[target])
      else:
        kernel.z(q[target])
      single_gate_count += 1

      control = (target + layer + step + 1) % qubits
      if control == target:
        control = (control + 1) % qubits
      other = (target + seed + layer + step + 2) % qubits
      while other in {target, control}:
        other = (other + 1) % qubits

      if sequence_index % 4 == 0:
        kernel.cx(q[control], q[target])
      elif sequence_index % 4 == 1:
        kernel.cy(q[control], q[target])
      elif sequence_index % 4 == 2:
        kernel.cz(q[control], q[target])
      else:
        kernel.swap(q[target], q[other])
      two_qubit_gate_count += 1

  return (kernel, single_gate_count + two_qubit_gate_count, single_gate_count,
          two_qubit_gate_count)


def build_sample_full_register_kernel(cudaq: Any,
                                      qubits: int) -> tuple[Any, int]:
  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  gate_count = 0

  for index in range(qubits):
    theta = 0.061 + 0.001 * index
    kernel.ry(theta, q[index])
    kernel.rz(-0.5 * theta, q[index])
    gate_count += 2
  kernel.mz(q)
  return kernel, gate_count


def build_sample_basis_kernel(cudaq: Any, qubits: int) -> tuple[Any, int, str]:
  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  kernel.mz(q)
  return kernel, 0, "0" * qubits


def partial_register_measurement_indices(qubits: int) -> list[int]:
  return list(range(0, qubits, 2))


def build_sample_partial_register_kernel(cudaq: Any,
                                         qubits: int) -> tuple[Any, int, int]:
  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  gate_count = 0

  for index in range(qubits):
    theta = 0.073 + 0.0013 * index
    kernel.ry(theta, q[index])
    kernel.rz(-0.5 * theta, q[index])
    gate_count += 2

  measured_qubits = partial_register_measurement_indices(qubits)
  for index in measured_qubits:
    kernel.mz(q[index])
  return kernel, gate_count, len(measured_qubits)


def build_sample_ghz_kernel(cudaq: Any, qubits: int) -> tuple[Any, int]:
  kernel = cudaq.make_kernel()
  q = kernel.qalloc(qubits)
  kernel.h(q[0])
  gate_count = 1
  for index in range(qubits - 1):
    kernel.cx(q[index], q[index + 1])
    gate_count += 1
  kernel.mz(q)
  return kernel, gate_count


def timed_repeats(action: Callable[[], Any], repeats: int) -> list[float]:
  timings: list[float] = []
  for _ in range(repeats):
    start = time.perf_counter()
    action()
    timings.append(time.perf_counter() - start)
  return timings


def summarize_timings(timings: list[float]) -> dict[str, float]:
  return {
      "elapsed_seconds_min": min(timings),
      "elapsed_seconds_median": statistics.median(timings),
      "elapsed_seconds_max": max(timings),
  }


def result_template(target: str, case: str, qubits: int, shots: int,
                    repeats: int) -> dict[str, Any]:
  return {
      "target": target,
      "case": case,
      "qubits": qubits,
      "shots": shots,
      "repeats": repeats,
      "estimated_state_bytes": estimated_state_bytes(qubits),
      "status": "planned",
      "metrics": metal_path_metrics(target, case),
  }


def run_case(cudaq: Any, target: str, case: str, qubits: int, shots: int,
             repeats: int, warmups: int, layers: int) -> dict[str, Any]:
  row = result_template(target, case, qubits, shots, repeats)
  row["status"] = "ok"
  row["warmups"] = warmups

  try:
    cudaq.reset_target()
    cudaq.set_target(target)

    if hasattr(cudaq, "set_random_seed"):
      cudaq.set_random_seed(13)

    if case == "gate-state":
      kernel, gate_count = build_gate_state_kernel(cudaq, qubits, layers)
      action = lambda: cudaq.get_state(kernel)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "layers": layers,
          "end_to_end_gate_state_throughput_per_second": gate_count / median
          if median > 0 else None,
      })
    elif case == "single-qubit-state":
      kernel, gate_count = build_single_qubit_state_kernel(cudaq, qubits,
                                                           layers)
      action = lambda: cudaq.get_state(kernel)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "layers": layers,
          "single_qubit_gate_state_throughput_per_second": gate_count / median
          if median > 0 else None,
      })
    elif case in SINGLE_GATE_STATE_CASES:
      gate_name, count_key, throughput_key = SINGLE_GATE_STATE_CASES[case]
      kernel, gate_count, state_prep_gate_count, single_gate_count = (
          build_single_gate_state_kernel(cudaq, qubits, layers, gate_name))
      action = lambda: cudaq.get_state(kernel)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "state_prep_gate_count": state_prep_gate_count,
          count_key: single_gate_count,
          "layers": layers,
          throughput_key: single_gate_count / median if median > 0 else None,
      })
    elif case == "controlled-state":
      kernel, gate_count = build_controlled_state_kernel(cudaq, qubits, layers)
      action = lambda: cudaq.get_state(kernel)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "layers": layers,
          "controlled_gate_state_throughput_per_second": gate_count / median
          if median > 0 else None,
      })
    elif case in CONTROLLED_GATE_STATE_CASES:
      gate_name, count_key, throughput_key = CONTROLLED_GATE_STATE_CASES[case]
      kernel, gate_count, state_prep_gate_count, controlled_gate_count = (
          build_controlled_gate_state_kernel(cudaq, qubits, layers, gate_name))
      action = lambda: cudaq.get_state(kernel)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "state_prep_gate_count": state_prep_gate_count,
          count_key: controlled_gate_count,
          "layers": layers,
          throughput_key: controlled_gate_count / median
          if median > 0 else None,
      })
    elif case == "cz-state":
      kernel, gate_count, state_prep_gate_count, cz_gate_count = (
          build_cz_state_kernel(cudaq, qubits, layers))
      action = lambda: cudaq.get_state(kernel)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "state_prep_gate_count": state_prep_gate_count,
          "cz_gate_count": cz_gate_count,
          "layers": layers,
          "cz_gate_state_throughput_per_second": cz_gate_count / median
          if median > 0 else None,
      })
    elif case == "two-qubit-state":
      kernel, gate_count = build_two_qubit_state_kernel(cudaq, qubits, layers)
      action = lambda: cudaq.get_state(kernel)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "layers": layers,
          "two_qubit_gate_state_throughput_per_second": gate_count / median
          if median > 0 else None,
      })
    elif case == "three-qubit-state":
      kernel, gate_count, state_prep_gate_count, three_qubit_gate_count = (
          build_three_qubit_state_kernel(cudaq, qubits, layers))
      action = lambda: cudaq.get_state(kernel)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "state_prep_gate_count": state_prep_gate_count,
          "three_qubit_gate_count": three_qubit_gate_count,
          "layers": layers,
          "three_qubit_gate_state_throughput_per_second":
              three_qubit_gate_count / median if median > 0 else None,
      })
    elif case == "qft-like-state":
      (kernel, gate_count, state_prep_gate_count, h_gate_count, crz_gate_count,
       swap_gate_count) = build_qft_like_state_kernel(cudaq, qubits, layers)
      qft_like_gate_count = h_gate_count + crz_gate_count + swap_gate_count
      action = lambda: cudaq.get_state(kernel)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "state_prep_gate_count": state_prep_gate_count,
          "qft_h_gate_count": h_gate_count,
          "qft_crz_gate_count": crz_gate_count,
          "qft_swap_gate_count": swap_gate_count,
          "qft_like_gate_count": qft_like_gate_count,
          "layers": layers,
          "qft_like_state_throughput_per_second": qft_like_gate_count / median
          if median > 0 else None,
      })
    elif case == "seeded-clifford-state":
      kernel, gate_count, single_gate_count, two_qubit_gate_count = (
          build_seeded_clifford_state_kernel(cudaq, qubits, layers))
      action = lambda: cudaq.get_state(kernel)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "seeded_clifford_single_gate_count": single_gate_count,
          "seeded_clifford_two_qubit_gate_count": two_qubit_gate_count,
          "seeded_clifford_gate_count": gate_count,
          "seeded_clifford_seed": SEEDED_CLIFFORD_SEED,
          "layers": layers,
          "seeded_clifford_state_throughput_per_second": gate_count / median
          if median > 0 else None,
      })
    elif case == "sample-ghz":
      kernel, gate_count = build_sample_ghz_kernel(cudaq, qubits)
      action = lambda: cudaq.sample(kernel, shots_count=shots)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "sample_latency_seconds_per_shot": median / shots,
          "sample_throughput_shots_per_second": shots / median
          if median > 0 else None,
      })
    elif case == "sample-basis":
      kernel, gate_count, basis_state = build_sample_basis_kernel(cudaq, qubits)
      action = lambda: cudaq.sample(kernel, shots_count=shots)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "deterministic_outcome_count": 1,
          "basis_state": basis_state,
          "sample_latency_seconds_per_shot": median / shots,
          "sample_throughput_shots_per_second": shots / median
          if median > 0 else None,
      })
    elif case == "sample-full-register":
      kernel, gate_count = build_sample_full_register_kernel(cudaq, qubits)
      action = lambda: cudaq.sample(kernel, shots_count=shots)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "sample_latency_seconds_per_shot": median / shots,
          "sample_throughput_shots_per_second": shots / median
          if median > 0 else None,
      })
    elif case == "sample-partial-register":
      kernel, gate_count, measured_qubit_count = (
          build_sample_partial_register_kernel(cudaq, qubits))
      action = lambda: cudaq.sample(kernel, shots_count=shots)
      for _ in range(warmups):
        action()
      timings = timed_repeats(action, repeats)
      metrics = summarize_timings(timings)
      median = metrics["elapsed_seconds_median"]
      metrics.update({
          "gate_count": gate_count,
          "measured_qubit_count": measured_qubit_count,
          "marginal_outcome_count": 1 << measured_qubit_count,
          "sample_latency_seconds_per_shot": median / shots,
          "sample_throughput_shots_per_second": shots / median
          if median > 0 else None,
      })
    else:
      raise ValueError(f"unsupported benchmark case: {case}")

    metrics.update(metal_path_metrics(target, case))
    metrics["process_max_rss_bytes_cumulative"] = process_max_rss_bytes()
    row["metrics"] = metrics
  except Exception as exc:
    row["status"] = "error"
    row["error"] = f"{type(exc).__name__}: {exc}"
  finally:
    try:
      cudaq.reset_target()
    except Exception:
      pass

  return row


def run_isolated_case(args: argparse.Namespace, target: str, case: str,
                      qubits: int, shots: int | None = None) -> dict[str, Any]:
  shots = args.shots if shots is None else shots

  def isolated_process_payload() -> dict[str, Any]:
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }

  def isolated_error(message: str) -> dict[str, Any]:
    row = result_template(target, case, qubits, shots, args.repeats)
    row["status"] = "error"
    row["error"] = message
    row["isolated_process"] = isolated_process_payload()
    return row

  with tempfile.TemporaryDirectory(prefix="mklq-bench-row-") as tmpdir:
    output = Path(tmpdir) / "row.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--targets",
        target,
        "--cases",
        case,
        "--qubits",
        str(qubits),
        "--shots",
        str(shots),
        "--repeats",
        str(args.repeats),
        "--warmups",
        str(args.warmups),
        "--layers",
        str(args.layers),
        "--allow-errors",
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)

    if output.exists():
      try:
        child_report = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(child_report, dict):
          return isolated_error(
              "invalid isolated benchmark JSON: expected object report")
        rows = child_report.get("results")
        if not isinstance(rows, list) or not rows:
          return isolated_error(
              "invalid isolated benchmark JSON: missing results[0]")
        row = rows[0]
        if not isinstance(row, dict):
          return isolated_error(
              "invalid isolated benchmark JSON: results[0] is not an object")
      except Exception as exc:
        return isolated_error(
            f"invalid isolated benchmark JSON: {type(exc).__name__}: {exc}")

      row["isolated_process"] = {
          **isolated_process_payload(),
          "runtime": child_report.get("runtime"),
      }
      return row

    return isolated_error(
        "isolated benchmark subprocess did not produce JSON output"
        f" (returncode={completed.returncode}): {completed.stderr.strip()}")


def normalized_shot_counts(args: argparse.Namespace) -> list[int]:
  shot_counts = getattr(args, "shot_counts", None)
  if shot_counts is None:
    shot_counts = [args.shots]
  if not shot_counts:
    raise ValueError("expected at least one shot count")
  args.shot_counts = list(shot_counts)
  args.shots = args.shot_counts[0]
  return args.shot_counts


def build_report(args: argparse.Namespace) -> dict[str, Any]:
  shot_counts = normalized_shot_counts(args)
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
          "targets": args.targets,
          "cases": args.cases,
          "qubits": args.qubits,
          "shots": args.shots,
          "shot_counts": shot_counts,
          "repeats": args.repeats,
          "warmups": args.warmups,
          "layers": args.layers,
          "dry_run": args.dry_run,
          "isolate_rows": args.isolate_rows,
          "command": sys.argv,
      },
      "target_notes": TARGET_NOTES,
      "results": [],
  }

  if args.dry_run:
    for target in args.targets:
      for case in args.cases:
        for qubits in args.qubits:
          for shots in shot_counts:
            report["results"].append(
              result_template(target, case, qubits, shots, args.repeats))
    return report

  if args.isolate_rows:
    for target in args.targets:
      for case in args.cases:
        for qubits in args.qubits:
          for shots in shot_counts:
            report["results"].append(
                run_isolated_case(args, target, case, qubits, shots))
    return report

  import cudaq  # Delayed import keeps --dry-run independent of CUDA-Q setup.
  report["runtime"] = runtime_snapshot(cudaq)

  for target in args.targets:
    for case in args.cases:
      for qubits in args.qubits:
        for shots in shot_counts:
          report["results"].append(
              run_case(cudaq, target, case, qubits, shots, args.repeats,
                       args.warmups, args.layers))
  return report


def write_report(report: dict[str, Any], output: str | None) -> None:
  payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if output:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
  else:
    print(payload, end="")


def positive_int(value: str) -> int:
  try:
    parsed = int(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f"invalid integer '{value}'") from exc
  if parsed < 1:
    raise argparse.ArgumentTypeError(f"expected positive integer, got {parsed}")
  return parsed


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Benchmark qpp-cpu, mklq-cpu, and mklq-metal targets.")
  parser.add_argument("--targets",
                      type=parse_csv,
                      default=default_targets_for_platform(),
                      help="Comma-separated target list.")
  parser.add_argument("--cases",
                      type=lambda value: validate_cases(parse_csv(value)),
                      default=list(DEFAULT_CASES),
                      help=("Comma-separated cases: gate-state,sample-basis,"
                            "sample-ghz,sample-full-register,"
                            "sample-partial-register,single-qubit-state,"
                            "h-state,y-state,rx-state,ry-state,rz-state,"
                            "controlled-state,ch-state,cy-state,crx-state,cry-state,crz-state,"
                            "cz-state,two-qubit-state,three-qubit-state,"
                            "qft-like-state,"
                            "seeded-clifford-state."))
  parser.add_argument("--qubits",
                      type=parse_qubits,
                      default=list(DEFAULT_QUBITS),
                      help="Comma-separated qubit counts.")
  parser.add_argument("--shots",
                      type=positive_int,
                      default=1024,
                      help="Shot count for sampling benchmarks.")
  parser.add_argument("--shot-counts",
                      type=parse_shot_counts,
                      default=None,
                      help=("Comma-separated shot counts. Overrides --shots "
                            "and expands benchmark rows across each count."))
  parser.add_argument("--repeats",
                      type=positive_int,
                      default=3,
                      help="Measured repeats per target/case/qubit row.")
  parser.add_argument("--warmups",
                      type=positive_int,
                      default=1,
                      help="Warmup executions before measured repeats.")
  parser.add_argument("--layers",
                      type=positive_int,
                      default=16,
                      help=("Layer count for gate-state and "
                            "single-qubit-state/h-state/rx-state/ry-state/"
                            "rz-state/y-state/controlled-state/ch-state/cy-state/crx-state/cry-state/"
                            "crz-state/cz-state/two-qubit-state/"
                            "three-qubit-state/"
                            "qft-like-state/seeded-clifford-state "
                            "benchmarks."))
  parser.add_argument("--output",
                      help="JSON output path. Defaults to stdout.")
  parser.add_argument("--dry-run",
                      action="store_true",
                      help="Write the benchmark plan without importing CUDA-Q.")
  parser.add_argument("--isolate-rows",
                      action="store_true",
                      help="Run each measured row in a fresh Python process.")
  parser.add_argument("--allow-errors",
                      action="store_true",
                      help="Return 0 even when one or more benchmark rows fail.")
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
