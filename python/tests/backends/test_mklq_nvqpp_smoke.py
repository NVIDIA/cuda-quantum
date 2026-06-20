# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from mklq_test_utils import mklq_targets_available


NVQPP_COMPILE_TIMEOUT_SECONDS = 90
NVQPP_RUN_TIMEOUT_SECONDS = 90

pytestmark = pytest.mark.skipif(not mklq_targets_available(),
                                reason="MKL-Q targets are not available")


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _nvqpp_path(repo_root):
    configured = os.environ.get("CUDAQ_NVQPP")
    if configured:
        return Path(configured)

    build_tree_nvqpp = repo_root / "build-python" / "bin" / "nvq++"
    if build_tree_nvqpp.exists():
        return build_tree_nvqpp

    discovered = shutil.which("nvq++")
    return Path(discovered) if discovered else build_tree_nvqpp


@pytest.mark.parametrize(("target", "marker"), [
    ("mklq-cpu", "mklq-target-marker:cpu"),
    ("mklq-metal", "mklq-target-marker:metal"),
])
def test_mklq_nvqpp_runtime_smoke_uses_requested_target(tmp_path, target,
                                                        marker):
    repo_root = _repo_root()
    nvqpp = _nvqpp_path(repo_root)
    source = (repo_root / "targettests" / "TargetConfig" /
              "mklq_runtime_smoke.cpp")

    if not nvqpp.exists():
        pytest.skip(f"build-tree nvq++ not found: {nvqpp}")
    if not source.exists():
        pytest.skip(f"MKL-Q runtime smoke source not found: {source}")

    workdir = tmp_path / target
    workdir.mkdir()
    executable = workdir / "smoke"

    compile_result = subprocess.run([
        str(nvqpp),
        "--target",
        target,
        str(source),
        "-o",
        str(executable),
    ],
                                    cwd=workdir,
                                    capture_output=True,
                                    text=True,
                                    timeout=NVQPP_COMPILE_TIMEOUT_SECONDS)
    assert compile_result.returncode == 0, (
        compile_result.stdout + compile_result.stderr)

    run_result = subprocess.run([str(executable)],
                               cwd=workdir,
                               capture_output=True,
                               text=True,
                               timeout=NVQPP_RUN_TIMEOUT_SECONDS)
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert marker in run_result.stdout
    assert "mklq-runtime-smoke-ok" in run_result.stdout
    assert "mklq-observe-smoke-ok" in run_result.stdout
    assert "mklq-mid-circuit-smoke-ok" in run_result.stdout

    other_marker = "mklq-target-marker:metal" if target == "mklq-cpu" else \
        "mklq-target-marker:cpu"
    assert other_marker not in run_result.stdout
