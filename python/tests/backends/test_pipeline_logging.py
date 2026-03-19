# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import json

import cudaq
import pytest
from cudaq import spin


@pytest.fixture(autouse=True)
def reset_target():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


TARGET_CONFIGS = [
    # `run` is not currently supported on these backend targets.
    pytest.param("infleqtion", {"emulate": True}, False, id="openqasm"),
    pytest.param("quantinuum", {"emulate": True}, False, id="qir"),
    pytest.param("remote-mqpu", {"auto_launch": "1"},
                 True,
                 id="remote-simulator"),
    # Keep one local/default case that also exercises `run`.
    pytest.param("qpp-cpu", {}, True, id="default"),
]


@cudaq.kernel
def sample_kernel():
    qubit = cudaq.qubit()
    h(qubit)
    mz(qubit)


@cudaq.kernel
def run_kernel() -> int:
    qubit = cudaq.qubit()
    h(qubit)
    return mz(qubit)


@cudaq.kernel
def observe_kernel():
    qubit = cudaq.qubit()
    h(qubit)


@pytest.mark.parametrize("target,target_kwargs,supports_run", TARGET_CONFIGS)
def test_pipeline_logging_decorator(tmp_path, monkeypatch, target,
                                    target_kwargs, supports_run):
    log_path = tmp_path / "pipeline.jsonl"
    monkeypatch.setenv("CUDAQ_PIPELINE_LOG", str(log_path))

    if not cudaq.has_target(target):
        pytest.skip(f"target '{target}' not available")
    cudaq.set_target(target, **target_kwargs)

    try:
        cudaq.sample(sample_kernel, shots_count=1)
        cudaq.observe(observe_kernel, spin.z(0))
        if supports_run:
            cudaq.run(run_kernel, shots_count=1)
    except RuntimeError as err:
        raise

    assert log_path.exists()

    entries = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    assert any(entry.get("type") == "configured" for entry in entries)
    assert any(entry.get("type") == "executed" for entry in entries)


def test_pipeline_logging_builder_default(tmp_path, monkeypatch):
    log_path = tmp_path / "pipeline.jsonl"
    monkeypatch.setenv("CUDAQ_PIPELINE_LOG", str(log_path))

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.h(qubit)
    kernel.mz(qubit)

    cudaq.sample(kernel, shots_count=1)

    assert log_path.exists()

    entries = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    assert any(entry.get("type") == "configured" for entry in entries)
    assert any(entry.get("type") == "executed" for entry in entries)
