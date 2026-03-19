# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import json

import cudaq
import pytest


@pytest.fixture(autouse=True)
def reset_target():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


def _run_pipeline_logging_check(tmp_path, monkeypatch, *, target=None, **target_kwargs):
    log_path = tmp_path / "pipeline.jsonl"
    monkeypatch.setenv("CUDAQ_PIPELINE_LOG", str(log_path))

    if target is not None:
        if not cudaq.has_target(target):
            pytest.skip(f"target '{target}' not available")
        try:
            cudaq.set_target(target, **target_kwargs)
        except RuntimeError as err:
            pytest.skip(f"unable to configure target '{target}': {err}")

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.h(qubit)
    kernel.mz(qubit)

    try:
        counts = cudaq.sample(kernel, shots_count=16)
    except RuntimeError as err:
        if target == "remote-mqpu" and "Unable to find a TCP/IP port" in str(err):
            pytest.skip(str(err))
        if target == "iqm" and "Unable to get quantum architecture" in str(err):
            pytest.skip(str(err))
        raise

    assert counts.count("0") + counts.count("1") > 0
    assert log_path.exists(), f"pipeline log not created at {log_path}"

    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert any(entry.get("type") == "configured" for entry in entries)
    assert any(entry.get("type") == "executed" for entry in entries)


TARGET_CONFIGS = [
    pytest.param("anyon", {"emulate": True}, id="anyon"),
    pytest.param("braket", {"emulate": True}, id="braket"),
    pytest.param("infleqtion", {"emulate": True}, id="infleqtion"),
    pytest.param("ionq", {"emulate": True}, id="ionq"),
    pytest.param("iqm", {"emulate": True}, id="iqm"),
    pytest.param("oqc", {"emulate": True}, id="oqc"),
    pytest.param("qci", {"emulate": True}, id="qci"),
    pytest.param("quantinuum", {"emulate": True}, id="quantinuum"),
    pytest.param("quantum_machines", {"emulate": True}, id="quantum_machines"),
    pytest.param("remote-mqpu", {"auto_launch": "1"}, id="remote-simulator"),
    pytest.param("scaleway", {"emulate": True}, id="scaleway"),
]


def test_pipeline_logging_default_simulator(tmp_path, monkeypatch):
    _run_pipeline_logging_check(tmp_path, monkeypatch)


@pytest.mark.parametrize("target,target_kwargs", TARGET_CONFIGS)
def test_pipeline_logging_target(tmp_path, monkeypatch, target, target_kwargs):
    _run_pipeline_logging_check(tmp_path, monkeypatch, target=target, **target_kwargs)
