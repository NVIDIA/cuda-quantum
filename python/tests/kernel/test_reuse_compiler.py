# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np

import os


def test_reuse(capfd, monkeypatch):
    """Test that we can build a very simple kernel and sample it."""

    os.environ['CUDAQ_LOG_LEVEL'] = 'info'
    import cudaq

    @cudaq.kernel
    def simple(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits - 1)):
            x.ctrl(qubit, qubits[i + 1])

    num_qubits = 5
    cudaq.sample(simple, num_qubits, shots_count=1)
    captured = capfd.readouterr()
    assert "Using cached JIT engine" not in captured.out
    cudaq.sample(simple, num_qubits, shots_count=1)
    captured = capfd.readouterr()
    assert "Using cached JIT engine" not in captured.out
    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        cudaq.sample(simple, num_qubits, shots_count=1)
        captured = capfd.readouterr()
        assert "Using cached JIT engine" not in captured.out
        cudaq.sample(simple, num_qubits, shots_count=1)
        captured = capfd.readouterr()
        assert "Using cached JIT engine" in captured.out
    cudaq.sample(simple, num_qubits, shots_count=1)
    captured = capfd.readouterr()
    assert "Using cached JIT engine" not in captured.out
