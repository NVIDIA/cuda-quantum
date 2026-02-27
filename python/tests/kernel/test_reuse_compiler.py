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

import cudaq


def test_reuse():
    """Test that we can reuse a compiled jit across launches"""

    @cudaq.kernel
    def simple(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        x(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits - 1)):
            x.ctrl(qubit, qubits[i + 1])

    res = cudaq.sample(simple, 2, shots_count=1)
    assert (res.count("11") == 1)
    res = cudaq.sample(simple, 3, shots_count=1)
    assert (res.count("111") == 1)
    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        res = cudaq.sample(simple, 4, shots_count=1)
        assert (res.count("1111") == 1)
        # Abuse the foot gun to make sure the cached kernel is rerun
        # (and therefore the number of qubits is the same)
        res = cudaq.sample(simple, 5, shots_count=1)
        assert (res.count("1111") == 1)
    res = cudaq.sample(simple, 6, shots_count=1)
    assert (res.count("111111") == 1)
