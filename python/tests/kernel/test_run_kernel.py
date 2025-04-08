# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np
from typing import Callable, List

import cudaq


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_simple_sampling_ghz():
    @cudaq.kernel
    def simple(numQubits: int) -> int:
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits - 1)):
            x.ctrl(qubit, qubits[i + 1])
        result = 0
        for i in range(numQubits):
            if mz(qubits[i]):
                result += 1
        return result

    shots = 100
    qubitCount = 4
    results = cudaq.run(simple, qubitCount, shots_count=shots)
    print(results)
    assert len(results) == shots
    for result in results:
        assert result == 0 or result == qubitCount # 00..0 or 1...11


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])