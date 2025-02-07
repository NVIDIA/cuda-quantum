# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_simple():

    n_qubits = 4
    n_rounds = 10

    explicit_kernel = cudaq.make_kernel()
    q = explicit_kernel.qalloc(n_qubits)

    for round in range(n_rounds):
        explicit_kernel.h(q[0])

    for i in range(1, n_qubits):
        explicit_kernel.cx(q[i - 1], q[i])

    explicit_kernel.mz(q)

    for i in range(n_qubits):
        explicit_kernel.reset(q[i])

    counts = cudaq.sample(explicit_kernel, explicit_measurements=True)
    # With 1000 shots of multiple rounds, we need to see different shot measurements.
    assert counts["0000"] != 1000
