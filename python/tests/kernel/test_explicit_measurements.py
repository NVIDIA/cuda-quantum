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
    counts.dump()

    # With 1000 shots of multiple rounds, we need to see different shot measurements.
    assert len(counts) > 1

    seq = counts.get_sequential_data()
    assert len(seq) == 1000
    assert len(seq[0]) == n_qubits * n_rounds


@pytest.mark.parametrize("target", [
    'density-matrix-cpu', 'nvidia', 'nvidia-fp64', 'nvidia-mqpu',
    'nvidia-mqpu-fp64', 'nvidia-mqpu-mps', 'qpp-cpu', 'stim', 'tensornet',
    'tensornet-mps'
])
def test_simulators(target):

    def can_set_target(name):
        target_installed = True
        try:
            cudaq.set_target(name)
        except RuntimeError:
            target_installed = False
        return target_installed

    if can_set_target(target):
        test_simple()
    else:
        pytest.skip("target not available")

    cudaq.reset_target()
