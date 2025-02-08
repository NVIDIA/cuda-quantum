# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest
import os


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def test_simple_kernel():

    @cudaq.kernel
    def explicit_kernel(n_qubits: int, n_rounds: int):
        q = cudaq.qvector(n_qubits)
        for round in range(n_rounds):
            h(q[0])
            for i in range(1, n_qubits):
                x.ctrl(q[i - 1], q[i])
            mz(q)
            reset(q)

    counts = cudaq.sample(explicit_kernel, 4, 10, explicit_measurements=True)
    # counts.dump()

    # With 1000 shots of multiple rounds, we need to see different shot measurements.
    assert len(counts) > 1

    seq = counts.get_sequential_data()
    assert len(seq) == 1000
    assert len(seq[0]) == 40


def test_simple_builder():

    n_qubits = 2
    n_rounds = 20

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
    # counts.dump()

    # With 1000 shots of multiple rounds, we need to see different shot measurements.
    assert len(counts) > 1

    seq = counts.get_sequential_data()
    assert len(seq) == 1000
    assert len(seq[0]) == n_qubits * n_rounds


@pytest.mark.parametrize("target", [
    "density-matrix-cpu", "nvidia", "nvidia-mqpu-mps", "qpp-cpu", "stim",
    "tensornet", "tensornet-mps"
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
        test_simple_kernel()
    else:
        pytest.skip("target not available")

    cudaq.reset_target()


@pytest.mark.parametrize("target, env_var",
                         [("anyon", ""), ("braket", ""),
                          ("infleqtion", "SUPERSTAQ_API_KEY"),
                          ("ionq", "IONQ_API_KEY"), ("quantinuum", ""),
                          ("quera", "")])
def test_unsupported_targets(target, env_var):
    if env_var:
        os.environ[env_var] = "foobar"

    cudaq.set_target(target)

    with pytest.raises(RuntimeError) as e:
        test_simple_kernel()
    assert "not supported on this target" in repr(
        e)
    os.environ.pop(env_var, None)
    cudaq.reset_target()


def test_error_cases():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        h(q[0])
        if mz(q[0]):
            x(q[1])

    # This is allowed
    cudaq.sample(kernel)

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel, explicit_measurements=True)
    assert "not supported on kernel with conditional logic on a measurement result" in repr(
        e)
