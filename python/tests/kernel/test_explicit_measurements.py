# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest
import os
import numpy as np

skipIfBraketNotInstalled = pytest.mark.skipif(
    not (cudaq.has_target("braket")),
    reason='Could not find `braket` in installation')


@pytest.fixture(autouse=True)
def do_something():
    cudaq.reset_target()
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def test_simple_kernel():

    num_shots = 50

    @cudaq.kernel
    def explicit_kernel(n_qubits: int, n_rounds: int):
        q = cudaq.qvector(n_qubits)
        for round in range(n_rounds):
            h(q[0])
            for i in range(1, n_qubits):
                x.ctrl(q[i - 1], q[i])
            mz(q)
            reset(q)

    counts = cudaq.sample(explicit_kernel,
                          4,
                          10,
                          explicit_measurements=True,
                          shots_count=num_shots)
    # counts.dump()

    # With many shots of multiple rounds, we need to see different shot measurements.
    assert len(counts) > 1

    seq = counts.get_sequential_data()
    assert len(seq) == num_shots
    assert len(seq[0]) == 40


def test_simple_builder():

    num_shots = 50
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

    counts = cudaq.sample(explicit_kernel,
                          explicit_measurements=True,
                          shots_count=num_shots)
    # counts.dump()

    # With many shots of multiple rounds, we need to see different shot measurements.
    assert len(counts) > 1

    seq = counts.get_sequential_data()
    assert len(seq) == num_shots
    assert len(seq[0]) == n_qubits * n_rounds


def test_sample_async():

    num_shots = 100

    @cudaq.kernel
    def kernel(theta: float, phi: float):
        qubits = cudaq.qvector(2)
        for round in range(10):
            rx(theta, qubits[0])
            ry(phi, qubits[0])
            x.ctrl(qubits[0], qubits[1])
            mz(qubits)

    future = cudaq.sample_async(kernel,
                                np.pi,
                                np.pi / 2.,
                                shots_count=num_shots,
                                explicit_measurements=True)
    counts = future.get()
    # Without explicit measurements, and only one round, we expect result like `{ 00:45 11:55 }`
    assert len(counts) > 2

    seq = counts.get_sequential_data()
    assert len(seq) == num_shots
    assert len(seq[0]) == 20  # num qubits * num_rounds


def test_named_measurement():
    """ Test for while using "explicit measurements" mode, the sample result 
        will not be saved to a mid-circuit measurement register. """

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        x(q[0])
        val = mz(q[1])

    counts = cudaq.sample(kernel)
    assert '__global__' in counts.register_names
    assert 'val' in counts.register_names

    counts = cudaq.sample(kernel, explicit_measurements=True)
    assert '__global__' in counts.register_names
    assert 'val' not in counts.register_names


def test_measurement_order():
    """ Test for if the "explicit measurements" option is enabled, the global 
        register contains the concatenated measurements in the order they were
        executed in the kernel. """

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        x(q[0])
        mz(q[1])
        mz(q[0])
        mz(q[2])

    counts = cudaq.sample(kernel)
    assert counts["100"] == 1000

    counts = cudaq.sample(kernel, explicit_measurements=True)
    assert counts["010"] == 1000

    @cudaq.kernel
    def kernel_with_loop():
        q = cudaq.qvector(3)
        for _ in range(3):
            x(q[0])
            mz(q[1])
            mz(q[0])
            mz(q[2])
            reset(q)

    counts = cudaq.sample(kernel_with_loop)
    assert counts["000"] == 1000  # due to reset

    counts = cudaq.sample(kernel_with_loop, explicit_measurements=True)
    assert counts["010010010"] == 1000


def test_multiple_measurements():

    @cudaq.kernel
    def measure_twice():
        q = cudaq.qubit()
        x(q)
        mz(q)
        mz(q)

    counts = cudaq.sample(measure_twice)
    assert counts["1"] == 1000

    counts = cudaq.sample(measure_twice, explicit_measurements=True)
    assert counts["11"] == 1000


def test_no_measurements():
    """ Test for kernels executed in "explicit measurements" mode must contain measurements. """

    @cudaq.kernel
    def no_measure_ops():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(no_measure_ops, explicit_measurements=True)
    assert "not supported on a kernel without any measurement" in repr(e)


def test_mx_my():

    @cudaq.kernel
    def my_kernel():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        mx(q[0])
        my(q[1])

    counts = cudaq.sample(my_kernel)
    assert len(counts) == 2

    counts = cudaq.sample(my_kernel, explicit_measurements=True)
    assert len(counts) == 4


# NOTE: Ref - https://github.com/NVIDIA/cuda-quantum/issues/1925
@pytest.mark.parametrize("target",
                         ["density-matrix-cpu", "nvidia", "qpp-cpu", "stim"])
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
                         [("anyon", ""), ("infleqtion", "SUPERSTAQ_API_KEY"),
                          ("ionq", "IONQ_API_KEY"), ("quantinuum", "")])
def test_unsupported_targets(target, env_var):
    if env_var:
        os.environ[env_var] = "foobar"

    cudaq.set_target(target)

    with pytest.raises(RuntimeError) as e:
        test_simple_kernel()
    assert "not supported on this target" in repr(e)
    os.environ.pop(env_var, None)
    cudaq.reset_target()


@skipIfBraketNotInstalled
@pytest.mark.parametrize("target", ["braket", "quera"])
def test_unsupported_targets2(target):
    cudaq.set_target(target)
    with pytest.raises(RuntimeError) as e:
        test_simple_kernel()
    assert "not supported on this target" in repr(e)
    cudaq.reset_target()


def test_error_cases():
    """ Test for throw error if user attempts to use a measurement result in 
        conditional logic. """

    @cudaq.kernel
    def kernel_with_conditional_on_measure():
        q = cudaq.qvector(2)
        h(q[0])
        if mz(q[0]):
            x(q[1])

    # This is allowed
    cudaq.sample(kernel_with_conditional_on_measure).dump()

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel_with_conditional_on_measure,
                     explicit_measurements=True)
    assert "not supported on kernel with conditional logic on a measurement result" in repr(
        e)

    ## NOTE: The following does not fail.
    ## Needs inlining of the function calls.
    # @cudaq.kernel
    # def measure(q: cudaq.qubit) -> bool:
    #     return mz(q)

    # @cudaq.kernel
    # def kernel_with_conditional_on_function():
    #     q = cudaq.qvector(2)
    #     h(q[0])
    #     if measure(q[0]):
    #         x(q[1])

    # with pytest.raises(RuntimeError) as e:
    #     cudaq.sample(kernel_with_conditional_on_function,
    #                  explicit_measurements=True)
    # assert "not supported on kernel with conditional logic on a measurement result" in repr(
    #     e)

    cudaq.__clearKernelRegistries()
