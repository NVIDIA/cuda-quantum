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
def reset_run_clear():
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
    """Named measurements preserve the named register."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        x(q[0])
        val = mz(q[1])

    for explicit in (None, True, False):
        counts = cudaq.sample(kernel, explicit_measurements=explicit)
        assert '__global__' in counts.register_names
        assert 'val' in counts.register_names
        assert counts["0"] == 1000
        assert counts.get_register_counts("val")["0"] == 1000


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
    assert counts["010"] == 1000

    counts = cudaq.sample(kernel, explicit_measurements=True)
    assert counts["010"] == 1000

    with pytest.raises(RuntimeError, match="explicit_measurements=false"):
        cudaq.sample(kernel, explicit_measurements=False)

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
    assert counts["010010010"] == 1000

    counts = cudaq.sample(kernel_with_loop, explicit_measurements=True)
    assert counts["010010010"] == 1000

    with pytest.raises(RuntimeError, match="explicit_measurements=false"):
        cudaq.sample(kernel_with_loop, explicit_measurements=False)


def test_multiple_measurements():

    @cudaq.kernel
    def measure_twice():
        q = cudaq.qubit()
        x(q)
        mz(q)
        mz(q)

    counts = cudaq.sample(measure_twice)
    assert counts["11"] == 1000

    counts = cudaq.sample(measure_twice, explicit_measurements=True)
    assert counts["11"] == 1000

    with pytest.raises(RuntimeError, match="explicit_measurements=false"):
        cudaq.sample(measure_twice, explicit_measurements=False)


def test_no_measurements():
    """No-measurement kernels use implicit final sampling."""

    @cudaq.kernel
    def no_measure_ops():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])

    for explicit in (None, True, False):
        counts = cudaq.sample(no_measure_ops, explicit_measurements=explicit)
        assert counts.get_total_shots() == 1000


def test_terminal_basis_measurements_allow_non_explicit():

    def check(kernel, expected=None):
        for explicit in (None, True, False):
            counts = cudaq.sample(kernel, explicit_measurements=explicit)
            assert counts.get_total_shots() == 1000
            if expected is not None:
                assert counts[expected] == 1000

    @cudaq.kernel
    def terminal_full_mz():
        q = cudaq.qvector(2)
        x(q[0])
        mz(q)

    check(terminal_full_mz, "10")

    @cudaq.kernel
    def terminal_mz_then_reset():
        q = cudaq.qvector(2)
        x(q[0])
        mz(q)
        reset(q)

    check(terminal_mz_then_reset, "10")

    @cudaq.kernel
    def terminal_mx():
        q = cudaq.qvector(2)
        h(q[0])
        x(q[1])
        h(q[1])
        mx(q)

    check(terminal_mx, "01")

    @cudaq.kernel
    def terminal_my():
        q = cudaq.qvector(2)
        my(q)

    check(terminal_my)


def test_default_measurement_order_issue_4153():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(4)
        x(q[2])
        mz(q[2])
        mz(q[0])
        mz(q[1])
        mz(q[3])

    counts = cudaq.sample(kernel)
    assert counts["1000"] == 1000


def test_mixed_basis_measurement_order_and_preservation():

    @cudaq.kernel
    def mixed_basis_kernel():
        q = cudaq.qvector(9)

        # Prepare a non-palindromic deterministic pattern over measured bits.
        # q0=0 (mz), q1=1 (mz), q2=1 (mx), q3=? (my), q4=0 (mz), q5=0 (mx),
        # q6=1 (mz) -> 011?001 in allocation order.
        x(q[1])
        x(q[2])
        h(q[2])
        h(q[5])
        x(q[6])

        # Mix measurement bases and execution order.
        mz(q[4])
        mx(q[2])
        my(q[3])
        mz(q[0])
        mx(q[5])
        mz(q[6])
        mz(q[1])

    counts = cudaq.sample(mixed_basis_kernel, shots_count=100)

    # Execution order was q4, q2, q3, q0, q5, q6, q1 => 01?0011.
    total_counts = 0
    for bits in counts:
        assert len(bits) == 7
        assert bits[0] == '0'
        assert bits[1] == '1'
        assert bits[3] == '0'
        assert bits[4] == '0'
        assert bits[5] == '1'
        assert bits[6] == '1'
        total_counts += counts[bits]

    assert total_counts == 100

    with pytest.raises(RuntimeError, match="explicit_measurements=false"):
        cudaq.sample(mixed_basis_kernel,
                     explicit_measurements=False,
                     shots_count=100)

    with pytest.raises(RuntimeError, match="explicit_measurements=false"):
        cudaq.sample_async(mixed_basis_kernel,
                           explicit_measurements=False,
                           shots_count=100).get()

    counts = cudaq.sample(mixed_basis_kernel,
                          explicit_measurements=True,
                          shots_count=100)

    # Execution order was q4, q2, q3, q0, q5, q6, q1 => 01?0011.
    total_counts = 0
    for bits in counts:
        assert len(bits) == 7
        assert bits[0] == '0'
        assert bits[1] == '1'
        assert bits[3] == '0'
        assert bits[4] == '0'
        assert bits[5] == '1'
        assert bits[6] == '1'
        total_counts += counts[bits]

    assert total_counts == 100


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

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel_with_conditional_on_measure,
                     explicit_measurements=True)
    assert "no longer support" in repr(e)

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
    # assert "no longer support" in repr(e)

    cudaq.__clearKernelRegistries()
