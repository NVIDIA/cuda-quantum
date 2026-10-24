# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""cudaq.control over kernels with mid-circuit measurement.

A measured-uncomputation block (allocate a scratch qubit, compute an AND into
it, use it, erase it with an X- or Y-basis measurement and outcome-conditioned
feedforward) is a unitary channel, so its controlled form must be trajectory
independent and equal the hand-built controlled reference. Shapes whose
controlled form would depend on the measurement outcome -- a Z-basis
measurement, a measured qubit the kernel did not allocate -- are rejected.
"""

import numpy as np
import pytest

import cudaq

RUNS = 12
THETA = 0.7
HALF_PI = 1.5707963267948966


@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()
    cudaq.__clearKernelRegistries()


def _trajectories(kernel, *args):
    states = []
    for seed in range(RUNS):
        cudaq.set_random_seed(1 + seed)
        states.append(np.array(cudaq.get_state(kernel, *args)))
    return states


def _distance(a, b):
    overlap = np.vdot(a, b)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-12 else 1.0
    return float(np.linalg.norm(b - phase * a))


# ---------------------------------------------------------------------------
# Payload gadget: compute anc = q0 AND q1, apply Z to it, erase in the X basis.
# The block equals CZ(q0, q1), so its controlled form is CCZ(c, q0, q1).
# ---------------------------------------------------------------------------


@cudaq.kernel
def mbu_payload(q: cudaq.qview):
    anc = cudaq.qubit()
    x.ctrl(q[0], q[1], anc)
    z(anc)
    fired = mx(anc)
    if fired:
        z.ctrl(q[0], q[1])
        x(anc)


@cudaq.kernel
def controlled_payload(theta: float):
    c = cudaq.qvector(1)
    q = cudaq.qvector(2)
    ry(theta, c[0])
    h(q[0])
    h(q[1])
    cudaq.control(mbu_payload, c[0], q)


@cudaq.kernel
def reference_payload(theta: float):
    c = cudaq.qvector(1)
    q = cudaq.qvector(2)
    ry(theta, c[0])
    h(q[0])
    h(q[1])
    z.ctrl(c[0], q[0], q[1])


@pytest.mark.parametrize("theta", [0.0, np.pi, THETA])
def test_controlled_measured_uncomputation_is_exact(theta):
    # The reference allocates no scratch qubit; compare the data+control
    # marginal, which the padded scratch qubit (restored to |0>) leaves intact.
    expected = np.array(cudaq.get_state(reference_payload, theta))
    for state in _trajectories(controlled_payload, theta):
        reduced = state.reshape(2,
                                -1)[0]  # scratch qubit (last allocated) is |0>
        assert abs(np.linalg.norm(reduced) - 1.0) < 1e-10
        assert _distance(expected, reduced) < 1e-10


# ---------------------------------------------------------------------------
# Identity-channel oracle: compute + measured erase with no payload is the
# identity on q, so its controlled form must equal the controlled unitary
# compute-uncompute (two Toffolis) on every trajectory, at definite and
# superposed control. On an unfixed build the superposed-control cases and the
# `mx` c=|0> case fail with distance ~1.1-1.4 (each trajectory collapses onto
# one control branch; the suppressed reset leaves the scratch qubit in |m>).
# The Y-basis erase uses a controlled-S^-1 fix-up.
# ---------------------------------------------------------------------------


@cudaq.kernel
def stack_mx(q: cudaq.qview):
    anc = cudaq.qvector(1)
    x.ctrl(q[0], q[1], anc[0])
    fired = mx(anc[0])
    if fired:
        z.ctrl(q[0], q[1])
        x(anc[0])


@cudaq.kernel
def stack_my(q: cudaq.qview):
    anc = cudaq.qvector(1)
    x.ctrl(q[0], q[1], anc[0])
    fired = my(anc[0])
    if fired:
        r1.ctrl(-HALF_PI, q[0], q[1])
        x(anc[0])
    else:
        r1.ctrl(HALF_PI, q[0], q[1])


@cudaq.kernel
def stack_unitary(q: cudaq.qview):
    anc = cudaq.qvector(1)
    x.ctrl(q[0], q[1], anc[0])
    x.ctrl(q[0], q[1], anc[0])


@cudaq.kernel
def stack_driver(which: int, mode: int):
    # mode: 0 -> c=|0>, 1 -> c=|1>, 2 -> superposed ry(THETA)
    c = cudaq.qvector(1)
    q = cudaq.qvector(2)
    if mode == 1:
        x(c[0])
    if mode == 2:
        ry(THETA, c[0])
    h(q[0])
    h(q[1])
    if which == 0:
        cudaq.control(stack_mx, c[0], q)
    elif which == 1:
        cudaq.control(stack_my, c[0], q)
    else:
        cudaq.control(stack_unitary, c[0], q)


@pytest.mark.parametrize("which", [0, 1], ids=["mx", "my"])
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["c=0", "c=1", "c=ry"])
def test_controlled_identity_stack_matches_unitary_oracle(which, mode):
    cudaq.set_random_seed(1)
    oracle = np.array(cudaq.get_state(stack_driver, 2, mode))
    for state in _trajectories(stack_driver, which, mode):
        assert _distance(oracle, state) < 1e-10


# ---------------------------------------------------------------------------
# Rejections. The compile-time diagnostic names the reason; at the Python
# level a failed specialization surfaces as the runtime's generic error.
# ---------------------------------------------------------------------------


def _assert_rejected(kernel):
    with pytest.raises(RuntimeError, match="could not autogenerate"):
        cudaq.get_state(kernel)


def test_z_basis_measurement_is_rejected():
    """`h` then `mz` is a Z-basis measurement under control: the controlled
    `h` leaves a |0> scratch qubit reading 0 with the control off and a coin
    with it on, so the outcome reveals the control."""

    @cudaq.kernel
    def stack_hmz(q: cudaq.qview):
        anc = cudaq.qubit()
        x.ctrl(q[0], q[1], anc)
        h(anc)
        fired = mz(anc)
        if fired:
            z.ctrl(q[0], q[1])
            x(anc)

    @cudaq.kernel
    def controlled():
        c = cudaq.qvector(1)
        q = cudaq.qvector(2)
        ry(THETA, c[0])
        h(q[0])
        h(q[1])
        cudaq.control(stack_hmz, c[0], q)

    _assert_rejected(controlled)


def test_measurement_of_argument_qubit_is_rejected():
    """The erase gadget on a caller-owned qubit: its state when the control is
    off is unknown to the kernel, so the uncontrolled measurement would
    disturb it."""

    @cudaq.kernel
    def erase(q: cudaq.qview):
        fired = mx(q[2])
        if fired:
            z.ctrl(q[0], q[1])
            x(q[2])

    @cudaq.kernel
    def controlled():
        c = cudaq.qvector(1)
        q = cudaq.qvector(3)
        ry(THETA, c[0])
        h(q[0])
        h(q[1])
        x.ctrl(q[0], q[1], q[2])
        cudaq.control(erase, c[0], q)

    _assert_rejected(controlled)


def test_measured_argument_with_user_state_is_rejected():
    """A caller-owned qubit carrying user state (|+>) that the kernel flips,
    measures and resets: with the control off the state must survive, but an
    uncontrolled measurement would collapse it, so the kernel is rejected."""

    @cudaq.kernel
    def flip_measure_reset(q: cudaq.qview):
        x(q[0])
        fired = mx(q[0])
        if fired:
            x(q[0])

    @cudaq.kernel
    def controlled():
        c = cudaq.qvector(1)
        q = cudaq.qvector(1)
        h(q[0])
        cudaq.control(flip_measure_reset, c[0], q)

    _assert_rejected(controlled)


def test_adjoint_control_of_measuring_kernel_is_rejected():

    @cudaq.kernel
    def controlled():
        c = cudaq.qvector(1)
        q = cudaq.qvector(2)
        cudaq.control(stack_mx, c[0], q)
        cudaq.adjoint(stack_mx, q)

    _assert_rejected(controlled)
