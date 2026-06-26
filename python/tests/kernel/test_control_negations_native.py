# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, pytest
import numpy as np
import cudaq

skip_if_no_gpu = pytest.mark.skipif(
    cudaq.num_available_gpus() == 0,
    reason="native negated-control path requires the `nvidia` target")


@pytest.fixture(autouse=True)
def reset_after():
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


# --- Open multi-controlled X (real amplitudes) ------------------------------
@cudaq.kernel
def negated_controls_x_native():
    c = cudaq.qvector(4)
    t = cudaq.qubit()
    h(c[0])
    h(c[1])
    h(c[2])
    h(c[3])
    x.ctrl(c[0], ~c[1], ~c[2], c[3], t)


@cudaq.kernel
def negated_controls_x_manual():
    c = cudaq.qvector(4)
    t = cudaq.qubit()
    h(c[0])
    h(c[1])
    h(c[2])
    h(c[3])
    # Explicit X conjugation of the negated controls (c1, c2).
    x(c[1])
    x(c[2])
    x.ctrl(c[0], c[1], c[2], c[3], t)
    x(c[1])
    x(c[2])


# --- Negated-control rotations / phase (complex amplitudes) -----------------
@cudaq.kernel
def negated_controls_rot_native():
    c = cudaq.qvector(3)
    t = cudaq.qubit()
    h(c[0])
    h(c[1])
    h(c[2])
    ry.ctrl(0.7, ~c[0], c[1], t)
    rz.ctrl(1.1, c[0], ~c[2], t)
    z.ctrl(~c[1], ~c[2], t)


@cudaq.kernel
def negated_controls_rot_manual():
    c = cudaq.qvector(3)
    t = cudaq.qubit()
    h(c[0])
    h(c[1])
    h(c[2])
    x(c[0])
    ry.ctrl(0.7, c[0], c[1], t)
    x(c[0])
    x(c[2])
    rz.ctrl(1.1, c[0], c[2], t)
    x(c[2])
    x(c[1])
    x(c[2])
    z.ctrl(c[1], c[2], t)
    x(c[1])
    x(c[2])


def _state(kernel):
    return np.array(cudaq.get_state(kernel), copy=True)


@skip_if_no_gpu
@pytest.mark.parametrize("native,manual", [
    (negated_controls_x_native, negated_controls_x_manual),
    (negated_controls_rot_native, negated_controls_rot_manual),
])
def test_native_negated_controls_match_x_expansion(native, manual):
    # On nvidia, the native negated-control path must produce the same state as an
    # explicit X conjugation of the same circuit, up to a global phase.
    cudaq.reset_target()
    cudaq.set_target("nvidia", option="fp64")
    sv_native = _state(native)
    sv_manual = _state(manual)
    cudaq.reset_target()
    overlap = abs(np.vdot(sv_manual, sv_native))
    assert np.isclose(overlap, 1.0, atol=1e-6), f"overlap={overlap}"


@skip_if_no_gpu
def test_native_negated_controls_correctness():
    # Deterministic truth-table check on the native path: c = |1001>, so the
    # positive controls (c0, c3) and negated controls (c1, c2) all fire and flip q.
    @cudaq.kernel
    def multi_control_negated():
        c = cudaq.qvector(4)
        q = cudaq.qubit()
        x(c[0], c[3])
        x.ctrl(c[0], ~c[1], ~c[2], c[3], q)

    cudaq.reset_target()
    cudaq.set_target("nvidia")
    counts = cudaq.sample(multi_control_negated)
    cudaq.reset_target()
    assert counts["10011"] == 1000


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
