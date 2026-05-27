# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import re
import cudaq
import pytest


def _count_errors(dem_text: str) -> int:
    """Count ``error(<prob>) ...`` lines in the Stim ``.dem`` text."""
    return dem_text.count("error(")


def _max_target_index(dem_text: str, prefix: str) -> int:
    """Return ``max(idx)+1`` for ``<prefix><idx>`` references in @p dem_text.

    Detectors / logical observables appear as TARGETS in ``error(...)`` lines
    (e.g. ``D5``, ``L2``) rather than as standalone instructions in the
    Stim .dem format. The implicit count is ``max(idx) + 1``.
    """
    matches = re.findall(rf"(?:^|\s){prefix}(\d+)", dem_text)
    if not matches:
        return 0
    return max(int(m) for m in matches) + 1


def _summary(dem_text: str) -> dict:
    return {
        "errors": _count_errors(dem_text),
        "detectors": _max_target_index(dem_text, "D"),
        "observables": _max_target_index(dem_text, "L"),
    }


@pytest.fixture(autouse=True)
def reset_run_clear():
    cudaq.reset_target()
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def test_trivial_empty_dem():
    """Kernel without QEC declarations yields an empty DEM."""

    @cudaq.kernel
    def trivial():
        q = cudaq.qubit()
        h(q)
        mz(q)

    dem_text = cudaq.dem_from_kernel(trivial)
    assert _summary(dem_text) == {"errors": 0, "detectors": 0, "observables": 0}


def test_no_noise_positional_kernel_args():
    """Kernel arguments follow the kernel; noise_model is keyword-only."""

    @cudaq.kernel
    def kernel(n_rounds: int):
        q = cudaq.qubit()
        m = mz(q)
        for _ in range(n_rounds):
            m_new = mz(q)
            cudaq.detector(m_new, m)
            m = m_new
        cudaq.logical_observable(m)

    dem_text = cudaq.dem_from_kernel(kernel, 2)
    assert _summary(dem_text) == {"errors": 0, "detectors": 2, "observables": 1}


def test_single_noisy_detector():
    """One X_ERROR + one detector → DEM has one error referencing D0."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        m = mz(q)
        cudaq.detector(m)

    noise = cudaq.NoiseModel()
    dem_text = cudaq.dem_from_kernel(kernel, noise_model=noise)
    summary = _summary(dem_text)
    assert summary == {"errors": 1, "detectors": 1, "observables": 0}
    assert "error(0.1" in dem_text
    assert "D0" in dem_text


def test_three_mz_multi_detector():
    """Three measurements + variadic detector + scalar observable."""

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        q2 = cudaq.qubit()
        x(q0)
        x(q1)
        cudaq.apply_noise(cudaq.XError, 0.05, q0)
        cudaq.apply_noise(cudaq.XError, 0.05, q1)
        cudaq.apply_noise(cudaq.XError, 0.05, q2)
        m0 = mz(q0)
        m1 = mz(q1)
        m2 = mz(q2)
        cudaq.detector(m0, m1, m2)
        cudaq.logical_observable(m0)

    noise = cudaq.NoiseModel()
    dem_text = cudaq.dem_from_kernel(kernel, noise_model=noise)
    summary = _summary(dem_text)
    assert summary == {"errors": 2, "detectors": 1, "observables": 1}


def test_memory_experiment_two_rounds():
    """Multi-round memory experiment with cross-round detectors."""

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        q2 = cudaq.qubit()
        # Round 0
        cudaq.apply_noise(cudaq.XError, 0.03, q0)
        cudaq.apply_noise(cudaq.XError, 0.03, q1)
        cudaq.apply_noise(cudaq.XError, 0.03, q2)
        m0_r0 = mz(q0)
        m1_r0 = mz(q1)
        m2_r0 = mz(q2)
        # Round 1
        cudaq.apply_noise(cudaq.XError, 0.03, q0)
        cudaq.apply_noise(cudaq.XError, 0.03, q1)
        cudaq.apply_noise(cudaq.XError, 0.03, q2)
        m0_r1 = mz(q0)
        m1_r1 = mz(q1)
        m2_r1 = mz(q2)
        cudaq.detector(m0_r0, m0_r1)
        cudaq.detector(m1_r0, m1_r1)
        cudaq.detector(m2_r0, m2_r1)
        cudaq.logical_observable(m0_r1, m1_r1, m2_r1)

    noise = cudaq.NoiseModel()
    dem_text = cudaq.dem_from_kernel(kernel, noise_model=noise)
    summary = _summary(dem_text)
    assert summary == {"errors": 4, "detectors": 3, "observables": 1}


def test_non_clifford_raises():
    """Non-Clifford gate triggers a Clifford-only diagnostic from Stim."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        ry(0.3, q)
        m = mz(q)
        cudaq.detector(m)

    with pytest.raises(RuntimeError, match=r"Clifford"):
        cudaq.dem_from_kernel(kernel)


def test_make_kernel_builder():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc()
    m = kernel.mz(q)
    kernel.detector(m)
    kernel.logical_observable(m)

    # Cross-round pair
    qsA = kernel.qalloc(2)
    qsB = kernel.qalloc(2)
    prev = kernel.mz(qsA)
    curr = kernel.mz(qsB)
    kernel.detectors(prev, curr)

    dem_text = cudaq.dem_from_kernel(kernel)
    assert _summary(dem_text) == {"errors": 0, "detectors": 3, "observables": 1}


def test_emulate_target_independent():
    """The DEM analysis runs through Stim regardless of the active target.

    Sets a hardware emulate target, then verifies a simple kernel still
    produces a DEM with the expected detector and observable references.
    """
    cudaq.set_target("ionq", emulate=True)
    try:

        @cudaq.kernel
        def kernel():
            q = cudaq.qubit()
            m = mz(q)
            cudaq.detector(m)
            cudaq.logical_observable(m)

        dem_text = cudaq.dem_from_kernel(kernel)
        assert _summary(dem_text) == {
            "errors": 0,
            "detectors": 1,
            "observables": 1
        }
    finally:
        cudaq.reset_target()


def test_dem_and_run():

    @cudaq.kernel
    def kernel() -> bool:
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        m = mz(q)
        cudaq.detector(m)
        return m[0] ^ m[1]

    dem_text = cudaq.dem_from_kernel(kernel)
    summary = _summary(dem_text)
    assert summary == {"errors": 0, "detectors": 1, "observables": 0}

    results = cudaq.run(kernel, shots_count=10)
    assert len(results) == 10
    assert all(False == r for r in results)


def test_conditional_feedback_rejected():

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        h(q0)
        m0 = mz(q0)
        if m0:
            x(q1)
        m1 = mz(q1)
        cudaq.detector(m0, m1)

    with pytest.raises(RuntimeError, match=r"branches on a measurement"):
        cudaq.dem_from_kernel(kernel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
