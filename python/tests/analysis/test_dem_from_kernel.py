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

        @cudaq.kernel
        def hyperedge_kernel():
            q0 = cudaq.qubit()
            q1 = cudaq.qubit()
            x.ctrl(q0, q1)
            m0 = mz(q0)
            m1 = mz(q1)
            cudaq.detector(m0)
            cudaq.detector(m0)
            cudaq.detector(m1)
            cudaq.detector(m1)

        pauli2_probs = [0.0] * 15
        pauli2_probs[4] = 0.25  # XX
        noise = cudaq.NoiseModel()
        noise.add_channel("x", [0, 1], cudaq.Pauli2(pauli2_probs))

        dem_raw = cudaq.dem_from_kernel(hyperedge_kernel, noise_model=noise)
        dem_decomposed = cudaq.dem_from_kernel(hyperedge_kernel,
                                               noise_model=noise,
                                               decompose_errors=True)

        assert "D0 D1 D2 D3" in dem_raw
        assert "^" not in dem_raw
        assert "D0 D1 D2 D3" not in dem_decomposed
        assert "^" in dem_decomposed
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


def test_decompose_errors_correlated_xx():
    """dem_options decompose_errors=True splits four-detector hyperedges into pair edges."""

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.125, q0)
        # pauli2 probabilities: IX,IY,IZ,XI,XX,XY,XZ,YI,YX,YY,YZ,ZI,ZX,ZY,ZZ
        cudaq.apply_noise(cudaq.Pauli2, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, q0, q1)
        m0 = mz(q0)
        m1 = mz(q1)
        cudaq.detector(m0)
        cudaq.detector(m0)
        cudaq.detector(m1)
        cudaq.detector(m1)

    noise = cudaq.NoiseModel()
    dem_raw = cudaq.dem_from_kernel(kernel, noise_model=noise)
    dem_decomposed = cudaq.dem_from_kernel(kernel,
                                           noise_model=noise,
                                           decompose_errors=True)

    assert "D0 D1 D2 D3" in dem_raw
    assert "^" not in dem_raw
    assert "D0 D1 D2 D3" not in dem_decomposed
    assert "^" in dem_decomposed
    assert "error(0.25) D0 D1 ^ D2 D3" in dem_decomposed


def test_allow_gauge_detectors():
    """allow_gauge_detectors=True permits detectors with non-deterministic parity."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        h(q)
        m = mz(q)
        cudaq.detector(m)

    with pytest.raises(Exception):
        cudaq.dem_from_kernel(kernel)

    dem = cudaq.dem_from_kernel(kernel, allow_gauge_detectors=True)
    assert isinstance(dem, str)


def test_decompose_and_ignore_failures():
    """Three detectors on the same measurement create a 3-way hyperedge that
    Stim cannot decompose into pairs.  decompose_errors=True must raise unless
    ignore_decomposition_failures=True, which silently accepts the bad edge."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        m = mz(q)

        # odd-cardinality hyperedge {D0, D1, D2}
        # with no other mechanism to decompose into.
        cudaq.detector(m)
        cudaq.detector(m)
        cudaq.detector(m)

    noise = cudaq.NoiseModel()

    # Without decompose_errors the raw hyperedge is returned fine.
    dem_raw = cudaq.dem_from_kernel(kernel, noise_model=noise)
    assert "D0 D1 D2" in dem_raw

    # decompose_errors=True on an odd hyperedge raises.
    with pytest.raises(Exception):
        cudaq.dem_from_kernel(kernel, noise_model=noise, decompose_errors=True)

    # ignore_decomposition_failures=True keeps the undecomposable edge as-is
    dem_ignored = cudaq.dem_from_kernel(kernel,
                                        noise_model=noise,
                                        decompose_errors=True,
                                        ignore_decomposition_failures=True)
    assert "D0 D1 D2" in dem_ignored
    assert "^" not in dem_ignored


def test_approximate_disjoint_errors_threshold():
    """Pauli1 with nonzero pX and pY cannot be expressed as independent errors;
    Stim raises unless approximate_disjoint_errors_threshold exceeds all components."""
    pX, pY = 0.05, 0.08

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.Pauli1, pX, pY, 0.0, q)
        m = mz(q)
        cudaq.detector(m)

    noise = cudaq.NoiseModel()
    with pytest.raises(Exception):
        cudaq.dem_from_kernel(kernel, noise_model=noise)
    with pytest.raises(Exception):
        cudaq.dem_from_kernel(kernel,
                              noise_model=noise,
                              approximate_disjoint_errors_threshold=0.06)
    dem = cudaq.dem_from_kernel(kernel,
                                noise_model=noise,
                                approximate_disjoint_errors_threshold=0.1)
    assert _count_errors(dem) > 0


def test_fold_loops_and_block_decomposition():
    """fold_loops is a no-op for flat circuits.  block_decomposition_from_introducing_
    remnant_edges raises when a hyperedge cannot be decomposed; Stim adds the flag name
    to the error message, distinguishing it from a plain decomposition failure."""
    noise = cudaq.NoiseModel()

    # Three detectors on one measurement create an odd-cardinality hyperedge that
    # cannot be decomposed.  block=True causes Stim to include the flag name in the
    # error message; block=False raises too but without that annotation.
    @cudaq.kernel
    def k_3det():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        m = mz(q)
        cudaq.detector(m)
        cudaq.detector(m)
        cudaq.detector(m)

    dem = cudaq.dem_from_kernel(k_3det, noise_model=noise)
    assert dem == cudaq.dem_from_kernel(k_3det,
                                        noise_model=noise,
                                        fold_loops=True)

    with pytest.raises(
            Exception,
            match="block_decomposition_from_introducing_remnant_edges"):
        cudaq.dem_from_kernel(
            k_3det,
            noise_model=noise,
            decompose_errors=True,
            block_decomposition_from_introducing_remnant_edges=True)


def test_dem_options_unknown_key_raises():
    """Passing an unknown keyword argument raises ValueError."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        mz(q)

    with pytest.raises(ValueError, match="unknown keyword argument"):
        cudaq.dem_from_kernel(kernel, not_a_real_option=True)


def test_return_measurement_matrices_no_detectors():
    """Kernel with no detectors or observables yields empty m2d and m2o matrices."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        mz(q)

    dem_text, m2d, m2o = cudaq.dem_from_kernel(kernel,
                                               return_measurement_matrices=True)
    assert m2d.shape == (0, 1)
    assert m2d.nnz == 0
    assert m2o.shape == (0, 1)
    assert m2o.nnz == 0


def test_return_measurement_matrices_two_rounds():
    """Two-round memory experiment: verify m2d and m2o shapes and mappings."""

    @cudaq.kernel
    def kernel(n_rounds: int):
        q = cudaq.qubit()
        m = mz(q)
        for _ in range(n_rounds):
            m_new = mz(q)
            cudaq.detector(m_new, m)
            m = m_new
        cudaq.logical_observable(m)

    dem_text, m2d, m2o = cudaq.dem_from_kernel(kernel,
                                               2,
                                               return_measurement_matrices=True)
    # 3 measurements (m0, m1, m2), 2 detectors, 1 observable
    assert m2d.shape == (2, 3)
    dense = m2d.toarray()
    # det0 = m0 XOR m1, det1 = m1 XOR m2
    assert dense[0, 0] == 1 and dense[0, 1] == 1 and dense[0, 2] == 0
    assert dense[1, 0] == 0 and dense[1, 1] == 1 and dense[1, 2] == 1
    # observable 0 = m2 (the last measurement)
    assert m2o.shape == (1, 3)
    obs_dense = m2o.toarray()
    assert obs_dense[0, 0] == 0 and obs_dense[0, 1] == 0 and obs_dense[0,
                                                                       2] == 1


def test_return_measurement_matrices_type_is_scipy_sparse():
    """return_measurement_matrices=True yields scipy CSR sparse matrices for both m2d and m2o."""
    import scipy.sparse as sp

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        m = mz(q)
        cudaq.detector(m)
        cudaq.logical_observable(m)

    dem_text, m2d, m2o = cudaq.dem_from_kernel(kernel,
                                               return_measurement_matrices=True)
    assert isinstance(dem_text, str)
    assert sp.issparse(m2d)
    assert sp.issparse(m2o)
    assert m2d.shape == (1, 1)
    assert m2d[0, 0] == 1
    assert m2o.shape == (1, 1)
    assert m2o[0, 0] == 1


def test_no_return_measurement_matrices_returns_string():
    """Without return_measurement_matrices the return value is a plain string."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        m = mz(q)
        cudaq.detector(m)

    result = cudaq.dem_from_kernel(kernel)
    assert isinstance(result, str)


def test_return_measurement_matrices_with_dem_options():
    """return_measurement_matrices=True and other dem_options work together.

    Passes decompose_errors=True alongside return_measurement_matrices=True to
    verify that both the DEM option (edge decomposition) and matrix output are
    applied in the same call.  Uses the two-round memory experiment: with
    decompose_errors the DEM is unchanged (single-detector edges are already
    decomposed), so we focus on verifying the option is forwarded by also
    requesting allow_gauge_detectors which would normally raise for the
    h/mz-without-reset pattern.  Here we use the round-trip kernel that has
    well-defined detectors so decompose_errors is the clean observable.
    """
    noise = cudaq.NoiseModel()

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        cudaq.apply_noise(cudaq.Pauli2, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, q0, q1)
        m0 = mz(q0)
        m1 = mz(q1)
        cudaq.detector(m0)
        cudaq.detector(m0)
        cudaq.detector(m1)
        cudaq.detector(m1)

    # Without decompose_errors the four-detector hyperedge is returned raw.
    dem_text, m2d, m2o = cudaq.dem_from_kernel(kernel,
                                               noise_model=noise,
                                               return_measurement_matrices=True)
    assert "D0 D1 D2 D3" in dem_text
    assert "^" not in dem_text
    assert m2d.shape == (4, 2)
    assert m2o.shape == (0, 2)

    # With decompose_errors=True the hyperedge is split and ^ appears.
    dem_decomposed, m2d2, m2o2 = cudaq.dem_from_kernel(
        kernel,
        noise_model=noise,
        decompose_errors=True,
        return_measurement_matrices=True)
    assert "^" in dem_decomposed
    assert "D0 D1 D2 D3" not in dem_decomposed
    # Matrices reflect the same circuit regardless of decomposition.
    assert m2d2.shape == (4, 2)
    assert m2o2.shape == (0, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
