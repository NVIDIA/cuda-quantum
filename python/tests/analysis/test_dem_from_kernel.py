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


# ---------------------------------------------------------------------------
# Return-type contract: dem_from_kernel always returns DEMResult
# ---------------------------------------------------------------------------


def test_dem_result_type():
    """dem_from_kernel always returns a DEMResult, never a bare str or tuple."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        m = mz(q)
        cudaq.detector(m)
        cudaq.logical_observable(m)

    result = cudaq.dem_from_kernel(kernel)
    assert isinstance(result, cudaq.DEMResult)


def test_dem_result_str_returns_dem_text():
    """str(result) equals result.dem — printing is unchanged."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        m = mz(q)
        cudaq.detector(m)

    result = cudaq.dem_from_kernel(kernel)
    assert str(result) == result.dem
    assert isinstance(result.dem, str)


def test_dem_result_repr():
    """repr(result) includes detector, observable, and measurement counts."""

    @cudaq.kernel
    def kernel(n_rounds: int):
        q = cudaq.qubit()
        m = mz(q)
        for _ in range(n_rounds):
            m_new = mz(q)
            cudaq.detector(m_new, m)
            m = m_new
        cudaq.logical_observable(m)

    result = cudaq.dem_from_kernel(kernel, 2)
    r = repr(result)
    assert "detectors=2" in r
    assert "observables=1" in r
    assert "measurements=3" in r


# ---------------------------------------------------------------------------
# Count fields: always present, even without matrices
# ---------------------------------------------------------------------------


def test_dem_result_count_fields():
    """num_detectors / num_observables / num_measurements are correct."""

    @cudaq.kernel
    def kernel(n_rounds: int):
        q = cudaq.qubit()
        m = mz(q)
        for _ in range(n_rounds):
            m_new = mz(q)
            cudaq.detector(m_new, m)
            m = m_new
        cudaq.logical_observable(m)

    result = cudaq.dem_from_kernel(kernel, 2)
    assert result.num_detectors == 2
    assert result.num_observables == 1
    assert result.num_measurements == 3


def test_dem_result_count_fields_when_matrices_skipped():
    """Count fields are populated even when return_measurement_matrices=False."""

    @cudaq.kernel
    def kernel(n_rounds: int):
        q = cudaq.qubit()
        m = mz(q)
        for _ in range(n_rounds):
            m_new = mz(q)
            cudaq.detector(m_new, m)
            m = m_new
        cudaq.logical_observable(m)

    result = cudaq.dem_from_kernel(kernel, 2, return_measurement_matrices=False)
    assert result.num_detectors == 2
    assert result.num_observables == 1
    assert result.num_measurements == 3


def test_dem_result_count_fields_no_detectors():
    """Kernel with no detectors: all counts are zero except num_measurements."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        mz(q)

    result = cudaq.dem_from_kernel(kernel)
    assert result.num_detectors == 0
    assert result.num_observables == 0
    assert result.num_measurements == 1


# ---------------------------------------------------------------------------
# matrices_computed flag
# ---------------------------------------------------------------------------


def test_dem_result_matrices_computed_true_by_default():
    """matrices_computed is True when return_measurement_matrices is not set."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        m = mz(q)
        cudaq.detector(m)

    result = cudaq.dem_from_kernel(kernel)
    assert result.matrices_computed is True


def test_dem_result_matrices_computed_false_when_opted_out():
    """matrices_computed is False when return_measurement_matrices=False."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        m = mz(q)
        cudaq.detector(m)

    result = cudaq.dem_from_kernel(kernel, return_measurement_matrices=False)
    assert result.matrices_computed is False


# ---------------------------------------------------------------------------
# m2d / m2o row lists (neutral C++ form)
# ---------------------------------------------------------------------------


def test_dem_result_m2d_row_lists():
    """m2d / m2o are lists-of-lists of measurement indices (not scipy)."""

    @cudaq.kernel
    def kernel(n_rounds: int):
        q = cudaq.qubit()
        m = mz(q)
        for _ in range(n_rounds):
            m_new = mz(q)
            cudaq.detector(m_new, m)
            m = m_new
        cudaq.logical_observable(m)

    result = cudaq.dem_from_kernel(kernel, 2)
    # m2d: 2 detectors, each referencing 2 consecutive measurements
    assert len(result.m2d) == 2
    assert sorted(result.m2d[0]) == [0, 1]  # det0 = m0 XOR m1
    assert sorted(result.m2d[1]) == [1, 2]  # det1 = m1 XOR m2
    # m2o: 1 observable referencing the last measurement
    assert len(result.m2o) == 1
    assert result.m2o[0] == [2]


def test_dem_result_m2d_empty_when_not_computed():
    """m2d / m2o are empty lists when matrices were not requested."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        m = mz(q)
        cudaq.detector(m)
        cudaq.logical_observable(m)

    result = cudaq.dem_from_kernel(kernel, return_measurement_matrices=False)
    assert result.m2d == [] or result.m2d == [[]]  # empty or not populated
    assert result.m2o == [] or result.m2o == [[]]


# ---------------------------------------------------------------------------
# m2d_matrix / m2o_matrix — scipy properties
# ---------------------------------------------------------------------------


def test_dem_result_m2d_matrix_no_detectors():
    """Kernel with no detectors / observables: empty matrices, not None."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        mz(q)

    result = cudaq.dem_from_kernel(kernel)
    assert result.m2d_matrix.shape == (0, 1)
    assert result.m2d_matrix.nnz == 0
    assert result.m2o_matrix.shape == (0, 1)
    assert result.m2o_matrix.nnz == 0


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------


def test_dem_result_annotations_empty_default():
    """annotations is an empty dict by default."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        mz(q)

    result = cudaq.dem_from_kernel(kernel)
    assert result.annotations == {}


def test_dem_result_annotations_set_at_construction():
    """annotations set via the constructor are readable on the result.

    Each property access returns a new Python dict copy (matching SampleResult
    semantics); in-place mutation of that copy does not persist.
    """
    result = cudaq.DEMResult("detector D0",
                             annotations={"source": "test_backend"})
    assert result.annotations["source"] == "test_backend"


# ---------------------------------------------------------------------------
# Constructor: DEMResult can be built from Python
# ---------------------------------------------------------------------------


def test_dem_result_constructor_basic():
    """DEMResult can be constructed directly from Python with typed fields."""
    dem_text = "error(0.1) D0\ndetector D0"
    result = cudaq.DEMResult(
        dem_text,
        m2d=[[0]],
        m2o=[],
        num_detectors=1,
        num_observables=0,
        num_measurements=1,
        annotations={"source": "hand-built"},
    )
    assert result.dem == dem_text
    assert result.num_detectors == 1
    assert result.num_observables == 0
    assert result.num_measurements == 1
    assert result.annotations["source"] == "hand-built"
    assert result.m2d == [[0]]
    assert result.m2o == []


def test_dem_result_constructor_defaults():
    """DEMResult constructor fields are all optional after dem."""
    result = cudaq.DEMResult("detector D0")
    assert result.num_detectors == 0
    assert result.num_observables == 0
    assert result.num_measurements == 0
    assert result.annotations == {}
    assert result.m2d == []
    assert result.m2o == []


# ---------------------------------------------------------------------------
# from_matrices classmethod
# ---------------------------------------------------------------------------


def test_dem_result_from_matrices():
    """CSR width is inferred unless an equal explicit count is supplied."""
    import numpy as np
    import scipy.sparse as sp

    dem_text = "error(0.1) D0"
    m2d_csr = sp.csr_matrix(np.array([[1, 0, 1]], dtype=np.uint8))
    m2o_csr = sp.csr_matrix(np.array([[0, 0, 1]], dtype=np.uint8))

    result = cudaq.DEMResult.from_matrices(
        dem_text,
        m2d_csr,
        m2o_csr,
        num_detectors=1,
        num_observables=1,
    )
    assert result.dem == dem_text
    assert result.num_detectors == 1
    assert result.num_observables == 1
    assert result.num_measurements == 3
    assert sorted(result.m2d[0]) == [0, 2]
    assert result.m2o[0] == [2]
    # Round-trip: the scipy properties should match the input
    import scipy.sparse as _sp
    assert (_sp.csr_matrix(result.m2d_matrix) - m2d_csr).nnz == 0
    assert (_sp.csr_matrix(result.m2o_matrix) - m2o_csr).nnz == 0


def test_dem_result_from_matrices_rejects_inconsistent_widths():
    """The two CSR matrices must describe the same measurement space."""
    import scipy.sparse as sp

    m2d_csr = sp.csr_matrix(([1], ([0], [2])), shape=(1, 3))
    m2o_csr = sp.csr_matrix(([1], ([0], [3])), shape=(1, 4))

    with pytest.raises(ValueError, match="same number of columns"):
        cudaq.DEMResult.from_matrices("error(0.1) D0", m2d_csr, m2o_csr)


def test_dem_result_from_matrices_rejects_incorrect_explicit_width():
    """An explicit measurement count is checked against both CSR matrices."""
    import scipy.sparse as sp

    m2d_csr = sp.csr_matrix(([1], ([0], [2])), shape=(1, 3))
    m2o_csr = sp.csr_matrix(([1], ([0], [2])), shape=(1, 3))

    with pytest.raises(ValueError, match="num_measurements must match"):
        cudaq.DEMResult.from_matrices("error(0.1)",
                                      m2d_csr,
                                      m2o_csr,
                                      num_measurements=2)


# ---------------------------------------------------------------------------
# Existing functional tests — updated to use .dem for string operations
# ---------------------------------------------------------------------------


def test_trivial_empty_dem():
    """Kernel without QEC declarations yields an empty DEM."""

    @cudaq.kernel
    def trivial():
        q = cudaq.qubit()
        h(q)
        mz(q)

    result = cudaq.dem_from_kernel(trivial)
    assert _summary(result.dem) == {
        "errors": 0,
        "detectors": 0,
        "observables": 0
    }


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

    result = cudaq.dem_from_kernel(kernel, 2)
    assert _summary(result.dem) == {
        "errors": 0,
        "detectors": 2,
        "observables": 1
    }


def test_single_noisy_detector():
    """One X_ERROR + one detector → DEM has one error referencing D0."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        m = mz(q)
        cudaq.detector(m)

    noise = cudaq.NoiseModel()
    result = cudaq.dem_from_kernel(kernel, noise_model=noise)
    assert _summary(result.dem) == {
        "errors": 1,
        "detectors": 1,
        "observables": 0
    }
    assert "error(0.1" in result.dem
    assert "D0" in result.dem


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
    result = cudaq.dem_from_kernel(kernel, noise_model=noise)
    assert _summary(result.dem) == {
        "errors": 2,
        "detectors": 1,
        "observables": 1
    }


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
    result = cudaq.dem_from_kernel(kernel, noise_model=noise)
    assert _summary(result.dem) == {
        "errors": 4,
        "detectors": 3,
        "observables": 1
    }


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

    result = cudaq.dem_from_kernel(kernel)
    assert _summary(result.dem) == {
        "errors": 0,
        "detectors": 3,
        "observables": 1
    }


def test_emulate_target_independent():
    """The DEM analysis runs through Stim regardless of the active target."""
    cudaq.set_target("ionq", emulate=True)
    try:

        @cudaq.kernel
        def kernel():
            q = cudaq.qubit()
            m = mz(q)
            cudaq.detector(m)
            cudaq.logical_observable(m)

        result = cudaq.dem_from_kernel(kernel)
        assert _summary(result.dem) == {
            "errors": 0,
            "detectors": 1,
            "observables": 1
        }

        counts = cudaq.sample(kernel, shots_count=10)
        assert counts["0"] == 10

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

        assert "D0 D1 D2 D3" in dem_raw.dem
        assert "^" not in dem_raw.dem
        assert "D0 D1 D2 D3" not in dem_decomposed.dem
        assert "^" in dem_decomposed.dem
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

    result = cudaq.dem_from_kernel(kernel)
    assert _summary(result.dem) == {
        "errors": 0,
        "detectors": 1,
        "observables": 0
    }

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
    """decompose_errors=True splits four-detector hyperedges into pair edges."""

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.125, q0)
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

    assert "D0 D1 D2 D3" in dem_raw.dem
    assert "^" not in dem_raw.dem
    assert "D0 D1 D2 D3" not in dem_decomposed.dem
    assert "^" in dem_decomposed.dem
    assert "error(0.25) D0 D1 ^ D2 D3" in dem_decomposed.dem


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

    result = cudaq.dem_from_kernel(kernel, allow_gauge_detectors=True)
    assert isinstance(result, cudaq.DEMResult)


def test_decompose_and_ignore_failures():
    """Three detectors on the same measurement create a 3-way hyperedge that
    Stim cannot decompose into pairs. decompose_errors=True must raise unless
    ignore_decomposition_failures=True, which silently accepts the bad edge."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        m = mz(q)

        # odd-cardinality hyperedge {D0, D1, D2}
        cudaq.detector(m)
        cudaq.detector(m)
        cudaq.detector(m)

    noise = cudaq.NoiseModel()

    # Without decompose_errors the raw hyperedge is returned fine.
    dem_raw = cudaq.dem_from_kernel(kernel, noise_model=noise)
    assert "D0 D1 D2" in dem_raw.dem

    # decompose_errors=True on an odd hyperedge raises.
    with pytest.raises(Exception):
        cudaq.dem_from_kernel(kernel, noise_model=noise, decompose_errors=True)

    # ignore_decomposition_failures=True keeps the undecomposable edge as-is.
    dem_ignored = cudaq.dem_from_kernel(kernel,
                                        noise_model=noise,
                                        decompose_errors=True,
                                        ignore_decomposition_failures=True)
    assert "D0 D1 D2" in dem_ignored.dem
    assert "^" not in dem_ignored.dem


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
    result = cudaq.dem_from_kernel(kernel,
                                   noise_model=noise,
                                   approximate_disjoint_errors_threshold=0.1)
    assert _count_errors(result.dem) > 0


def test_fold_loops_and_block_decomposition():
    """fold_loops is a no-op for flat circuits. block_decomposition_from_introducing_
    remnant_edges raises when a hyperedge cannot be decomposed."""
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

    result = cudaq.dem_from_kernel(k_3det, noise_model=noise)
    result_folded = cudaq.dem_from_kernel(k_3det,
                                          noise_model=noise,
                                          fold_loops=True)
    assert result.dem == result_folded.dem

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


# ---------------------------------------------------------------------------
# GF(2) matrix correctness
# ---------------------------------------------------------------------------


def test_return_measurement_matrices_no_detectors():
    """Kernel with no detectors or observables yields empty matrices, not None."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        mz(q)

    result = cudaq.dem_from_kernel(kernel)
    m2d = result.m2d_matrix
    m2o = result.m2o_matrix
    assert m2d.shape == (0, 1)
    assert m2d.nnz == 0
    assert m2o.shape == (0, 1)
    assert m2o.nnz == 0


def test_return_measurement_matrices_two_rounds():
    """m2d_matrix has shape (num_detectors, num_measurements) with correct entries."""

    @cudaq.kernel
    def kernel(n_rounds: int):
        q = cudaq.qubit()
        m = mz(q)
        for _ in range(n_rounds):
            m_new = mz(q)
            cudaq.detector(m_new, m)
            m = m_new
        cudaq.logical_observable(m)

    result = cudaq.dem_from_kernel(kernel, 2)
    m2d = result.m2d_matrix
    m2o = result.m2o_matrix

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
    """m2d_matrix and m2o_matrix are scipy CSR matrices when computed."""
    import scipy.sparse as sp

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        m = mz(q)
        cudaq.detector(m)
        cudaq.logical_observable(m)

    result = cudaq.dem_from_kernel(kernel)
    m2d = result.m2d_matrix
    m2o = result.m2o_matrix
    assert isinstance(result.dem, str)
    assert sp.issparse(m2d)
    assert sp.issparse(m2o)
    assert m2d.shape == (1, 1)
    assert m2d[0, 0] == 1
    assert m2o.shape == (1, 1)
    assert m2o[0, 0] == 1


def test_no_return_measurement_matrices_gives_none_matrices():
    """Without return_measurement_matrices the matrix properties are None."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        m = mz(q)
        cudaq.detector(m)

    result = cudaq.dem_from_kernel(kernel, return_measurement_matrices=False)
    assert result.m2d_matrix is None
    assert result.m2o_matrix is None


def test_return_measurement_matrices_duplicate_targets_cancel():
    """detector(m, m) — GF(2) cancellation: the m2d row is all-zero."""
    noise = cudaq.NoiseModel()

    @cudaq.kernel
    def duplicate_detector_kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        m = mz(q)
        cudaq.detector(m, m)

    result = cudaq.dem_from_kernel(duplicate_detector_kernel, noise_model=noise)
    m2d = result.m2d_matrix
    m2o = result.m2o_matrix
    # Stim itself sees detector(m XOR m) = detector(0) — no error mechanism
    # touches D0, so the DEM lists the detector but no error lines.
    assert "detector D0" in result.dem
    assert "error(" not in result.dem
    assert m2d.shape == (1, 1)
    assert m2d[0, 0] == 0  # double reference cancels in GF(2)
    assert m2o.shape == (0, 1)


def test_return_measurement_matrices_odd_duplicate_survives():
    """detector(m, m, m) — three XOR'd copies reduce to one."""
    noise = cudaq.NoiseModel()

    @cudaq.kernel
    def triple_ref_kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        m = mz(q)
        cudaq.detector(m, m, m)

    result = cudaq.dem_from_kernel(triple_ref_kernel, noise_model=noise)
    assert "error(" in result.dem
    assert result.m2d_matrix.shape == (1, 1)
    assert result.m2d_matrix[0, 0] == 1


def test_return_measurement_matrices_mixed_duplicate():
    """detector(m0, m1, m1): m0 survives, m1 cancels."""
    noise = cudaq.NoiseModel()

    @cudaq.kernel
    def mixed_dup_kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        m0 = mz(q)
        m1 = mz(q)
        cudaq.detector(m0, m1, m1)

    result = cudaq.dem_from_kernel(mixed_dup_kernel, noise_model=noise)
    assert result.m2d_matrix.shape == (1, 2)
    dense = result.m2d_matrix.toarray()
    assert dense[0, 0] == 1  # m0 survives
    assert dense[0, 1] == 0  # m1 cancels


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
    result = cudaq.dem_from_kernel(kernel, noise_model=noise)
    assert "D0 D1 D2 D3" in result.dem
    assert "^" not in result.dem
    assert result.m2d_matrix.shape == (4, 2)
    assert result.m2o_matrix.shape == (0, 2)

    # With decompose_errors=True the hyperedge is split.
    result2 = cudaq.dem_from_kernel(kernel,
                                    noise_model=noise,
                                    decompose_errors=True)
    assert "^" in result2.dem
    assert "D0 D1 D2 D3" not in result2.dem
    # Matrices reflect the circuit, not the DEM decomposition.
    assert result2.m2d_matrix.shape == (4, 2)
    assert result2.m2o_matrix.shape == (0, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
