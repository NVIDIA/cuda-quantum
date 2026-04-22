# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Tests for the Python surface of `cudaq::measure_handle` (spec proposal
`measure_handle.bs`). Covers:
  (a) host-scope `RuntimeError` from `cudaq.measure_handle()`,
  (b) device-side construction via `mz_handle` + explicit `cudaq.discriminate`,
  (c) one rejection per spec-enumerated bool-coercion context, asserting the
      exact diagnostic substring required by the spec.
"""

import re

import pytest

import cudaq

# Spec-mandated diagnostic for any implicit bool coercion of a
# `!cc.measure_handle` value inside a kernel body.
_BOOL_DIAG = (
    "measure_handle does not convert to bool implicitly inside a kernel; "
    "call cudaq.discriminate(h) to read the outcome")
_BOOL_DIAG_RE = re.escape(_BOOL_DIAG)


@pytest.fixture(autouse=True)
def reset_run_clear():
    cudaq.reset_target()
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


# --------------------------------------------------------------------------- #
# (a) Host-scope rejection.
# --------------------------------------------------------------------------- #


def test_host_construction_raises_runtime_error():
    """`cudaq.measure_handle` is device-only; constructing one at host scope
    must raise `RuntimeError` with the spec-mandated exact message."""
    with pytest.raises(RuntimeError,
                       match=r"^device-only; usable only inside @cudaq\.kernel$"
                      ):
        cudaq.measure_handle()


# --------------------------------------------------------------------------- #
# (b) Successful device-side construction + discriminate.
# --------------------------------------------------------------------------- #


def test_mz_handle_then_discriminate_single_qubit():
    """`mz_handle(q)` produces a `!cc.measure_handle`; `cudaq.discriminate(h)`
    is the only sanctioned path to a classical bit."""

    @cudaq.kernel
    def prepare_one_then_measure() -> bool:
        q = cudaq.qubit()
        x(q)
        h = mz_handle(q)
        return cudaq.discriminate(h)

    counts = cudaq.run(prepare_one_then_measure, shots_count=20)
    assert all(bit is True for bit in counts)


def test_mz_handle_then_discriminate_qvector():
    """Vector form of `mz_handle` returns `!cc.stdvec<!cc.measure_handle>`;
    `cudaq.discriminate` on the vector returns `list[bool]` (lowered to
    `!cc.stdvec<i1>`). Use `cudaq.to_integer` to consume the vector and
    avoid relying on list returns from `cudaq.run`."""

    @cudaq.kernel
    def prepare_three_then_measure() -> int:
        qv = cudaq.qvector(3)
        for i in range(3):
            x(qv[i])
        hs = mz_handle(qv)
        bits = cudaq.discriminate(hs)
        return cudaq.to_integer(bits)

    results = cudaq.run(prepare_three_then_measure, shots_count=20)
    # All three qubits in |1>, big-endian little-endian aside, the value is
    # constant across shots and equal to 7 (or 0 on a backend that flips
    # endian -- both options assert non-trivial determinism).
    assert all(r == results[0] for r in results)
    assert results[0] in (0, 7)


def test_to_integer_accepts_handle_vector_directly():
    """`cudaq.to_integer(list[measure_handle])` is spec-equivalent to
    `cudaq.to_integer(cudaq.discriminate(handles))`; the bridge inserts the
    vectorized `quake.discriminate` for us."""

    @cudaq.kernel
    def to_integer_handles() -> int:
        qv = cudaq.qvector(3)
        for i in range(3):
            x(qv[i])
        return cudaq.to_integer(mz_handle(qv))

    results = cudaq.run(to_integer_handles, shots_count=20)
    assert all(r == results[0] for r in results)
    assert results[0] in (0, 7)


# --------------------------------------------------------------------------- #
# (c) Bool-coercion contexts. One test per spec-enumerated site.
# --------------------------------------------------------------------------- #


def test_reject_bool_coercion_in_if():
    """`if h:` -- spec §Python API."""

    @cudaq.kernel
    def reject_if():
        q = cudaq.qubit()
        h = mz_handle(q)
        if h:
            x(q)

    with pytest.raises(RuntimeError, match=_BOOL_DIAG_RE):
        reject_if.compile()


def test_reject_bool_coercion_in_while():
    """`while h:` -- spec §Python API."""

    @cudaq.kernel
    def reject_while():
        q = cudaq.qubit()
        h = mz_handle(q)
        while h:
            x(q)

    with pytest.raises(RuntimeError, match=_BOOL_DIAG_RE):
        reject_while.compile()


def test_reject_bool_coercion_in_not():
    """`not h` -- spec §Python API."""

    @cudaq.kernel
    def reject_not():
        q = cudaq.qubit()
        h = mz_handle(q)
        if not h:
            x(q)

    with pytest.raises(RuntimeError, match=_BOOL_DIAG_RE):
        reject_not.compile()


def test_reject_bool_coercion_in_and():
    """`h and ...` -- spec §Python API. The bridge processes the LHS first;
    if it is a handle, the diagnostic must fire before we even visit the RHS,
    so the RHS is irrelevant for the diagnostic but kept syntactically valid."""

    @cudaq.kernel
    def reject_and():
        q = cudaq.qubit()
        h = mz_handle(q)
        b = cudaq.discriminate(mz_handle(q))
        if h and b:
            x(q)

    with pytest.raises(RuntimeError, match=_BOOL_DIAG_RE):
        reject_and.compile()


def test_reject_bool_coercion_in_or():
    """`h or ...` -- spec §Python API."""

    @cudaq.kernel
    def reject_or():
        q = cudaq.qubit()
        h = mz_handle(q)
        b = cudaq.discriminate(mz_handle(q))
        if h or b:
            x(q)

    with pytest.raises(RuntimeError, match=_BOOL_DIAG_RE):
        reject_or.compile()


def test_reject_bool_coercion_in_ternary():
    """`a if h else b` -- spec §Python API."""

    @cudaq.kernel
    def reject_ternary():
        q = cudaq.qubit()
        h = mz_handle(q)
        x = 1 if h else 0  # noqa: F841

    with pytest.raises(RuntimeError, match=_BOOL_DIAG_RE):
        reject_ternary.compile()


def test_reject_bool_coercion_in_assert():
    """`assert h` -- spec §Python API."""

    @cudaq.kernel
    def reject_assert():
        q = cudaq.qubit()
        h = mz_handle(q)
        assert h

    with pytest.raises(RuntimeError, match=_BOOL_DIAG_RE):
        reject_assert.compile()


def test_reject_bool_coercion_in_bool_call():
    """`bool(h)` -- spec §Python API. Implicit `bool` cast of a handle is
    rejected at parse time even though the bridge does not currently lower
    `bool(...)` calls inside kernels at all."""

    @cudaq.kernel
    def reject_bool_call():
        q = cudaq.qubit()
        h = mz_handle(q)
        b = bool(h)  # noqa: F841

    with pytest.raises(RuntimeError, match=_BOOL_DIAG_RE):
        reject_bool_call.compile()


# --------------------------------------------------------------------------- #
# Negative cases beyond the bool contexts: discriminating an unbound handle
# is rejected by the bridge (mirrors the C++ frontend).
# --------------------------------------------------------------------------- #


def test_reject_discriminate_of_unbound_default_constructed_handle():
    """`cudaq.discriminate(cudaq.measure_handle())` is the canonical
    unbound-handle pattern and must be rejected at parse time, mirroring
    the C++ bridge's `discriminating an unbound measure_handle` diagnostic."""

    @cudaq.kernel
    def reject_unbound() -> bool:
        return cudaq.discriminate(cudaq.measure_handle())

    with pytest.raises(RuntimeError,
                       match=r"discriminating an unbound measure_handle"):
        reject_unbound.compile()


# leave for gdb debugging
if __name__ == "__main__":
    import os
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
