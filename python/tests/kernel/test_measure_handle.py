# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the Python surface of ``cudaq::measure_handle`` (spec proposal
``cudaq-spec/proposals/measure_handle.bs``) under the *single-API* shape:

    * ``mz`` / ``mx`` / ``my`` return ``cudaq.measure_handle`` (scalar) or
      ``list[cudaq.measure_handle]`` (vector form on a ``qvector``/``qview``).
    * Implicit ``bool`` coercion of a handle anywhere inside a kernel body
      is *accepted* and lowered to ``quake.discriminate`` -- there is no
      "use ``cudaq.discriminate`` instead" diagnostic any more.  The legacy
      ``mz_handle`` / ``mx_handle`` / ``my_handle`` and ``cudaq.discriminate``
      symbols are gone.
    * ``cudaq.to_bools(list[measure_handle])`` is the only sanctioned bulk
      discrimination path; ``cudaq.to_integer(list[measure_handle])`` is
      rejected and must be composed as ``cudaq.to_integer(cudaq.to_bools(h))``.
    * Default-constructed ``cudaq.measure_handle()`` reaching a coercion
      site is the canonical *unbound-handle* pattern; the bridge surfaces
      the spec diagnostic ``discriminating an unbound measure_handle``
      (mirrors ``lib/Frontend/nvqpp/ConvertExpr.cpp:699``).
    * Entry-point kernels may not name ``measure_handle`` -- directly or
      transitively -- in any parameter or return position; the diagnostic
      is the spec-canonical ``measure_handle cannot cross the host-device
      boundary; entry-point kernels must discriminate first`` (matches
      ``lib/Frontend/nvqpp/ASTBridge.cpp:679`` and the C++ AST-error
      oracle ``test/AST-error/measure_handle.cpp``).

These tests are the Python counterpart of the C++ AST-Quake oracle in
``test/AST-Quake/measure_handle.cpp`` (IR shape) and the C++ AST-error
oracle in ``test/AST-error/measure_handle.cpp`` (diagnostics).  They
exercise the AST-bridge code paths added in commit ``51b07e3fe1``
([Python] Rewire ast_bridge to single measure_handle API + bool-coercion +
to_bools).
"""

import re

import pytest

import cudaq

# ---------------------------------------------------------------------------
# Spec-mandated diagnostic strings.  Keep these in sync with the messages
# emitted by `python/cudaq/kernel/ast_bridge.py` (and the C++ ones in
# `lib/Frontend/nvqpp/`); the whole point of the spec mandating exact
# wording is that all frontends agree.
# ---------------------------------------------------------------------------
_UNBOUND_DIAG = "discriminating an unbound measure_handle"
_BOUNDARY_DIAG = ("measure_handle cannot cross the host-device boundary; "
                  "entry-point kernels must discriminate first")


@pytest.fixture(autouse=True)
def reset_run_clear():
    cudaq.reset_target()
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def _ir(kernel):
    """Materialize MLIR for ``kernel`` and return it as a string.  Triggers
    AST bridge codegen (so any spec diagnostic fires here) without running
    the kernel on a simulator -- enough for IR-shape and negative tests."""
    return str(kernel)


# ---------------------------------------------------------------------------
# Host-scope rejection (spec `measure_handle.bs`, Host-Device Boundary).
# `cudaq.measure_handle()` is device-only; constructing one at host scope
# raises `RuntimeError` with the spec-mandated exact message.
# ---------------------------------------------------------------------------


def test_host_construction_raises_runtime_error():
    with pytest.raises(
            RuntimeError,
            match=r"^device-only; usable only inside @cudaq\.kernel$"):
        cudaq.measure_handle()


def test_host_to_bools_raises_runtime_error():
    """`cudaq.to_bools` is similarly device-only; the host stub exists so
    ``cudaq.to_bools`` is a discoverable public symbol but raises if
    invoked outside a kernel."""
    with pytest.raises(
            RuntimeError,
            match=r"^device-only; usable only inside @cudaq\.kernel$"):
        cudaq.to_bools([])


# ---------------------------------------------------------------------------
# Scalar `mz` / `mx` / `my` return a `!cc.measure_handle` and do *not*
# inline a `quake.discriminate` (spec `measure_handle.bs`, Lowering).
# ---------------------------------------------------------------------------


def test_scalar_mz_emits_handle_no_discriminate():

    @cudaq.kernel
    def k():
        q = cudaq.qubit()
        h = mz(q)  # noqa: F841

    ir = _ir(k)
    assert "quake.mz" in ir
    assert "!cc.measure_handle" in ir
    assert "quake.discriminate" not in ir, (
        "scalar mz with no coercion site must not emit a discriminate; "
        "got:\n" + ir)


def test_scalar_mx_my_emit_handles():

    @cudaq.kernel
    def kx():
        q = cudaq.qubit()
        h = mx(q)  # noqa: F841

    @cudaq.kernel
    def ky():
        q = cudaq.qubit()
        h = my(q)  # noqa: F841

    irx, iry = _ir(kx), _ir(ky)
    assert re.search(r"quake\.mx.*-> !cc\.measure_handle", irx), irx
    assert re.search(r"quake\.my.*-> !cc\.measure_handle", iry), iry
    assert "quake.discriminate" not in irx
    assert "quake.discriminate" not in iry


def test_vector_mz_emits_stdvec_of_handles():

    @cudaq.kernel
    def k():
        qv = cudaq.qvector(3)
        hs = mz(qv)  # noqa: F841

    ir = _ir(k)
    assert "!cc.stdvec<!cc.measure_handle>" in ir
    assert "quake.discriminate" not in ir


# ---------------------------------------------------------------------------
# Bool-coercion sites lower to `quake.discriminate`.  One test per
# spec-enumerated context (cf. ``__discriminateIfMeasureHandle`` in
# ``ast_bridge.py``).  These are *positive* tests: the bridge must accept
# the code and emit the discriminate; the legacy "use cudaq.discriminate
# instead" rejection is gone.
# ---------------------------------------------------------------------------


def test_bool_coercion_in_if():

    @cudaq.kernel
    def k():
        q = cudaq.qubit()
        h = mz(q)
        if h:
            x(q)

    ir = _ir(k)
    assert re.search(r"quake\.discriminate.*-> i1", ir), ir


def test_bool_coercion_in_while():

    @cudaq.kernel
    def k():
        q = cudaq.qubit()
        h = mz(q)
        while h:
            x(q)
            h = mz(q)

    ir = _ir(k)
    assert re.search(r"quake\.discriminate.*-> i1", ir), ir


def test_bool_coercion_in_not():

    @cudaq.kernel
    def k():
        q = cudaq.qubit()
        h = mz(q)
        if not h:
            x(q)

    ir = _ir(k)
    assert re.search(r"quake\.discriminate.*-> i1", ir), ir


@pytest.mark.skip(reason="Python AST bridge does not currently support "
                  "ternary `IfExp` (raises 'does not currently support "
                  "ternary IfExp expressions' from `visit_IfExp`); the "
                  "bridge is wired up for the bool-coercion at the "
                  "ternary test (`__discriminateIfMeasureHandle` is "
                  "called from the IfExp visitor) but the visitor errors "
                  "out before reaching it.  Re-enable when generic "
                  "ternary support lands.")
def test_bool_coercion_in_ternary():

    @cudaq.kernel
    def k() -> int:
        q = cudaq.qubit()
        h = mz(q)
        return 1 if h else 0

    ir = _ir(k)
    assert re.search(r"quake\.discriminate.*-> i1", ir), ir


def test_bool_coercion_in_bool_call():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        return bool(mz(q))

    ir = _ir(k)
    assert re.search(r"quake\.discriminate.*-> i1", ir), ir


def test_bool_coercion_in_return():
    """Returning a `measure_handle` from a `-> bool` kernel coerces at the
    return site."""

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        h = mz(q)
        return h

    ir = _ir(k)
    assert re.search(r"quake\.discriminate.*-> i1", ir), ir


@pytest.mark.skip(reason="Python AST bridge does not currently support "
                  "annotated assignments (`AnnAssign`, e.g. `b: bool = "
                  "...`); the bridge raises 'does not currently support "
                  "AnnAssign expressions' before the bool-coercion site "
                  "can fire.  The bool-coercion-on-assign path is still "
                  "exercised end-to-end by `test_bool_coercion_in_return` "
                  "and `test_bool_coercion_in_bool_call`.  Re-enable when "
                  "AnnAssign support lands.")
def test_bool_coercion_via_assignment_to_bool_var():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        b: bool = mz(q)
        return b

    ir = _ir(k)
    assert re.search(r"quake\.discriminate.*-> i1", ir), ir


# ---------------------------------------------------------------------------
# Equality / inequality of two handles: each operand discriminates first,
# then the comparison runs at `i1`.
# ---------------------------------------------------------------------------


def test_handle_equality_emits_two_discriminates_and_cmpi():

    @cudaq.kernel
    def k() -> bool:
        qv = cudaq.qvector(2)
        return mz(qv[0]) == mz(qv[1])

    ir = _ir(k)
    n_disc = ir.count("quake.discriminate")
    n_mz = ir.count("quake.mz")
    assert n_disc >= 2 and n_disc == n_mz, (
        f"each mz must contribute exactly one discriminate; "
        f"got {n_disc} disc / {n_mz} mz:\n{ir}")
    assert "arith.cmpi" in ir, ir


def test_handle_inequality_emits_two_discriminates_and_cmpi():

    @cudaq.kernel
    def k() -> bool:
        qv = cudaq.qvector(2)
        return mz(qv[0]) != mz(qv[1])

    ir = _ir(k)
    assert ir.count("quake.discriminate") >= 2
    assert "arith.cmpi" in ir


# ---------------------------------------------------------------------------
# Short-circuit semantics for `and` / `or`: the second operand's
# `quake.discriminate` must be materialized *inside* the short-circuit
# branch, not at the top of the function.  Mirrors the C++ behavior in
# `lib/Frontend/nvqpp/ConvertExpr.cpp` and the structural test in
# `python/tests/mlir/bug_1875.py`.
# ---------------------------------------------------------------------------


def test_and_short_circuits_second_discriminate():

    @cudaq.kernel
    def k():
        qv = cudaq.qvector(2)
        h0 = mz(qv[0])
        h1 = mz(qv[1])
        if h0 and h1:
            x(qv[0])

    ir = _ir(k)
    discr = [m.start() for m in re.finditer(r"quake\.discriminate", ir)]
    ccif = ir.find("cc.if")
    assert len(discr) >= 2, ir
    assert ccif != -1 and discr[1] > ccif, (
        "second discriminate (RHS of `and`) must be inside the short-circuit "
        "branch, not at function entry:\n" + ir)


def test_or_short_circuits_second_discriminate():

    @cudaq.kernel
    def k():
        qv = cudaq.qvector(2)
        h0 = mz(qv[0])
        h1 = mz(qv[1])
        if h0 or h1:
            x(qv[0])

    ir = _ir(k)
    discr = [m.start() for m in re.finditer(r"quake\.discriminate", ir)]
    ccif = ir.find("cc.if")
    assert len(discr) >= 2, ir
    assert ccif != -1 and discr[1] > ccif, (
        "second discriminate (RHS of `or`) must be inside the short-circuit "
        "branch, not at function entry:\n" + ir)


# ---------------------------------------------------------------------------
# `cudaq.to_bools` (vector discrimination) and composition with
# `cudaq.to_integer`.  These are the only sanctioned ways to consume a
# `list[measure_handle]`.
# ---------------------------------------------------------------------------


def test_to_bools_lowers_to_vectorized_discriminate():

    @cudaq.kernel
    def k() -> list[bool]:
        qv = cudaq.qvector(3)
        return cudaq.to_bools(mz(qv))

    ir = _ir(k)
    assert "!cc.stdvec<!cc.measure_handle>" in ir
    assert re.search(
        r"quake\.discriminate.*!cc\.stdvec<!cc\.measure_handle>.*"
        r"!cc\.stdvec<i1>", ir), ir


def test_to_integer_composes_with_to_bools():
    """The spec-mandated composition for converting a register measurement
    into a host-side integer: ``to_integer(to_bools(mz(qv)))``."""

    @cudaq.kernel
    def k() -> int:
        qv = cudaq.qvector(3)
        return cudaq.to_integer(cudaq.to_bools(mz(qv)))

    ir = _ir(k)
    assert "!cc.stdvec<!cc.measure_handle>" in ir, ir
    assert "!cc.stdvec<i1>" in ir, ir


def test_to_integer_rejects_raw_handle_vector():
    """Passing a ``list[measure_handle]`` directly to ``cudaq.to_integer`` is
    rejected; the diagnostic must guide users to the ``to_bools``
    composition (mirrors the C++ bridge's overload-set behavior)."""

    @cudaq.kernel
    def k() -> int:
        qv = cudaq.qvector(3)
        return cudaq.to_integer(mz(qv))

    with pytest.raises(RuntimeError,
                       match=r"to_integer.*cudaq\.to_bools|"
                       r"cudaq\.to_bools.*to_integer"):
        k.compile()


# ---------------------------------------------------------------------------
# Unbound-handle diagnostic: a default-constructed `cudaq.measure_handle()`
# that reaches a `bool`-coercion site without being bound by `mz`/`mx`/`my`
# triggers the spec diagnostic, fired exactly once (the bridge uses a
# placeholder discriminate to keep the AST visitor's value stack balanced
# and suppress an outer "unsupported" diagnostic).
# ---------------------------------------------------------------------------


def test_unbound_handle_in_return_is_diagnosed():

    @cudaq.kernel
    def k() -> bool:
        h = cudaq.measure_handle()
        return h

    with pytest.raises(RuntimeError, match=re.escape(_UNBOUND_DIAG)) as ei:
        k.compile()
    assert str(ei.value).count(_UNBOUND_DIAG) == 1, (
        "unbound-handle diagnostic must fire exactly once (no double-emit "
        "from a residual outer 'unsupported' branch); got:\n" + str(ei.value))


def test_unbound_handle_in_if_test_is_diagnosed():

    @cudaq.kernel
    def k():
        h = cudaq.measure_handle()
        if h:
            pass

    with pytest.raises(RuntimeError, match=re.escape(_UNBOUND_DIAG)):
        k.compile()


def test_unbound_handle_in_bool_call_is_diagnosed():

    @cudaq.kernel
    def k() -> bool:
        return bool(cudaq.measure_handle())

    with pytest.raises(RuntimeError, match=re.escape(_UNBOUND_DIAG)):
        k.compile()


# ---------------------------------------------------------------------------
# Entry-point boundary rule (spec `measure_handle.bs`, Host-Device
# Boundary; mirrors `test/AST-error/measure_handle.cpp`).  `measure_handle`
# and types transitively containing it are forbidden in entry-point kernel
# parameter and return positions.  Diagnostic wording is the spec-canonical
# message shared with the C++ frontend.
# ---------------------------------------------------------------------------


def test_boundary_direct_handle_parameter_is_rejected():

    @cudaq.kernel
    def k(h: cudaq.measure_handle):
        pass

    with pytest.raises(RuntimeError, match=re.escape(_BOUNDARY_DIAG)):
        k.compile()


def test_boundary_direct_handle_return_is_rejected():

    @cudaq.kernel
    def k() -> cudaq.measure_handle:
        q = cudaq.qubit()
        return mz(q)

    with pytest.raises(RuntimeError, match=re.escape(_BOUNDARY_DIAG)):
        k.compile()


def test_boundary_handle_vector_parameter_is_rejected():
    """``list[cudaq.measure_handle]`` in an entry-point parameter exercises
    the recursive ``containsMeasureHandle`` walk via ``cc.stdvec``."""

    @cudaq.kernel
    def k(hs: list[cudaq.measure_handle]):
        pass

    with pytest.raises(RuntimeError, match=re.escape(_BOUNDARY_DIAG)):
        k.compile()


# leave for gdb debugging
if __name__ == "__main__":
    import os
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
