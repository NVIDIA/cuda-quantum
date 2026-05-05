# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import re
import pytest
import cudaq

_UNBOUND_DIAG = "discriminating an unbound measurement handle"
_BOUNDARY_DIAG = ("measurement handle cannot cross the host-device boundary; "
                  "entry-point kernels must discriminate first")


@pytest.fixture(autouse=True)
def reset_run_clear():
    cudaq.reset_target()
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def _ir(kernel):
    """Materialize MLIR for ``kernel`` and return it as a string. Triggers
    AST bridge codegen (so any frontend diagnostic fires here) without
    running the kernel on a simulator -- enough for IR-shape and negative
    tests."""
    return str(kernel)


def test_host_construction_raises_runtime_error():
    with pytest.raises(
            RuntimeError,
            match=r"^device-only; usable only inside @cudaq\.kernel$"):
        cudaq.measure_handle()


def test_host_to_bools_raises_runtime_error():
    with pytest.raises(
            RuntimeError,
            match=r"^device-only; usable only inside @cudaq\.kernel$"):
        cudaq.to_bools([])


# Scalar `mz` / `mx` / `my` return a `!cc.measure_handle` and do *not*
# inline a `quake.discriminate`.


def test_scalar_mz_emits_handle_no_discriminate():

    @cudaq.kernel
    def k():
        q = cudaq.qubit()
        h = mz(q)

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
        h = mx(q)

    @cudaq.kernel
    def ky():
        q = cudaq.qubit()
        h = my(q)

    irx, iry = _ir(kx), _ir(ky)
    assert re.search(r"quake\.mx.*-> !cc\.measure_handle", irx), irx
    assert re.search(r"quake\.my.*-> !cc\.measure_handle", iry), iry
    assert "quake.discriminate" not in irx
    assert "quake.discriminate" not in iry


def test_vector_mz_emits_stdvec_of_handles():

    @cudaq.kernel
    def k():
        qv = cudaq.qvector(3)
        hs = mz(qv)

    ir = _ir(k)
    assert "!cc.stdvec<!cc.measure_handle>" in ir
    assert "quake.discriminate" not in ir


# ---------------------------------------------------------------------------
# Bool-coercion sites lower to `quake.discriminate`.
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


def test_bool_coercion_in_bool_call():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        return bool(mz(q))

    ir = _ir(k)
    assert re.search(r"quake\.discriminate.*-> i1", ir), ir


def test_bool_coercion_in_return():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        h = mz(q)
        return h

    ir = _ir(k)
    assert re.search(r"quake\.discriminate.*-> i1", ir), ir


# ---------------------------------------------------------------------------
# Equality / inequality of two handles
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
# Short-circuit semantics for `and` / `or`
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
# `cudaq.to_integer`.
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

    @cudaq.kernel
    def k() -> int:
        qv = cudaq.qvector(3)
        return cudaq.to_integer(cudaq.to_bools(mz(qv)))

    ir = _ir(k)
    assert "!cc.stdvec<!cc.measure_handle>" in ir, ir
    assert "!cc.stdvec<i1>" in ir, ir


def test_to_integer_rejects_raw_handle_vector():

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
# Entry-point boundary rule
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
