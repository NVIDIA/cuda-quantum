# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import re
from typing import Callable

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


# ---------------------------------------------------------------------------
# `cudaq.to_integer` rejects a raw `!cc.stdvec<!cc.measure_handle>`
# without an intervening `cudaq.to_bools`.
# ---------------------------------------------------------------------------


def test_to_integer_rejects_raw_handle_vector():

    @cudaq.kernel
    def k() -> int:
        qv = cudaq.qvector(3)
        return cudaq.to_integer(mz(qv))

    with pytest.raises(RuntimeError,
                       match=re.escape(
                           "`cudaq.to_integer` does not accept a "
                           "`list[cudaq.measure_handle]`; compose with "
                           "`cudaq.to_integer(cudaq.to_bools(handles))`")):
        k.compile()


# ---------------------------------------------------------------------------
# `cudaq.measure_handle()` and `cudaq.to_bools(...)` argument-shape diagnostics
# ---------------------------------------------------------------------------


def test_measure_handle_constructor_with_argument_is_rejected():

    @cudaq.kernel
    def k() -> bool:
        h = cudaq.measure_handle(1)
        return h

    with pytest.raises(RuntimeError,
                       match=re.escape("cudaq.measure_handle() takes no "
                                       "arguments")):
        k.compile()


def test_to_bools_with_no_argument_is_rejected():

    @cudaq.kernel
    def k() -> int:
        return cudaq.to_integer(cudaq.to_bools())

    with pytest.raises(
            RuntimeError,
            match=re.escape("cudaq.to_bools expects a single argument")):
        k.compile()


def test_to_bools_with_scalar_handle_is_rejected():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        h = mz(q)
        bs = cudaq.to_bools(h)
        return bs[0]

    with pytest.raises(
            RuntimeError,
            match=re.escape(
                "cudaq.to_bools expects a list[cudaq.measure_handle] argument")
    ):
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


def test_unbound_handle_in_while_test_is_diagnosed():

    @cudaq.kernel
    def k():
        h = cudaq.measure_handle()
        while h:
            pass

    with pytest.raises(RuntimeError, match=re.escape(_UNBOUND_DIAG)):
        k.compile()


def test_unbound_handle_in_unary_not_is_diagnosed():

    @cudaq.kernel
    def k() -> bool:
        return not cudaq.measure_handle()

    with pytest.raises(RuntimeError, match=re.escape(_UNBOUND_DIAG)):
        k.compile()


def test_unbound_handle_in_boolop_and_is_diagnosed():
    # The lhs of `and` is the spec-listed bool-coercion site; the unbound
    # diagnostic must fire on it before short-circuit evaluation considers
    # the rhs.

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        return cudaq.measure_handle() and mz(q)

    with pytest.raises(RuntimeError, match=re.escape(_UNBOUND_DIAG)):
        k.compile()


def test_unbound_handle_in_compare_eq_is_diagnosed():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        return cudaq.measure_handle() == mz(q)

    with pytest.raises(RuntimeError, match=re.escape(_UNBOUND_DIAG)):
        k.compile()


# ---------------------------------------------------------------------------
# List-comprehension pre-allocation of `cudaq.measure_handle()` placeholders
# ---------------------------------------------------------------------------


def test_listcomp_default_handles_overwritten_by_mz_compiles():

    @cudaq.kernel
    def k():
        qv = cudaq.qvector(3)
        handles = [cudaq.measure_handle() for _ in range(3)]
        for i in range(3):
            handles[i] = mz(qv[i])

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


def test_boundary_handle_vector_return_is_rejected():

    @cudaq.kernel
    def k() -> list[cudaq.measure_handle]:
        qv = cudaq.qvector(2)
        return mz(qv)

    with pytest.raises(RuntimeError, match=re.escape(_BOUNDARY_DIAG)):
        k.compile()


def test_boundary_handle_in_tuple_parameter_is_rejected():

    @cudaq.kernel
    def k(t: tuple[cudaq.measure_handle, bool]):
        pass

    with pytest.raises(RuntimeError, match=re.escape(_BOUNDARY_DIAG)):
        k.compile()


def test_boundary_handle_in_callable_parameter_is_admissible():

    @cudaq.kernel
    def k(fn: Callable[[cudaq.measure_handle], None]):
        pass

    k.compile()


def test_boundary_make_kernel_handle_arg_is_rejected():
    # `cudaq.make_kernel(...)` constructs an entry-point FuncOp directly
    # without going through the AST-bridge boundary check. The boundary rule
    # is enforced inside `PyKernel.__init__` so this path cannot be sidestepped.
    with pytest.raises(RuntimeError, match=re.escape(_BOUNDARY_DIAG)):
        cudaq.make_kernel(cudaq.measure_handle)


# ---------------------------------------------------------------------------
# Scalar arithmetic coercion
# ---------------------------------------------------------------------------


def test_return_handle_to_int_kernel_discriminates():

    @cudaq.kernel
    def k() -> int:
        q = cudaq.qubit()
        x(q)
        return mz(q)

    results = cudaq.run(k, shots_count=1)
    assert len(results) == 1
    assert results[0] == 1


def test_return_handle_to_float_kernel_discriminates():

    @cudaq.kernel
    def k() -> float:
        q = cudaq.qubit()
        x(q)
        return mz(q)

    results = cudaq.run(k, shots_count=1)
    assert len(results) == 1
    assert results[0] == 1.0


def test_int_handle_in_arithmetic_promotes_through_bool():

    @cudaq.kernel
    def k() -> int:
        q = cudaq.qubit()
        x(q)
        return (mz(q) + 1)

    results = cudaq.run(k, shots_count=1)
    assert len(results) == 1
    assert results[0] == 2


# leave for gdb debugging
if __name__ == "__main__":
    import os
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
