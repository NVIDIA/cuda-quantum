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


def assert_unbound_once(kernel, source):
    with pytest.raises(RuntimeError, match=re.escape(_UNBOUND_DIAG)) as error:
        kernel.compile()
    message = str(error.value)
    assert message.count(_UNBOUND_DIAG) == 1
    assert f"(offending source -> {source}" in message


def test_host_construction_raises_runtime_error():
    with pytest.raises(RuntimeError,
                       match="can be used only in CUDA-Q kernels"):
        cudaq.measure_handle()


def test_host_to_bools_raises_runtime_error():
    with pytest.raises(RuntimeError,
                       match="can be used only in CUDA-Q kernels"):
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
        mh = cudaq.measure_handle()
        return mh

    assert_unbound_once(k, "return mh")


def test_unbound_handle_in_if_test_is_diagnosed():

    @cudaq.kernel
    def k():
        mh = cudaq.measure_handle()
        if mh:
            pass

    assert_unbound_once(k, "if mh:")


def test_unbound_handle_in_bool_call_is_diagnosed():

    @cudaq.kernel
    def k() -> bool:
        return bool(cudaq.measure_handle())

    assert_unbound_once(k, "bool(cudaq.measure_handle())")


def test_unbound_handle_in_while_test_is_diagnosed():

    @cudaq.kernel
    def k():
        mh = cudaq.measure_handle()
        while mh:
            pass

    assert_unbound_once(k, "while mh:")


def test_unbound_handle_in_unary_not_is_diagnosed():

    @cudaq.kernel
    def k() -> bool:
        return not cudaq.measure_handle()

    assert_unbound_once(k, "not cudaq.measure_handle()")


def test_unbound_handle_in_boolop_and_is_diagnosed():
    # The lhs of `and` is the spec-listed bool-coercion site; the unbound
    # diagnostic must fire on it before short-circuit evaluation considers
    # the rhs.

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        return cudaq.measure_handle() and mz(q)

    assert_unbound_once(k, "cudaq.measure_handle() and mz(q)")


def test_unbound_handle_in_boolop_or_is_diagnosed():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        return cudaq.measure_handle() or mz(q)

    assert_unbound_once(k, "cudaq.measure_handle() or mz(q)")


def test_unbound_handle_in_compare_eq_is_diagnosed():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        return cudaq.measure_handle() == mz(q)

    assert_unbound_once(k, "cudaq.measure_handle() == mz(q)")


def test_unbound_handle_in_arithmetic_is_diagnosed():

    @cudaq.kernel
    def k() -> int:
        return cudaq.measure_handle() + 1

    assert_unbound_once(k, "cudaq.measure_handle() + 1")


def test_scalar_measurement_results_and_bind_before_use_compile():

    @cudaq.kernel
    def kx() -> bool:
        q = cudaq.qubit()
        return bool(mx(q))

    @cudaq.kernel
    def ky() -> bool:
        q = cudaq.qubit()
        return bool(my(q))

    @cudaq.kernel
    def kz() -> bool:
        q = cudaq.qubit()
        return bool(mz(q))

    @cudaq.kernel
    def bind_before_use() -> bool:
        q = cudaq.qubit()
        mh = cudaq.measure_handle()
        mh = mz(q)
        return bool(mh)

    for kernel in (kx, ky, kz, bind_before_use):
        kernel.compile()


def test_scalar_measure_handle_reassignment_to_default_is_diagnosed():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        mh = mz(q)
        unused = bool(mh)
        mh = cudaq.measure_handle()
        return bool(mh)

    assert_unbound_once(k, "bool(mh)")


def test_scalar_measure_handle_use_before_bind_is_diagnosed():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        mh = cudaq.measure_handle()
        use_before_bind = bool(mh)
        mh = mz(q)
        return use_before_bind

    assert_unbound_once(k, "bool(mh)")


def test_scalar_measure_handle_cache_is_per_alloca():

    @cudaq.kernel
    def k() -> bool:
        q = cudaq.qubit()
        bound_mh = mz(q)
        unused = bool(bound_mh)
        unbound_mh = cudaq.measure_handle()
        return bool(unbound_mh)

    assert_unbound_once(k, "bool(unbound_mh)")


def test_scalar_measure_handle_uncertain_provenance_compiles():

    @cudaq.kernel
    def one_branch(condition: bool) -> bool:
        q = cudaq.qubit()
        mh = cudaq.measure_handle()
        if condition:
            mh = mz(q)
        return bool(mh)

    @cudaq.kernel
    def both_branches(condition: bool) -> bool:
        q = cudaq.qubit()
        mh = cudaq.measure_handle()
        if condition:
            mh = mx(q)
        else:
            mh = my(q)
        return bool(mh)

    @cudaq.kernel
    def loop_carried(count: int) -> bool:
        q = cudaq.qubit()
        mh = cudaq.measure_handle()
        for _ in range(count):
            mh = mz(q)
        return bool(mh)

    @cudaq.kernel
    def alias_branch(condition: bool) -> bool:
        q = cudaq.qubit()
        mh = cudaq.measure_handle()
        mh_alias = mh
        if condition:
            mh_alias = mz(q)
        return bool(mh)

    for kernel in (one_branch, both_branches, loop_carried, alias_branch):
        kernel.compile()


def test_scalar_measure_handle_function_values_compile():

    @cudaq.kernel
    def produce_mh(q: cudaq.qubit) -> cudaq.measure_handle:
        return mz(q)

    @cudaq.kernel
    def consume_mh(q: cudaq.qubit, mh: cudaq.measure_handle) -> bool:
        return bool(mh)

    @cudaq.kernel
    def call_result() -> bool:
        q = cudaq.qubit()
        mh = produce_mh(q)
        return bool(mh)

    consume_mh.compile()
    call_result.compile()


def test_repeated_bound_handle_coercions_scan_once(monkeypatch):
    from cudaq.kernel.ast_bridge import PyASTBridge

    scan_name = "_PyASTBridge__scanMeasureHandleAllocaStores"
    lookup_name = "_PyASTBridge__isProvablyUnboundHandleSource"
    original_scan = getattr(PyASTBridge, scan_name)
    original_lookup = getattr(PyASTBridge, lookup_name)
    calls = {"lookup": 0, "scan": 0}

    def counting_scan(self, storage):
        calls["scan"] += 1
        return original_scan(self, storage)

    def counting_lookup(self, value):
        calls["lookup"] += 1
        return original_lookup(self, value)

    monkeypatch.setattr(PyASTBridge, scan_name, counting_scan)
    monkeypatch.setattr(PyASTBridge, lookup_name, counting_lookup)

    @cudaq.kernel
    def k():
        q = cudaq.qubit()
        mh = mz(q)
        mh_alias = mh
        value_0 = bool(mh)
        value_1 = bool(mh_alias)
        value_2 = bool(mh)
        value_3 = bool(mh_alias)
        value_4 = bool(mh)
        value_5 = bool(mh_alias)
        value_6 = bool(mh)
        value_7 = bool(mh_alias)

    k.compile()
    assert calls == {"lookup": 8, "scan": 1}


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


def test_boundary_handle_vector_via_alias_is_rejected():
    import cudaq as cq

    @cq.kernel
    def k(hs: list[cq.measure_handle]):
        pass

    with pytest.raises(RuntimeError, match=re.escape(_BOUNDARY_DIAG)):
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


def test_handle_vector_issue_4527():

    @cudaq.kernel
    def k() -> int:
        qv = cudaq.qvector(2)
        combined = [cudaq.measure_handle() for _ in range(2)]
        x(qv[0])
        h0 = mz(qv[0])
        h1 = mz(qv[1])
        combined[0] = h0
        combined[1] = h1
        first = cudaq.to_integer(cudaq.to_bools(combined))
        x(qv[0])
        h2 = mz(qv[0])
        h3 = mz(qv[1])
        combined[0] = h2
        combined[1] = h3
        second = cudaq.to_integer(cudaq.to_bools(combined))
        return first * 16 + second

    results = cudaq.run(k, shots_count=1)
    assert len(results) == 1
    assert results[0] == 16


# ---------------------------------------------------------------------------
# Cross-scope reassignment of a `list[measure_handle]`.
# ---------------------------------------------------------------------------


def test_handle_vector_cross_round_reassignment_in_loop():

    @cudaq.kernel
    def k() -> list[bool]:
        qv = cudaq.qvector(3)
        x(qv)
        mvec = mz(qv)
        for _ in range(2):
            m_new = mz(qv)
            mvec = m_new
        return mvec

    results = cudaq.run(k, shots_count=1)
    assert len(results) == 1
    assert results[0] == [True, True, True]


def test_handle_vector_reassignment_in_conditional():

    @cudaq.kernel
    def k() -> list[bool]:
        qv = cudaq.qvector(2)
        mvec = mz(qv)
        if True:
            x(qv)
            mvec = mz(qv)
        return mvec

    results = cudaq.run(k, shots_count=1)
    assert len(results) == 1
    assert results[0] == [True, True]


def test_discriminated_bool_cross_scope_reassignment():

    @cudaq.kernel
    def k() -> bool:
        qs = cudaq.qvector(2)
        b = bool(mz(qs[0]))
        if True:
            x(qs[1])
            b = bool(mz(qs[1]))
        return b

    results = cudaq.run(k, shots_count=1)
    assert len(results) == 1
    assert results[0] == True


# leave for gdb debugging
if __name__ == "__main__":
    import os
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
