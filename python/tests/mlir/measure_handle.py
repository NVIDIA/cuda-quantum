# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os
import pytest
import cudaq

# ---------------------------------------------------------------------------
# Scalar `mz` / `mx` / `my` on a single qubit produce `!cc.measure_handle`
# and do not inline a `quake.discriminate`.
# ---------------------------------------------------------------------------


def test_scalar_mz_emits_handle_no_discriminate():

    @cudaq.kernel
    def kernel_scalar_mz():
        q = cudaq.qubit()
        h = mz(q)

    print(kernel_scalar_mz)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_scalar_mz
# CHECK-SAME:      attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "h" : (!quake.ref) -> !cc.measure_handle
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }


def test_scalar_mx_emits_handle():

    @cudaq.kernel
    def kernel_scalar_mx():
        q = cudaq.qubit()
        h = mx(q)

    print(kernel_scalar_mx)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_scalar_mx
# CHECK-SAME:      attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.mx %[[VAL_0]] name "h" : (!quake.ref) -> !cc.measure_handle
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }


def test_scalar_my_emits_handle():

    @cudaq.kernel
    def kernel_scalar_my():
        q = cudaq.qubit()
        h = my(q)

    print(kernel_scalar_my)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_scalar_my
# CHECK-SAME:      attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.my %[[VAL_0]] name "h" : (!quake.ref) -> !cc.measure_handle
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }


def test_vector_mz_emits_stdvec_of_handles():

    @cudaq.kernel
    def kernel_vector_mz():
        qv = cudaq.qvector(3)
        hs = mz(qv)

    print(kernel_vector_mz)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_vector_mz
# CHECK-SAME:      attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "hs" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# Bool-coercion sites (sanctioned `bool` contexts: `if`, `while`, `not`,
# explicit `bool(...)`, and `-> bool` returns) lower to a
# `quake.discriminate` adjacent to the consumer.
# ---------------------------------------------------------------------------


def test_bool_coercion_in_if():

    @cudaq.kernel
    def kernel_if():
        q = cudaq.qubit()
        h = mz(q)
        if h:
            x(q)

    print(kernel_if)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_if
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "h" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!cc.measure_handle) -> i1
# CHECK:           cc.if(%[[VAL_2]]) {
# CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           }
# CHECK:           return
# CHECK:         }


def test_bool_coercion_in_while():

    @cudaq.kernel
    def kernel_while():
        q = cudaq.qubit()
        h = mz(q)
        while h:
            x(q)
            h = mz(q)

    print(kernel_while)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_while
# CHECK:           %[[VAL_0:.*]] = quake.mz {{.*}} : (!quake.ref) -> !cc.measure_handle
# CHECK:           cc.loop while
# CHECK:             %[[VAL_1:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
# CHECK:             cc.condition %[[VAL_1]]
# CHECK:           return
# CHECK:         }


def test_bool_coercion_in_not():

    @cudaq.kernel
    def kernel_not():
        q = cudaq.qubit()
        h = mz(q)
        if not h:
            x(q)

    print(kernel_not)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_not
# CHECK:           %[[VAL_0:.*]] = quake.mz {{.*}} : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_1:.*]] = quake.discriminate %[[VAL_0]] : (!cc.measure_handle) -> i1
# CHECK:           %[[VAL_2:.*]] = arith.xori %[[VAL_1]], %{{.*}} : i1
# CHECK:           cc.if(%[[VAL_2]])
# CHECK:           return
# CHECK:         }


def test_bool_coercion_in_bool_call():

    @cudaq.kernel
    def kernel_bool_call() -> bool:
        q = cudaq.qubit()
        return bool(mz(q))

    print(kernel_bool_call)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_bool_call
# CHECK-SAME:      -> i1
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!cc.measure_handle) -> i1
# CHECK:           return %[[VAL_2]] : i1
# CHECK:         }


def test_bool_coercion_in_return():

    @cudaq.kernel
    def kernel_return() -> bool:
        q = cudaq.qubit()
        h = mz(q)
        return h

    print(kernel_return)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_return
# CHECK-SAME:      -> i1
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "h" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!cc.measure_handle) -> i1
# CHECK:           return %[[VAL_2]] : i1
# CHECK:         }

# ---------------------------------------------------------------------------
# Equality / inequality of two handles must materialize one
# `quake.discriminate` per `quake.mz` and compare the resulting `i1`s.
# ---------------------------------------------------------------------------


def test_handle_equality_emits_two_discriminates_and_cmpi():

    @cudaq.kernel
    def kernel_eq() -> bool:
        qv = cudaq.qvector(2)
        return mz(qv[0]) == mz(qv[1])

    print(kernel_eq)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_eq
# CHECK-SAME:      -> i1
# CHECK:           %[[VAL_0:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_1:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_0]] : (!cc.measure_handle) -> i1
# CHECK:           %[[VAL_3:.*]] = quake.discriminate %[[VAL_1]] : (!cc.measure_handle) -> i1
# CHECK:           %[[VAL_4:.*]] = arith.cmpi eq, %[[VAL_2]], %[[VAL_3]] : i1
# CHECK:           return %[[VAL_4]] : i1
# CHECK:         }


def test_handle_inequality_emits_two_discriminates_and_cmpi():

    @cudaq.kernel
    def kernel_ne() -> bool:
        qv = cudaq.qvector(2)
        return mz(qv[0]) != mz(qv[1])

    print(kernel_ne)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_ne
# CHECK-SAME:      -> i1
# CHECK:           %[[VAL_0:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_1:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_0]] : (!cc.measure_handle) -> i1
# CHECK:           %[[VAL_3:.*]] = quake.discriminate %[[VAL_1]] : (!cc.measure_handle) -> i1
# CHECK:           %[[VAL_4:.*]] = arith.cmpi ne, %[[VAL_2]], %[[VAL_3]] : i1
# CHECK:           return %[[VAL_4]] : i1
# CHECK:         }

# ---------------------------------------------------------------------------
# Short-circuit semantics for `and` / `or`: the second handle's
# `quake.discriminate` must materialize lazily inside the short-circuit
# branch (a `cc.if` returning `i1`), not at function entry.
# ---------------------------------------------------------------------------


def test_and_short_circuits_second_discriminate():

    @cudaq.kernel
    def kernel_and():
        qv = cudaq.qvector(2)
        h0 = mz(qv[0])
        h1 = mz(qv[1])
        if h0 and h1:
            x(qv[0])

    print(kernel_and)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_and
# CHECK:           %[[H0:.*]] = quake.mz %{{.*}} name "h0" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[H1:.*]] = quake.mz %{{.*}} name "h1" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[D0:.*]] = quake.discriminate %[[H0]] : (!cc.measure_handle) -> i1
# CHECK:           %{{.*}} = cc.if(%{{.*}}) -> i1 {
# CHECK:             %[[D1:.*]] = quake.discriminate %[[H1]] : (!cc.measure_handle) -> i1
# CHECK:           }
# CHECK:           return
# CHECK:         }


def test_or_short_circuits_second_discriminate():

    @cudaq.kernel
    def kernel_or():
        qv = cudaq.qvector(2)
        h0 = mz(qv[0])
        h1 = mz(qv[1])
        if h0 or h1:
            x(qv[0])

    print(kernel_or)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_or
# CHECK:           %[[H0:.*]] = quake.mz %{{.*}} name "h0" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[H1:.*]] = quake.mz %{{.*}} name "h1" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[D0:.*]] = quake.discriminate %[[H0]] : (!cc.measure_handle) -> i1
# CHECK:           %{{.*}} = cc.if(%[[D0]]) -> i1 {
# CHECK:             %[[D1:.*]] = quake.discriminate %[[H1]] : (!cc.measure_handle) -> i1
# CHECK:           }
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# `cudaq.to_bools` (vector discrimination) and composition with
# `cudaq.to_integer`. The vector discriminate lowers to a single
# `quake.discriminate` taking `!cc.stdvec<!cc.measure_handle>` and
# producing `!cc.stdvec<i1>`.
# ---------------------------------------------------------------------------


def test_to_bools_lowers_to_vectorized_discriminate():

    @cudaq.kernel
    def kernel_to_bools() -> list[bool]:
        qv = cudaq.qvector(3)
        return cudaq.to_bools(mz(qv))

    print(kernel_to_bools)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_to_bools
# CHECK-SAME:      -> !cc.stdvec<i1>
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.stdvec<i1>
# CHECK:           return %{{.*}} : !cc.stdvec<i1>
# CHECK:         }


def test_to_integer_composes_with_to_bools():

    @cudaq.kernel
    def kernel_to_integer() -> int:
        qv = cudaq.qvector(3)
        return cudaq.to_integer(cudaq.to_bools(mz(qv)))

    print(kernel_to_integer)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_to_integer
# CHECK-SAME:      -> i64
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.stdvec<i1>
# CHECK:           cc.stdvec_size %[[VAL_2]] : (!cc.stdvec<i1>) -> i64
# CHECK:           cc.stdvec_data %[[VAL_2]] : (!cc.stdvec<i1>) -> !cc.ptr<!cc.array<i8 x ?>>
# CHECK-NOT:       quake.discriminate
# CHECK:           return %{{.*}} : i64
# CHECK:         }

# ---------------------------------------------------------------------------
# Kernel-builder (`cudaq.make_kernel()`) emission.
# ---------------------------------------------------------------------------


def test_builder_scalar_mz_emits_handle():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc()
    kernel.h(q)
    kernel.mz(q, "scalarHandle")
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME:      () attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "scalarHandle" : (!quake.ref) -> !cc.measure_handle
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }


def test_builder_vector_mz_emits_stdvec_of_handles():
    kernel = cudaq.make_kernel()
    qv = kernel.qalloc(3)
    kernel.h(qv)
    kernel.mz(qv, "vectorHandle")
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME:      () attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "vectorHandle" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }


def test_builder_mx_my_emit_handles():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc()
    kernel.h(q)
    kernel.mx(q, "xHandle")
    kernel.my(q, "yHandle")
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME:      () attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_1:.*]] = quake.mx %[[VAL_0]] name "xHandle" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_2:.*]] = quake.my %[[VAL_0]] name "yHandle" : (!quake.ref) -> !cc.measure_handle
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# List-comprehension pre-allocation of `cudaq.measure_handle()` placeholders.
# ---------------------------------------------------------------------------


def test_listcomp_default_handle_pre_allocation():

    @cudaq.kernel
    def kernel_listcomp_handles():
        qv = cudaq.qvector(4)
        handles = [cudaq.measure_handle() for _ in range(4)]
        for i in range(4):
            handles[i] = mz(qv[i])

    print(kernel_listcomp_handles)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_listcomp_handles
# CHECK-SAME:      attributes {"cudaq-entrypoint"
# CHECK:           %[[VEQ:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[ARR:.*]] = cc.alloca !cc.array<!cc.measure_handle x 4>
# CHECK:           cc.loop
# CHECK:             %[[UNDEF:.*]] = cc.undef !cc.measure_handle
# CHECK:             %[[SLOT:.*]] = cc.compute_ptr %[[ARR]]{{.*}} : (!cc.ptr<!cc.array<!cc.measure_handle x 4>>, i64) -> !cc.ptr<!cc.measure_handle>
# CHECK:             cc.store %[[UNDEF]], %[[SLOT]] : !cc.ptr<!cc.measure_handle>
# CHECK:           cc.loop
# CHECK:             %[[REF:.*]] = quake.extract_ref %[[VEQ]]
# CHECK:             %[[MEAS:.*]] = quake.mz %[[REF]] : (!quake.ref) -> !cc.measure_handle
# CHECK:             cc.store %[[MEAS]]
# CHECK-NOT:       quake.discriminate
# CHECK:           return

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
