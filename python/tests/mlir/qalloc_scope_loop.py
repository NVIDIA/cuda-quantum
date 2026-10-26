# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_qalloc_freed_each_for_iteration():
    """A qubit allocated in a `for` body is freed at the end of each pass."""

    @cudaq.kernel
    def for_kernel():
        r = cudaq.qubit()
        for i in range(3):
            q = cudaq.qvector(2)
            h(q[0])
            x.ctrl(q[0], r)
        mz(r)

    print(for_kernel)


def test_qalloc_freed_each_while_iteration():
    """A qubit allocated in a `while` body is freed at the end of each pass."""

    @cudaq.kernel
    def while_kernel(n: int):
        r = cudaq.qubit()
        i = 0
        while i < n:
            q = cudaq.qvector(2)
            h(q[0])
            x.ctrl(q[0], r)
            i = i + 1
        mz(r)

    print(while_kernel)


def test_qalloc_freed_at_end_of_loop_else():
    """A qubit allocated in a `while` loop's `else` block is freed when it
    ends."""

    @cudaq.kernel
    def else_kernel(n: int):
        r = cudaq.qubit()
        i = 0
        while i < n:
            h(r)
            i = i + 1
        else:
            q = cudaq.qvector(2)
            h(q[0])
            x.ctrl(q[0], r)
        mz(r)

    print(else_kernel)


def test_qalloc_freed_at_end_of_for_else():
    """A qubit allocated in a `for` loop's `else` block is freed when it ends."""

    @cudaq.kernel
    def for_else_kernel(n: int):
        r = cudaq.qubit()
        for i in range(n):
            h(r)
        else:
            q = cudaq.qvector(2)
            h(q[0])
            x.ctrl(q[0], r)
        mz(r)

    print(for_else_kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__for_kernel..
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           cc.loop while
# CHECK:           } do {
# CHECK:             cc.scope {
# CHECK:               %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
# CHECK:               quake.dealloc %[[VAL_1]] : !quake.veq<2>
# CHECK:             }
# CHECK:           } step {
# CHECK:           quake.dealloc %[[VAL_0]] : !quake.ref

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__while_kernel..
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           cc.loop while
# CHECK:           } do {
# CHECK:             %[[VAL_1:.*]] = cc.scope -> (i64) {
# CHECK:               %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
# CHECK:               quake.dealloc %[[VAL_2]] : !quake.veq<2>
# CHECK:             }
# CHECK:           } step {
# CHECK:           quake.dealloc %[[VAL_0]] : !quake.ref

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__else_kernel..
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           cc.loop while
# CHECK:           } else {
# CHECK:             cc.scope {
# CHECK:               %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
# CHECK:               quake.dealloc %[[VAL_1]] : !quake.veq<2>
# CHECK:             }
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_0]] : !quake.ref

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__for_else_kernel..
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           cc.loop while
# CHECK:           } else {
# CHECK:             cc.scope {
# CHECK:               %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
# CHECK:               quake.dealloc %[[VAL_1]] : !quake.veq<2>
# CHECK:             }
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_0]] : !quake.ref
