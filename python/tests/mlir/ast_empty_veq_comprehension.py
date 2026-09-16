# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

# Regression coverage for a qubit list-comprehension whose filter never
# matches, e.g. `[qs[i] for i in range(4) if i > 10]`. There is no valid
# `!quake.veq<0>` to build in these cases (a genuinely empty veq is not a
# meaningful "vector of zero qubits" to operate on), so the bridge must produce
# `cc.poison` on that path instead and never lets it flow into a real
# quantum operation. The compiler is free to eliminate the poison value and all
# operations that dataflow from it.

import cudaq


def test_empty_comprehension_len():

    @cudaq.kernel
    def kernel1() -> int:
        qs = cudaq.qvector(4)
        return len([qs[i] for i in range(4) if i > 10])

    # a filter that never matches must report length 0, not garbage
    assert kernel1() == 0
    print(kernel1)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1..
# CHECK-SAME: () -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_6:.*]] = cc.alloca !cc.array<i64 x 4>
# CHECK:           %[[VAL_7:.*]]:2 = cc.loop while ((%[[VAL_8:.*]] = %{{.*}}, %[[VAL_9:.*]] = %{{.*}}) -> (i64, i64)) {
# CHECK:           } do {
# CHECK:             %[[VAL_13:.*]] = cc.if(%{{.*}}) -> i64 {
# CHECK:               %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_6]][%{{.*}}] : (!cc.ptr<!cc.array<i64 x 4>>, i64) -> !cc.ptr<i64>
# CHECK:               cc.store %{{.*}}, %[[VAL_14]] : !cc.ptr<i64>
# CHECK:             } else {
# CHECK:             }
# CHECK:           } step {
# CHECK:           %[[VAL_18:.*]] = arith.cmpi sgt, %[[VAL_7]]#1, %{{.*}} : i64
# CHECK:           %[[VAL_19:.*]] = cc.if(%[[VAL_18]]) -> !quake.veq<?> {
# CHECK:             %[[VAL_21:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%{{.*}}] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             %[[VAL_22:.*]] = quake.concat %[[VAL_21]] : (!quake.ref) -> !quake.veq<1>
# CHECK:             %[[VAL_23:.*]] = quake.relax_size %[[VAL_22]] : (!quake.veq<1>) -> !quake.veq<?>
# CHECK:             %[[VAL_27:.*]]:2 = cc.loop while ((%[[VAL_28:.*]] = %{{.*}}, %[[VAL_29:.*]] = %[[VAL_23]]) -> (i64, !quake.veq<?>)) {
# CHECK:             } do {
# CHECK:               %[[VAL_31:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%{{.*}}] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               %[[VAL_32:.*]] = quake.concat %{{.*}}, %[[VAL_31]] : (!quake.veq<?>, !quake.ref) -> !quake.veq<?>
# CHECK:             } step {
# CHECK:             }
# CHECK:             cc.continue %{{.*}} : !quake.veq<?>
# CHECK:           } else {
# CHECK:             %[[VAL_35:.*]] = cc.poison !quake.veq<0>
# CHECK:             %[[VAL_36:.*]] = quake.relax_size %[[VAL_35]] : (!quake.veq<0>) -> !quake.veq<?>
# CHECK:             cc.continue %[[VAL_36]] : !quake.veq<?>
# CHECK:           }
# CHECK:           %[[VAL_38:.*]] = quake.veq_size %[[VAL_19]] : (!quake.veq<?>) -> i64
# CHECK:           quake.dealloc %[[VAL_5]] : !quake.veq<4>
# CHECK:           return %[[VAL_38]] : i64
# CHECK:         }


def test_empty_comprehension_broadcast_target():

    @cudaq.kernel
    def kernel2():
        qs = cudaq.qvector(4)
        x([qs[i] for i in range(4) if i > 10])

    # broadcasting a gate over zero targets must be a no-op
    counts = cudaq.sample(kernel2)
    assert len(counts) == 1 and '0000' in counts
    print(kernel2)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2..
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_18:.*]] = cc.if(%{{.*}}) -> !quake.veq<?> {
# CHECK:           } else {
# CHECK:             %[[VAL_34:.*]] = cc.poison !quake.veq<0>
# CHECK:             %[[VAL_35:.*]] = quake.relax_size %[[VAL_34]] : (!quake.veq<0>) -> !quake.veq<?>
# CHECK:             cc.continue %[[VAL_35]] : !quake.veq<?>
# CHECK:           }
# CHECK:           %[[VAL_37:.*]] = quake.veq_size %[[VAL_18]] : (!quake.veq<?>) -> i64
# CHECK:           cc.loop while
# CHECK-SAME:      -> (i64)) {
# CHECK:             %[[VAL_40:.*]] = arith.cmpi slt, %{{.*}}, %[[VAL_37]] : i64
# CHECK:           } do {
# CHECK:             %[[VAL_42:.*]] = quake.extract_ref %[[VAL_18]]{{\[}}%{{.*}}] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_42]] : (!quake.ref) -> ()
# CHECK:           } step {
# CHECK:           } {invariant}
# CHECK:           quake.dealloc %[[VAL_5]] : !quake.veq<4>
# CHECK:           return
# CHECK:         }


def test_empty_comprehension_controls():

    @cudaq.kernel
    def kernel3():
        qs = cudaq.qvector(4)
        target = cudaq.qubit()
        x.ctrl([qs[i] for i in range(4) if i > 10], target)

    # a gate controlled by zero qubits must act unconditionally
    counts = cudaq.sample(kernel3)
    assert len(counts) == 1 and '00001' in counts
    print(kernel3)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3..
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_19:.*]] = cc.if(%{{.*}}) -> !quake.veq<?> {
# CHECK:           } else {
# CHECK:             %[[VAL_35:.*]] = cc.poison !quake.veq<0>
# CHECK:             %[[VAL_36:.*]] = quake.relax_size %[[VAL_35]] : (!quake.veq<0>) -> !quake.veq<?>
# CHECK:             cc.continue %[[VAL_36]] : !quake.veq<?>
# CHECK:           }
# CHECK:           quake.x {{\[}}%[[VAL_19]]] %[[VAL_6]] : (!quake.veq<?>, !quake.ref) -> ()
# CHECK-DAG:       quake.dealloc %[[VAL_5]] : !quake.veq<4>
# CHECK-DAG:       quake.dealloc %[[VAL_6]] : !quake.ref
# CHECK:           return
# CHECK:         }
