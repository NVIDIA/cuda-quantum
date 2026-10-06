# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

# Regression coverage for `for i in range(...)` and
# `for i, v in enumerate(range(...))`: the induction variable (and, for
# `enumerate`, the paired index) must be driven directly off the loop's
# own loop-carried arguments. There must be no intermediary `cc.alloca`
# buffer of consecutive/derived integers materialized just to iterate over
# it -- that buffer serves no purpose the loop bounds/step don't already
# provide, and it makes the generated IR harder for downstream passes to
# reason about. This also exercises `break`/`continue` threading their
# loop-carried arguments correctly through such a loop.

import cudaq


def test_range_with_step():

    @cudaq.kernel
    def kernel_range() -> int:
        total = 0
        for i in range(2, 9, 3):
            total += i
        return total

    assert kernel_range() == 15
    print(kernel_range)


def test_enumerate_range_with_step():

    @cudaq.kernel
    def kernel_enumerate() -> int:
        total = 0
        for idx, val in enumerate(range(5, 20, 5)):
            total += idx * val
        return total

    assert kernel_enumerate() == 40
    print(kernel_enumerate)


def test_range_break_continue():

    @cudaq.kernel
    def kernel_break_continue(n: int) -> int:
        total = 0
        for i in range(10):
            if i == n:
                break
            if i % 2 == 0:
                continue
            total += i
        return total

    assert kernel_break_continue(5) == 4
    print(kernel_break_continue)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_range..
# CHECK-SAME: () -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-NOT:       cc.alloca !cc.array
# CHECK:           %[[VAL_4:.*]]:2 = cc.loop while ((%[[VAL_5:.*]] = %{{.*}}, %[[VAL_6:.*]] = %{{.*}}) -> (i64, i64)) {
# CHECK:             %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %{{.*}} : i64
# CHECK:             cc.condition %[[VAL_7]](%[[VAL_5]], %[[VAL_6]] : i64, i64)
# CHECK:           } do {
# CHECK:             %[[VAL_9:.*]] = arith.muli %[[VAL_5]], %{{.*}} : i64
# CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %{{.*}} : i64
# CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_6]], %[[VAL_10]] : i64
# CHECK:           } step {
# CHECK:           } {normalized}
# CHECK:           return %{{.*}} : i64
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_enumerate..
# CHECK-SAME: () -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-NOT:       cc.alloca !cc.array
# CHECK:           %[[VAL_4:.*]]:2 = cc.loop while ((%[[VAL_5:.*]] = %{{.*}}, %[[VAL_6:.*]] = %{{.*}}) -> (i64, i64)) {
# CHECK:           } do {
# CHECK:             %[[VAL_9:.*]] = arith.muli %[[VAL_5]], %{{.*}} : i64
# CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %{{.*}} : i64
# CHECK:             %[[VAL_11:.*]] = arith.divsi %[[VAL_9]], %{{.*}} : i64
# CHECK:             %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[VAL_10]] : i64
# CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_6]], %[[VAL_12]] : i64
# CHECK:           } step {
# CHECK:           } {normalized}
# CHECK:           return %{{.*}} : i64
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_break_continue..
# CHECK-SAME: (%[[VAL_0:.*]]: i64) -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-NOT:       cc.alloca !cc.array
# CHECK:           %[[VAL_4:.*]]:2 = cc.loop while ((%[[VAL_5:.*]] = %{{.*}}, %[[VAL_6:.*]] = %{{.*}}) -> (i64, i64)) {
# CHECK:           } do {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_0]] : i64
# CHECK:             cf.cond_br %[[VAL_8]], ^bb1, ^bb2
# CHECK:           ^bb1:
# CHECK:             cc.break %[[VAL_5]], %[[VAL_6]] : i64, i64
# CHECK:           ^bb2:
# CHECK:             %[[VAL_9:.*]] = arith.remui %[[VAL_5]], %{{.*}} : i64
# CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %{{.*}} : i64
# CHECK:             cf.cond_br %[[VAL_10]], ^bb3, ^bb4
# CHECK:           ^bb3:
# CHECK:             cc.continue %[[VAL_5]], %[[VAL_6]] : i64, i64
# CHECK:           ^bb4:
# CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_6]], %[[VAL_5]] : i64
# CHECK:             cc.continue %[[VAL_5]], %[[VAL_11]] : i64, i64
# CHECK:           } step {
# CHECK:           }
# CHECK:           return %{{.*}} : i64
# CHECK:         }
