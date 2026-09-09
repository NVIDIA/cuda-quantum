/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

// Regression test: a `cudaq::range`-based `for` loop threads its induction
// variable as loop-carried SSA state (a `cc.loop` result), unlike a plain
// `for`/`while` loop which keeps its induction variable in a `cc.alloca`
// (and so has zero loop-carried results). `break`/`continue` inside such a
// loop must forward that loop-carried state to `cc.unwind_break`/
// `cc.unwind_continue`, or the arity check in
// UnwindBreakOp::verify()/UnwindContinueOp::verify() rejects the module.

#include <cudaq.h>

__qpu__ std::int64_t test_break(std::int64_t n) {
  std::int64_t total = 0;
  for (auto i : cudaq::range(3)) {
    if (i == n)
      break;
    total += 1;
  }
  return total;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test_break.
// CHECK-SAME:      (%[[VAL_0:.*]]: i64) -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %{{.*}}) -> (i64)) {
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i64):
// CHECK:             cc.if(%{{.*}}) {
// CHECK:               cc.unwind_break %[[VAL_7]] : i64
// CHECK:             }
// CHECK:             cc.continue %[[VAL_7]] : i64
// CHECK:           } step {
// CHECK:           } {invariant}
// CHECK:           return %{{.*}} : i64
// CHECK:         }
// clang-format on

__qpu__ std::int64_t test_continue(std::int64_t n) {
  std::int64_t total = 0;
  for (auto i : cudaq::range(5)) {
    if (i == n)
      continue;
    total += i;
  }
  return total;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test_continue.
// CHECK-SAME:      (%[[VAL_0:.*]]: i64) -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %{{.*}}) -> (i64)) {
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i64):
// CHECK:             cc.if(%{{.*}}) {
// CHECK:               cc.unwind_continue %[[VAL_7]] : i64
// CHECK:             }
// CHECK:           } step {
// CHECK:           } {invariant}
// CHECK:           return %{{.*}} : i64
// CHECK:         }
// clang-format on

// Nested loops: outer classical `for` (zero loop-carried results), inner
// `cudaq::range` `for` (one loop-carried result). `break` in the inner loop
// must target the inner loop's arity, not the outer loop's.
__qpu__ std::int64_t test_nested(std::int64_t n) {
  std::int64_t total = 0;
  for (int i = 0; i < 3; i++) {
    for (auto j : cudaq::range(5)) {
      if (j == n)
        break;
      total += 1;
    }
  }
  return total;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test_nested.
// CHECK-SAME:      (%[[VAL_0:.*]]: i64) -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           cc.loop while {
// CHECK:           } do {
// CHECK:             %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %{{.*}}) -> (i64)) {
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_11:.*]]: i64):
// CHECK:               cc.if(%{{.*}}) {
// CHECK:                 cc.unwind_break %[[VAL_11]] : i64
// CHECK:               }
// CHECK:               cc.continue %[[VAL_11]] : i64
// CHECK:             } step {
// CHECK:             } {invariant}
// CHECK:           } step {
// CHECK:           }
// CHECK:           return %{{.*}} : i64
// CHECK:         }
// clang-format on
