/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

// A comma expression evaluates its lhs for side-effects only and takes its
// value from the rhs.

// Two induction variables, stepped with a comma.
struct comma_in_for_loop {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    for (int i = 0, j = 3; i < 4; i++, j--)
      h(q[0]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__comma_in_for_loop() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_5:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_3]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:             %[[VAL_6:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_7:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_1]] : i32
// CHECK:               cc.condition %[[VAL_8]]
// CHECK:             } do {
// CHECK:               %[[VAL_9:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.h %[[VAL_9]] : (!quake.ref) -> ()
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_10:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_0]] : i32
// CHECK:               cc.store %[[VAL_11]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               %[[VAL_12:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:               %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_0]] : i32
// CHECK:               cc.store %[[VAL_13]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

// The value is the rhs, `b + 3`.
struct comma_takes_rhs_value {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    int a = 0, b = 0;
    a = (b = 2, b + 3);
    if (a == 5)
      x(q[1]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__comma_takes_rhs_value() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_1]] : i32
// CHECK:           cc.store %[[VAL_8]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_0]] : i32
// CHECK:           cc.if(%[[VAL_10]]) {
// CHECK:             %[[VAL_11:.*]] = quake.extract_ref %[[VAL_4]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x %[[VAL_11]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

// An lvalue rhs used as a value the load.
struct comma_with_lvalue_rhs {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    int a = 0, b = 0;
    a = (b = 2, b);
    if (a == 2)
      x(q[1]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__comma_with_lvalue_rhs() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_5]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_1]] : i32
// CHECK:           cc.if(%[[VAL_7]]) {
// CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_2]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x %[[VAL_8]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

// An lvalue rhs assigned through the address.
struct comma_result_is_assignable {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    int a = 0, b = 0;
    (a, b) = 5;
    if (b == 5)
      x(q[1]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__comma_result_is_assignable() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_0]] : i32
// CHECK:           cc.if(%[[VAL_6]]) {
// CHECK:             %[[VAL_7:.*]] = quake.extract_ref %[[VAL_2]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x %[[VAL_7]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

// Left associative (a chain yields its last operand).
struct comma_chain {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    int a = 0, b = 0, c = 0;
    a = (b = 1, c = 2, b + c);
    if (a == 3)
      x(q[1]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__comma_chain() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           cc.store %[[VAL_10]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_11]], %[[VAL_0]] : i32
// CHECK:           cc.if(%[[VAL_12]]) {
// CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_4]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x %[[VAL_13]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

// Lowest precedence: `a = 1, a = 2` is `(a = 1), (a = 2)`.
struct comma_binds_looser_than_assign {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    int a = 0;
    a = 1, a = 2;
    if (a == 2)
      x(q[1]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__comma_binds_looser_than_assign() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           cc.if(%[[VAL_6]]) {
// CHECK:             %[[VAL_7:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x %[[VAL_7]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

// Parenthesized, the comma is the assignment's rhs, so `a` never sees 1.
struct comma_as_assign_operand {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    int a = 0, b = 0;
    a = (b = 1, 2);
    if (a == 2)
      x(q[1]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__comma_as_assign_operand() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_2]] : i32
// CHECK:           cc.if(%[[VAL_7]]) {
// CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x %[[VAL_8]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on
