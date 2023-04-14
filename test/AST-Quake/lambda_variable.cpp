/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s
// XFAIL: *

// Test lambdas that are created within kernels and passed to user-defined
// kernels as an argument. Since the lambda is an argument, it is not possible
// or sufficient to try to inline a template specialization.

#include <cudaq.h>

struct test3_callee {
  void operator()(std::function<void(cudaq::qubit &)> &&callback,
                  cudaq::qreg<> &s) __qpu__ {
    callback(s[0]);
    callback(s[1]);
  }
};

struct test3_caller {
  void operator()() __qpu__ {
    cudaq::qreg q(2);
    test3_callee{}(
        [](cudaq::qubit &r) __qpu__ {
          h(r);
          y(r);
        },
        q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3_callee
// CHECK-SAME:     (%[[VAL_0:.*]]: !cc.lambda<(!quake.ref) -> ()>,
// CHECK-SAME:      %[[VAL_1:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %{{.*}}[%{{.*}}] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_4]] : (!cc.lambda<(!quake.ref) -> ()>, !quake.ref) -> ()
// CHECK:           %[[VAL_7:.*]] = quake.extract_ref %{{.*}}[%{{.*}}] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_7]] : (!cc.lambda<(!quake.ref) -> ()>, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3_caller
// CHECK-SAME: () attributes {
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<?>[%{{.*}} : i64]
// CHECK:           %[[VAL_4:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_5:.*]]: !quake.ref):
// CHECK:             cc.scope {
// CHECK:               quake.h %[[VAL_5]]
// CHECK:               quake.y %[[VAL_5]]
// CHECK:             }
// CHECK:           } : !cc.lambda<(!quake.ref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__test3_callee{{.*}}(%[[VAL_4]], %[[VAL_2]]) : (!cc.lambda<(!quake.ref) -> ()>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// This is a template case (`auto`), so use the specialization that `callback`
// is resolved to in the AST.
struct test4_callee {
   void operator()(cudaq::signature<void(cudaq::qubit &)> auto &&callback,
                  cudaq::qreg<> &s) __qpu__ {
    callback(s[0]);
    callback(s[1]);
  }
};

struct test4_caller {
  void operator()() __qpu__ {
    cudaq::qreg q(2);
    test4_callee{}(
        [](cudaq::qubit &r) __qpu__ {
          h(r);
          x(r);
        },
        q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test4_caller
// CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_1:.*]] = arith.extsi %[[VAL_0]] : i32 to i64
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<?>[%[[VAL_1]] : i64]
// CHECK:           %[[VAL_3:.*]] = cc.undef !llvm.struct<"test4_callee", ()>
// CHECK:           %[[VAL_4:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_5:.*]]: !quake.ref):
// CHECK:             cc.scope {
// CHECK:               quake.h %[[VAL_5]] :
// CHECK:               quake.x %[[VAL_5]] : (!quake.ref) -> ()
// CHECK:             }
// CHECK:           } : !cc.lambda<(!quake.ref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__instance_test4_callee{{.*}}(%[[VAL_4]], %[[VAL_2]]) : (!cc.lambda<(!quake.ref) -> ()>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL: func.func @__nvqpp__mlirgen__instance_test4_callee
// CHECK-SAME:   (%[[VAL_0:.*]]: !cc.lambda<(!quake.ref) -> ()>,
// CHECK-SAME:    %[[VAL_1:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_1]][%[[VAL_3]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN12test4_callerclEvEUlRN5cudaq5quditILm2EEEE_(%[[VAL_4]]) : (!quake.ref) -> ()
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_6:.*]] = arith.extsi %[[VAL_5]] : i32 to i64
// CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_1]][%[[VAL_6]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN12test4_callerclEvEUlRN5cudaq5quditILm2EEEE_(%[[VAL_7]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ZN12test4_callerclEvEUlRN5cudaq5quditILm2EEEE_(
// CHECK-SAME:                                                                                          %[[VAL_0:.*]]: !quake.ref) attributes {"cudaq-kernel"} {
// CHECK:           quake.h %[[VAL_0]] :
// CHECK:           quake.x %[[VAL_0]] :
// CHECK:           return
// CHECK:         }
