/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3_callee(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.callable<(!quake.ref) -> ()>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_2]] : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<?>) -> !quake.ref
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_3]] : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3_caller() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.struct<"test3_callee" {}>
// CHECK:           %[[VAL_3:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !quake.ref{{.*}}):
// CHECK:             quake.h %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:             quake.y %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__test3_callee(%[[VAL_5:.*]], %[[VAL_1]]) : (!cc.callable<(!quake.ref) -> ()>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ZN12test3_callercl
// CHECK-SAME:      _(%[[VAL_0:.*]]: !quake.ref{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.y %[[VAL_0]] : (!quake.ref) -> ()
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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test4_caller() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.struct<"test4_callee" {}>
// CHECK:           %[[VAL_3:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !quake.ref{{.*}}):
// CHECK:             quake.h %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:             quake.x %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__instance_test4_calleeZN12test4_caller[[LAM4:.*]](%[[VAL_5:.*]], %[[VAL_1]]) : (!cc.callable<(!quake.ref) -> ()>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ZN12test4_caller
// CHECK-SAME:      [[LAM:.*]]_(%[[VAL_0:.*]]: !quake.ref{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_test4_calleeZN12test4_caller
// CHECK-SAME:      [[LAM4]](%[[VAL_0:.*]]: !cc.callable<(!quake.ref) -> ()>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN12test4_caller[[LAM]]_(%[[VAL_2]]) : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN12test4_caller[[LAM]]_(%[[VAL_3]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }
