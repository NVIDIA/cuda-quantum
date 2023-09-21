/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --lambda-lifting --canonicalize | FileCheck --check-prefixes=LIFT %s

#include <cudaq.h>

struct test5_callee {
  void operator()(std::function<void(cudaq::qubit &)> &&callback,
                  cudaq::qreg<> &s) __qpu__ {
     callback(s[0]);
     callback(s[1]);
     callback(s[2]);
  }
};

// Callable class, no data members.
struct test5_callable {
  void operator()(cudaq::qubit &q) __qpu__ {
    h(q);
    x(q);
    z(q);
  }
};

struct test5_caller {
  void operator()() __qpu__ {
    cudaq::qreg q(3);
    test5_callee{}(test5_callable{}, q);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__test5_callee
// CHECK-SAME:   (%[[VAL_0:.*]]: !cc.callable<(!quake.ref) -> ()>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}})
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_4]] : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ()
// CHECK:           %[[VAL_7:.*]] = quake.extract_ref %
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_7]] : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ()
// CHECK:           %[[VAL_10:.*]] = quake.extract_ref %
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_10]] : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ()
// CHECK:           return

// CHECK-LABEL: func.func @__nvqpp__mlirgen__test5_callable
// CHECK-SAME:   (%[[VAL_0:.*]]: !quake.ref{{.*}})
// CHECK:           quake.h %[[VAL_0]]
// CHECK:           quake.x %[[VAL_0]]
// CHECK:           quake.z %[[VAL_0]]

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test5_caller
// CHECK-SAME: () attributes
// CHECK:           %[[VAL_5:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_6:.*]]: !quake.ref
// CHECK:             func.call @__nvqpp__mlirgen__test5_callable{{.*}}(%[[VAL_6]]) : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__test5_callee{{.*}}(%[[VAL_5]], %{{.*}}) : (!cc.callable<(!quake.ref) -> ()>, !quake.veq<?>) -> ()

// LIFT-LABEL:   func.func @__nvqpp__mlirgen__test5_callee
// LIFT:           %[[VAL_6:.*]] = cc.callable_func %{{.*}} : (!cc.callable<(!quake.ref) -> ()>) -> ((!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ())
// LIFT:           call_indirect %[[VAL_6]](%{{.*}}, %{{.*}}) : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ()
// LIFT:           %[[VAL_8:.*]] = cc.callable_func %{{.*}} : (!cc.callable<(!quake.ref) -> ()>) -> ((!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ())
// LIFT:           call_indirect %[[VAL_8]](%{{.*}}, %{{.*}}) : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ()
// LIFT:           %[[VAL_10:.*]] = cc.callable_func %{{.*}} : (!cc.callable<(!quake.ref) -> ()>) -> ((!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ())
// LIFT:           call_indirect %[[VAL_10]](%{{.*}}, %{{.*}}) : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ()
// LIFT:           return

// LIFT-LABEL:   func.func @__nvqpp__mlirgen__test5_caller
// LIFT:           %[[VAL_2:.*]] = cc.instantiate_callable @__nvqpp__callable.thunk.lambda.0() : () -> !cc.callable<(!quake.ref) -> ()>
// LIFT:           call @__nvqpp__mlirgen__test5_callee{{.*}}(%[[VAL_2]], %
// LIFT:           return

// LIFT-LABEL:   func.func private @__nvqpp__callable.thunk.lambda.0
// LIFT-SAME:        (%[[VAL_0:.*]]: !cc.callable<(!quake.ref) -> ()>,
// LIFT-SAME:        %[[VAL_1:.*]]: !quake.ref) {
// LIFT: call @__nvqpp__lifted.lambda.0(%[[VAL_1]]) : (!quake.ref) -> ()
// LIFT: return
// LIFT: }

// LIFT-LABEL: func.func private @__nvqpp__lifted.lambda.0
// LIFT-SAME:   (%[[VAL_0:.*]]: !quake.ref) {
// LIFT: call @__nvqpp__mlirgen__test5_callable{{.*}}(%[[VAL_0]]) : (!quake.ref) -> ()
// LIFT: return
// LIFT: }
