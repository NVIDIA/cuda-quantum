/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

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
// CHECK-SAME:   (%[[VAL_0:.*]]: !cc.lambda<(!quake.qref) -> ()>,
// CHECK-SAME:   %[[VAL_1:.*]]: !quake.qvec<?>)
// CHECK:           %[[VAL_4:.*]] = quake.qextract %
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_4]] : (!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ()
// CHECK:           %[[VAL_7:.*]] = quake.qextract %
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_7]] : (!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ()
// CHECK:           %[[VAL_10:.*]] = quake.qextract %
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_10]] : (!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ()
// CHECK:           return

// CHECK-LABEL: func.func @__nvqpp__mlirgen__test5_callable
// CHECK-SAME:   (%[[VAL_0:.*]]: !quake.qref)
// CHECK:           quake.h (%[[VAL_0]])
// CHECK:           quake.x (%[[VAL_0]])
// CHECK:           quake.z (%[[VAL_0]])

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test5_caller
// CHECK-SAME: () attributes
// CHECK:           %[[VAL_5:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_6:.*]]: !quake.qref):
// CHECK:             func.call @__nvqpp__mlirgen__test5_callable{{.*}}(%[[VAL_6]]) : (!quake.qref) -> ()
// CHECK:           } : !cc.lambda<(!quake.qref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__test5_callee{{.*}}(%[[VAL_5]], %{{.*}}) : (!cc.lambda<(!quake.qref) -> ()>, !quake.qvec<?>) -> ()

// LIFT-LABEL:   func.func @__nvqpp__mlirgen__test5_callee
// LIFT:           %[[VAL_6:.*]] = cc.callable_func %{{.*}} : (!cc.lambda<(!quake.qref) -> ()>) -> ((!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ())
// LIFT:           call_indirect %[[VAL_6]](%{{.*}}, %{{.*}}) : (!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ()
// LIFT:           %[[VAL_8:.*]] = cc.callable_func %{{.*}} : (!cc.lambda<(!quake.qref) -> ()>) -> ((!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ())
// LIFT:           call_indirect %[[VAL_8]](%{{.*}}, %{{.*}}) : (!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ()
// LIFT:           %[[VAL_10:.*]] = cc.callable_func %{{.*}} : (!cc.lambda<(!quake.qref) -> ()>) -> ((!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ())
// LIFT:           call_indirect %[[VAL_10]](%{{.*}}, %{{.*}}) : (!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ()
// LIFT:           return

// LIFT-LABEL:   func.func @__nvqpp__mlirgen__test5_caller
// LIFT:           %[[VAL_2:.*]] = cc.instantiate_callable @__nvqpp__callable.thunk.lambda.0() : () -> !cc.lambda<(!quake.qref) -> ()>
// LIFT:           call @__nvqpp__mlirgen__test5_callee{{.*}}(%[[VAL_2]], %
// LIFT:           return

// LIFT-LABEL:   func.func private @__nvqpp__callable.thunk.lambda.0
// LIFT-SAME:        (%[[VAL_0:.*]]: !cc.lambda<(!quake.qref) -> ()>,
// LIFT-SAME:        %[[VAL_1:.*]]: !quake.qref) {
// LIFT: call @__nvqpp__lifted.lambda.0(%[[VAL_1]]) : (!quake.qref) -> ()
// LIFT: return
// LIFT: }

// LIFT-LABEL: func.func private @__nvqpp__lifted.lambda.0
// LIFT-SAME:   (%[[VAL_0:.*]]: !quake.qref) {
// LIFT: call @__nvqpp__mlirgen__test5_callable{{.*}}(%[[VAL_0]]) : (!quake.qref) -> ()
// LIFT: return
// LIFT: }
