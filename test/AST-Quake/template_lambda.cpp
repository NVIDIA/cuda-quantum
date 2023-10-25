/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct thisWorks {
  void operator()(cudaq::qubit &q) __qpu__ { x(q); }
};

struct test {
  template <typename Callable>
  void operator()(Callable &&callable) __qpu__ {
    cudaq::qreg q(2);
    cudaq::control(callable, q[0], q[1]);
  }
};

int main() {
  test{}([](cudaq::qubit &q) __qpu__ { x(q); });
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__thisWorks

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Z4mainE3$_0(
// CHECK:           quake.x %{{.*}} :

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_test
// CHECK-SAME:        (%[[VAL_0:.*]]: !cc.callable<(!quake.ref) -> ()>
// CHECK-NOT:       %[[VAL_0]]
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_6:.*]] = quake.extract_ref %{{.*}}[0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_9:.*]] = quake.extract_ref %{{.*}}[1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.apply @__nvqpp__mlirgen__Z4mainE3$_0 [%[[VAL_6]]] %[[VAL_9]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return

