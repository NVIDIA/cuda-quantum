/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct MyKernel {
  template <typename StatePrep>
  void operator()(StatePrep &&statePrep) __qpu__ {
    cudaq::qreg q(2);
    statePrep(q);
  }
};

int main() {
  auto bell = [](cudaq::qreg<> &q) __qpu__ {
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  };

  MyKernel k;
  k(bell);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Z4mainE3$_0
// CHECK-SAME:        (%[[VAL_0:.*]]: !quake.veq<?>{{.*}}) attributes
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<?>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_6]]] %[[VAL_9]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_MyKernel
// CHECK-SAME:        (%[[VAL_0:.*]]: !cc.callable<(!quake.veq<?>) -> ()>{{.*}}) attributes
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = quake.relax_size %[[VAL_3]] :
// CHECK:           call @__nvqpp__mlirgen__Z4mainE3$_0(%[[VAL_4]]) : (!quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

