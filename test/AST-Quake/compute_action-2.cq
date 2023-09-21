/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --apply-op-specialization | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --lambda-lifting --canonicalize --apply-op-specialization -o %t && FileCheck --check-prefix=LAMBDA %s < %t && FileCheck --check-prefix=LAMBDA2 %s < %t

#include <cudaq.h>

__qpu__ void magic_func(cudaq::qreg<> &q) {
  auto nQubits = q.size();
  for (int step = 0; step < 100; ++step) {
    for (int j = 0; j < nQubits; j++)
      rx(-.01, q[j]);
    for (int i = 0; i < nQubits - 1; i++) {
      cudaq::compute_action([&]() { x<cudaq::ctrl>(q[i], q[i + 1]); },
                            [&]() { rz(-.01, q[i + 1]); });
    }
  }
}

struct ctrlHeisenberg {
  void operator()(int nQubits) __qpu__ {
    cudaq::qubit ctrl1;
    cudaq::qreg q(nQubits);
    cudaq::control(magic_func, ctrl1, q);
  }
};

// CHECK-LABEL:   func.func private @__nvqpp__mlirgen__function_magic_func.
// CHECK-SAME: .ctrl(%[[VAL_0:.*]]: !quake.veq<?>, %{{.*}}: !quake.veq<?>) {
// CHECK:           cc.scope {
// CHECK:             cc.loop while {
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 cc.scope {
// CHECK:                   cc.loop while {
// CHECK:                   } do {
// CHECK:                     quake.rx (%{{.*}}) [%[[VAL_0]]] %{{.*}} : (f64, !quake.veq<?>, !quake.ref) -> ()
// CHECK:                   } step {
// CHECK:                   }
// CHECK:                 }
// CHECK:                 cc.scope {
// CHECK:                   cc.loop while {
// CHECK:                   } do {
// CHECK:                     %[[VAL_28:.*]] = cc.create_lambda {
// CHECK:                       quake.x [%{{.*}}] %{{.*}} : (!quake.ref, !quake.ref) -> ()
// CHECK:                     } : !cc.callable<() -> ()>
// CHECK:                     %[[VAL_36:.*]] = cc.create_lambda {
// CHECK:                       quake.rz (%{{.*}}) [%[[VAL_0]]] %{{.*}} : (f64, !quake.veq<?>, !quake.ref) -> ()
// CHECK:                     } : !cc.callable<() -> ()>
// CHECK:                     quake.compute_action %[[VAL_28]], %[[VAL_36]] : !cc.callable<() -> ()>, !cc.callable<() -> ()>
// CHECK:                     cc.continue
// CHECK:                   } step {
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             } step {
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ctrlHeisenberg(
// CHECK-SAME:        %[[VAL_0:.*]]: i32)
// CHECK:           %[[VAL_1:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = arith.extsi %[[VAL_3]] : i32 to i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_4]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.concat %[[VAL_2]] : (!quake.ref) -> !quake.veq<?>
// CHECK:           call @__nvqpp__mlirgen__function_magic_func.{{.*}}.ctrl(%[[VAL_6]], %[[VAL_5]]) : (!quake.veq<?>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

//===----------------------------------------------------------------------===//

// LAMBDA-LABEL:   func.func private @__nvqpp__lifted.lambda.0.adj(
// LAMBDA-SAME:      %{{[^:]*}}: !cc.ptr<i32>, %{{[^:]*}}: !quake.veq<?>, %{{[^:]*}}: i32) {
// LAMBDA:           quake.x [%{{.*}}] %{{.*}} : (!quake.ref, !quake.ref) -> ()
// LAMBDA:           return

// LAMBDA2-LABEL:   func.func private @__nvqpp__lifted.lambda.1.ctrl(
// LAMBDA2-SAME:      %[[VAL_0:.*]]: !quake.veq<?>, %{{.*}}: !cc.ptr<i32>, %{{.*}}: i32, %{{.*}}: !quake.veq<?>, %{{.*}}: f64) {
// LAMBDA2:           quake.rz (%{{.*}}) [%[[VAL_0]]] %{{.*}} : (f64, !quake.veq<?>, !quake.ref) -> ()
// LAMBDA2:           return

// LAMBDA-LABEL:   func.func private @__nvqpp__mlirgen__function_magic_func.
// LAMBDA-SAME:    .ctrl(%[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.veq<?>) {
// LAMBDA:           cc.scope {
// LAMBDA:             cc.loop while {
// LAMBDA:             } do {
// LAMBDA:               cc.scope {
// LAMBDA:                 cc.scope {
// LAMBDA:                   cc.loop while {
// LAMBDA:                   } do {
// LAMBDA:                     quake.rx (%{{.*}}) [%[[VAL_0]]] %{{.*}} : (f64, !quake.veq<?>, !quake.ref) -> ()
// LAMBDA:                     cc.continue
// LAMBDA:                   } step {
// LAMBDA:                   }
// LAMBDA:                 }
// LAMBDA:                 cc.scope {
// LAMBDA:                   cc.loop while {
// LAMBDA:                   } do {
// LAMBDA:                     func.call @__nvqpp__lifted.lambda.0(%{{.*}}, %[[VAL_1]], %{{.*}}) : (!cc.ptr<i32>, !quake.veq<?>, i32) -> ()
// LAMBDA:                     %[[VAL_28:.*]] = quake.concat %[[VAL_0]] : (!quake.veq<?>) -> !quake.veq<?>
// LAMBDA:                     func.call @__nvqpp__lifted.lambda.1.ctrl(%[[VAL_28]], %{{.*}}, %{{.*}}, %[[VAL_1]], %{{.*}}) : (!quake.veq<?>, !cc.ptr<i32>, i32, !quake.veq<?>, f64) -> ()
// LAMBDA:                     func.call @__nvqpp__lifted.lambda.0.adj(%{{.*}}, %[[VAL_1]], %{{.*}}) : (!cc.ptr<i32>, !quake.veq<?>, i32) -> ()
// LAMBDA:                   } step {
// LAMBDA:                   }
// LAMBDA:                 }
// LAMBDA:               }
// LAMBDA:             } step {
// LAMBDA:             }
// LAMBDA:           }
// LAMBDA:           return
// LAMBDA:         }

// LAMBDA-LABEL:   func.func @__nvqpp__mlirgen__ctrlHeisenberg(
// LAMBDA-SAME:      %{{.*}}: i32)
// LAMBDA:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// LAMBDA:           %[[VAL_6:.*]] = quake.concat %[[VAL_2]] : (!quake.ref) -> !quake.veq<?>
// LAMBDA:           call @__nvqpp__mlirgen__function_magic_func.{{.*}}.ctrl(%[[VAL_6]], %{{.*}}) : (!quake.veq<?>, !quake.veq<?>) -> ()
// LAMBDA:           return
// LAMBDA:         }

// LAMBDA-LABEL:   func.func private @__nvqpp__lifted.lambda.0(
// LAMBDA-SAME:      %[[VAL_0:.*]]: !cc.ptr<i32>, %[[VAL_1:.*]]: !quake.veq<?>, %{{.*}}: i32) {
// LAMBDA:           quake.x [%{{.*}}] %{{.*}} : (!quake.ref, !quake.ref) -> ()
// LAMBDA:           return
// LAMBDA:         }

