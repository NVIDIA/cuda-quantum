/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --canonicalize --apply-op-specialization | FileCheck %s
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
// CHECK-SAME: .ctrl(%[[VAL_0:.*]]: !quake.qvec<?>, %{{.*}}: !quake.qvec<?>) {
// CHECK:           cc.scope {
// CHECK:             cc.loop while {
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 cc.scope {
// CHECK:                   cc.loop while {
// CHECK:                   } do {
// CHECK:                     quake.rx (%{{.*}}) [%[[VAL_0]]] %{{.*}} : (f64, !quake.qvec<?>, !quake.qref) -> ()
// CHECK:                   } step {
// CHECK:                   }
// CHECK:                 }
// CHECK:                 cc.scope {
// CHECK:                   cc.loop while {
// CHECK:                   } do {
// CHECK:                     %[[VAL_28:.*]] = cc.create_lambda {
// CHECK:                       quake.x [%{{.*}}] %{{.*}} : (!quake.qref, !quake.qref) -> ()
// CHECK:                     } : !cc.lambda<() -> ()>
// CHECK:                     %[[VAL_36:.*]] = cc.create_lambda {
// CHECK:                       quake.rz (%{{.*}}) [%[[VAL_0]]] %{{.*}} : (f64, !quake.qvec<?>, !quake.qref) -> ()
// CHECK:                     } : !cc.lambda<() -> ()>
// CHECK:                     quake.compute_action %[[VAL_28]], %[[VAL_36]] : !cc.lambda<() -> ()>, !cc.lambda<() -> ()>
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
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[VAL_0]], %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.qref
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_4:.*]] = arith.extsi %[[VAL_3]] : i32 to i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca[%[[VAL_4]] : i64] !quake.qvec<?>
// CHECK:           %[[VAL_6:.*]] = quake.concat %[[VAL_2]] : (!quake.qref) -> !quake.qvec<?>
// CHECK:           call @__nvqpp__mlirgen__function_magic_func.{{.*}}.ctrl(%[[VAL_6]], %[[VAL_5]]) : (!quake.qvec<?>, !quake.qvec<?>) -> ()
// CHECK:           return
// CHECK:         }

//===----------------------------------------------------------------------===//

// LAMBDA-LABEL:   func.func private @__nvqpp__lifted.lambda.0.adj(
// LAMBDA-SAME:      %{{[^:]*}}: memref<i32>, %{{[^:]*}}: !quake.qvec<?>) {
// LAMBDA:           quake.x [%{{.*}}] %{{.*}} : (!quake.qref, !quake.qref) -> ()
// LAMBDA:           return

// LAMBDA2-LABEL:   func.func private @__nvqpp__lifted.lambda.1.ctrl(
// LAMBDA2-SAME:      %[[VAL_0:.*]]: !quake.qvec<?>, %{{.*}}: memref<i32>, %{{.*}}: !quake.qvec<?>) {
// LAMBDA2:           quake.rz (%{{.*}}) [%[[VAL_0]]] %{{.*}} : (f64, !quake.qvec<?>, !quake.qref) -> ()
// LAMBDA2:           return

// LAMBDA-LABEL:   func.func private @__nvqpp__mlirgen__function_magic_func.
// LAMBDA-SAME:    .ctrl(%[[VAL_0:.*]]: !quake.qvec<?>, %[[VAL_1:.*]]: !quake.qvec<?>) {
// LAMBDA:           cc.scope {
// LAMBDA:             cc.loop while {
// LAMBDA:             } do {
// LAMBDA:               cc.scope {
// LAMBDA:                 cc.scope {
// LAMBDA:                   cc.loop while {
// LAMBDA:                   } do {
// LAMBDA:                     quake.rx (%{{.*}}) [%[[VAL_0]]] %{{.*}} : (f64, !quake.qvec<?>, !quake.qref) -> ()
// LAMBDA:                     cc.continue
// LAMBDA:                   } step {
// LAMBDA:                   }
// LAMBDA:                 }
// LAMBDA:                 cc.scope {
// LAMBDA:                   cc.loop while {
// LAMBDA:                   } do {
// LAMBDA:                     func.call @__nvqpp__lifted.lambda.0(%{{.*}}, %[[VAL_1]]) : (memref<i32>, !quake.qvec<?>) -> ()
// LAMBDA:                     %[[VAL_28:.*]] = quake.concat %[[VAL_0]] : (!quake.qvec<?>) -> !quake.qvec<?>
// LAMBDA:                     func.call @__nvqpp__lifted.lambda.1.ctrl(%[[VAL_28]], %{{.*}}, %[[VAL_1]]) : (!quake.qvec<?>, memref<i32>, !quake.qvec<?>) -> ()
// LAMBDA:                     func.call @__nvqpp__lifted.lambda.0.adj(%{{.*}}, %[[VAL_1]]) : (memref<i32>, !quake.qvec<?>) -> ()
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
// LAMBDA:           %[[VAL_2:.*]] = quake.alloca !quake.qref
// LAMBDA:           %[[VAL_6:.*]] = quake.concat %[[VAL_2]] : (!quake.qref) -> !quake.qvec<?>
// LAMBDA:           call @__nvqpp__mlirgen__function_magic_func.{{.*}}.ctrl(%[[VAL_6]], %{{.*}}) : (!quake.qvec<?>, !quake.qvec<?>) -> ()
// LAMBDA:           return
// LAMBDA:         }

// LAMBDA-LABEL:   func.func private @__nvqpp__lifted.lambda.0(
// LAMBDA-SAME:      %[[VAL_0:.*]]: memref<i32>, %[[VAL_1:.*]]: !quake.qvec<?>) {
// LAMBDA:           quake.x [%{{.*}}] %{{.*}} : (!quake.qref, !quake.qref) -> ()
// LAMBDA:           return
// LAMBDA:         }

