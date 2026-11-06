/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --canonicalize --cse | cudaq-translate --convert-to=qir | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --pass-pipeline='builtin.module(jit-deploy-pipeline,decomposition{basis=h,rx,ry,rz,x,y,z,x(1)})' | FileCheck --check-prefix=DEPLOY --implicit-check-not='neg [' %s
// clang-format on

#include "cudaq/qis/qubit_qis.h"

struct Stuart {
  void operator()() __qpu__ {
    cudaq::qarray<5> qreg;
    y<cudaq::ctrl>(!qreg[0], qreg[1], qreg[4]);
    z<cudaq::ctrl>(qreg[2], !qreg[3], qreg[4]);
  }
};

// clang-format off
// CHECK-LABEL: define void @__nvqpp__mlirgen__Stuart()
// CHECK:         %[[VAL_0:.*]] = tail call ptr @__quantum__rt__qubit_allocate()
// CHECK:         %[[VAL_1:.*]] = tail call ptr @__quantum__rt__qubit_allocate()
// CHECK:         %[[VAL_2:.*]] = tail call ptr @__quantum__rt__qubit_allocate()
// CHECK:         %[[VAL_3:.*]] = tail call ptr @__quantum__rt__qubit_allocate()
// CHECK:         %[[VAL_4:.*]] = tail call ptr @__quantum__rt__qubit_allocate()
// CHECK:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets_v2(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__nvqir__qis__y__ctl_values, i64 1, ptr %[[VAL_0]], ptr %[[VAL_1]], ptr %[[VAL_4]])
// CHECK:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets_v2(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__nvqir__qis__z__ctl_values, i64 2, ptr %[[VAL_2]], ptr %[[VAL_3]], ptr %[[VAL_4]])
// CHECK-DAG:     tail call void @__quantum__rt__qubit_release(ptr %[[VAL_0]])
// CHECK-DAG:     tail call void @__quantum__rt__qubit_release(ptr %[[VAL_1]])
// CHECK-DAG:     tail call void @__quantum__rt__qubit_release(ptr %[[VAL_2]])
// CHECK-DAG:     tail call void @__quantum__rt__qubit_release(ptr %[[VAL_3]])
// CHECK-DAG:     tail call void @__quantum__rt__qubit_release(ptr %[[VAL_4]])
// CHECK:         ret void
// CHECK:       }
// clang-format on

// Hardware targets run their mid-level pipeline after `jit-deploy-pipeline`.
// The `braket` basis decomposition accepts a one-control X with a negated
// control, so the deploy stage must expand negated controls before it.
struct OpenCNot {
  void operator()() __qpu__ {
    cudaq::qubit control, target;
    x<cudaq::ctrl>(!control, target);
  }
};

// clang-format off
// DEPLOY-LABEL:   func.func @__nvqpp__mlirgen__OpenCNot() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// DEPLOY:           %[[ALLOCA_0:.*]] = quake.alloca !quake.ref
// DEPLOY:           %[[ALLOCA_1:.*]] = quake.alloca !quake.ref
// DEPLOY:           quake.x %[[ALLOCA_0]] : (!quake.ref) -> ()
// DEPLOY:           quake.x {{\[}}%[[ALLOCA_0]]] %[[ALLOCA_1]] : (!quake.ref, !quake.ref) -> ()
// DEPLOY:           quake.x %[[ALLOCA_0]] : (!quake.ref) -> ()
// DEPLOY:           return
// DEPLOY:         }
// clang-format on
