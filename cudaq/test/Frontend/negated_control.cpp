/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --canonicalize --cse | cudaq-translate --convert-to=qir | FileCheck %s
// clang-format on

#include <cudaq.h>

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
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_0]])
// CHECK:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__y__ctl, ptr %[[VAL_0]], ptr %[[VAL_1]], ptr %[[VAL_4]])
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_0]])
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_3]])
// CHECK:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__z__ctl, ptr %[[VAL_2]], ptr %[[VAL_3]], ptr %[[VAL_4]])
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_3]])
// CHECK-DAG:     tail call void @__quantum__rt__qubit_release(ptr %[[VAL_0]])
// CHECK-DAG:     tail call void @__quantum__rt__qubit_release(ptr %[[VAL_1]])
// CHECK-DAG:     tail call void @__quantum__rt__qubit_release(ptr %[[VAL_2]])
// CHECK-DAG:     tail call void @__quantum__rt__qubit_release(ptr %[[VAL_3]])
// CHECK-DAG:     tail call void @__quantum__rt__qubit_release(ptr %[[VAL_4]])
// CHECK:         ret void
// CHECK:       }
// clang-format on
