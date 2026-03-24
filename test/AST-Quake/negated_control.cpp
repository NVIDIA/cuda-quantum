/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --canonicalize --cse | cudaq-translate --convert-to=qir | FileCheck %s

#include <cudaq.h>

struct Stuart {
   void operator() () __qpu__ {
      cudaq::qarray<5> qreg;
      y<cudaq::ctrl>(!qreg[0], qreg[1], qreg[4]);
      z<cudaq::ctrl>(qreg[2], !qreg[3], qreg[4]);
   }
};

// CHECK-LABEL: define void @__nvqpp__mlirgen__Stuart()
// CHECK:         %[[VAL_0:.*]] = tail call ptr @__quantum__rt__qubit_allocate_array(i64 5)
// CHECK:         %[[VAL_2:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 0)
// CHECK:         %[[VAL_4:.*]] = load ptr, ptr %[[VAL_2]], align 8
// CHECK:         %[[VAL_5:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 1)
// CHECK:         %[[VAL_7:.*]] = load ptr, ptr %[[VAL_5]], align 8
// CHECK:         %[[VAL_8:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 4)
// CHECK:         %[[VAL_10:.*]] = load ptr, ptr %[[VAL_8]], align 8
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_4]])
// CHECK:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__y__ctl, ptr %[[VAL_4]], ptr %[[VAL_7]], ptr %[[VAL_10]])
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_4]])
// CHECK:         %[[VAL_11:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 2)
// CHECK:         %[[VAL_13:.*]] = load ptr, ptr %[[VAL_11]], align 8
// CHECK:         %[[VAL_14:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 3)
// CHECK:         %[[VAL_15:.*]] = load ptr, ptr %[[VAL_14]], align 8
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_15]])
// CHECK:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__z__ctl, ptr %[[VAL_13]], ptr %[[VAL_15]], ptr %[[VAL_10]])
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_15]])
// CHECK:         tail call void @__quantum__rt__qubit_release_array(ptr %[[VAL_0]])
// CHECK:         ret void
// CHECK:       }
// CHECK:         ret void

