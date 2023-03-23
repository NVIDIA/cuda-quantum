/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --canonicalize --cse | cudaq-translate --convert-to=qir | FileCheck %s

#include <cudaq.h>

struct Stuart {
   void operator() () __qpu__ {
      cudaq::qreg<5> qreg;
      y<cudaq::ctrl>(!qreg[0], qreg[1], qreg[4]);
      z<cudaq::ctrl>(qreg[2], !qreg[3], qreg[4]);
   }
};

// CHECK-LABEL: define void @__nvqpp__mlirgen__Stuart()
// CHECK:         %[[VAL_0:.*]] = tail call ptr @__quantum__rt__qubit_allocate_array(i64 5)
// CHECK:         %[[VAL_2:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 0)
// CHECK:         %[[VAL_5:.*]] = load ptr, ptr %[[VAL_2]], align 8
// CHECK:         %[[VAL_6:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 1)
// CHECK:         %[[VAL_8:.*]] = load ptr, ptr %[[VAL_6]], align 8
// CHECK:         %[[VAL_9:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 4)
// CHECK:         %[[VAL_11:.*]] = load ptr, ptr %[[VAL_9]], align 8
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_5]])
// CHECK:         tail call void (i64, ptr, ...) @invokeWithControlQubits(i64 2, ptr nonnull @__quantum__qis__y__ctl, ptr %[[VAL_5]], ptr %[[VAL_8]], ptr %[[VAL_11]])
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_5]])
// CHECK:         %[[VAL_12:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 2)
// CHECK:         %[[VAL_14:.*]] = load ptr, ptr %[[VAL_12]], align 8
// CHECK:         %[[VAL_15:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 3)
// CHECK:         %[[VAL_17:.*]] = load ptr, ptr %[[VAL_15]], align 8
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_17]])
// CHECK:         tail call void (i64, ptr, ...) @invokeWithControlQubits(i64 2, ptr nonnull @__quantum__qis__z__ctl, ptr %[[VAL_14]], ptr %[[VAL_17]], ptr %[[VAL_11]])
// CHECK:         tail call void @__quantum__qis__x(ptr %[[VAL_17]])
