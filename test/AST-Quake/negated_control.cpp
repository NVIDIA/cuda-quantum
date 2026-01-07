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
// CHECK:         %[[VAL_0:.*]] = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 5)
// CHECK:         %[[VAL_2:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 0)
// CHECK:         %[[VAL_4:.*]] = load %Qubit*, %Qubit** %[[VAL_2]], align 8
// CHECK:         %[[VAL_5:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 1)
// CHECK:         %[[VAL_6:.*]] = bitcast %Qubit** %[[VAL_5]] to i8**
// CHECK:         %[[VAL_7:.*]] = load i8*, i8** %[[VAL_6]], align 8
// CHECK:         %[[VAL_8:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 4)
// CHECK:         %[[VAL_9:.*]] = bitcast %Qubit** %[[VAL_8]] to i8**
// CHECK:         %[[VAL_10:.*]] = load i8*, i8** %[[VAL_9]], align 8
// CHECK:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_4]])
// CHECK:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__y__ctl to i8*), %Qubit* %[[VAL_4]], i8* %[[VAL_7]], i8* %[[VAL_10]])
// CHECK:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_4]])
// CHECK:         %[[VAL_11:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 2)
// CHECK:         %[[VAL_12:.*]] = bitcast %Qubit** %[[VAL_11]] to i8**
// CHECK:         %[[VAL_13:.*]] = load i8*, i8** %[[VAL_12]], align 8
// CHECK:         %[[VAL_14:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 3)
// CHECK:         %[[VAL_15:.*]] = load %Qubit*, %Qubit** %[[VAL_14]], align 8
// CHECK:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_15]])
// CHECK:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__z__ctl to i8*), i8* %[[VAL_13]], %Qubit* %[[VAL_15]], i8* %[[VAL_10]])
// CHECK:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_15]])
// CHECK:         tail call void @__quantum__rt__qubit_release_array(%Array* %[[VAL_0]])
// CHECK:         ret void
// CHECK:       }
// CHECK:         ret void

