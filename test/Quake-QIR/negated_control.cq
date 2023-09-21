/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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
// CHECK:         %[[VAL_0:.*]] = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 5)
// CHECK:         %[[VAL_2:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 0)
// CHECK:         %[[VAL_3:.*]] = bitcast i8* %[[VAL_2]] to %Qubit**
// CHECK:         %[[VAL_5:.*]] = load %Qubit*, %Qubit** %[[VAL_3]], align 8
// CHECK:         %[[VAL_6:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 1)
// CHECK:         %[[VAL_7:.*]] = bitcast i8* %[[VAL_6]] to %Qubit**
// CHECK:         %[[VAL_8:.*]] = load %Qubit*, %Qubit** %[[VAL_7]], align 8
// CHECK:         %[[VAL_9:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 4)
// CHECK:         %[[VAL_10:.*]] = bitcast i8* %[[VAL_9]] to %Qubit**
// CHECK:         %[[VAL_11:.*]] = load %Qubit*, %Qubit** %[[VAL_10]], align 8
// CHECK:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_5]])
// CHECK:         tail call void (i64, void (%Array*, %Qubit*)*, ...) @invokeWithControlQubits(i64 2, void (%Array*, %Qubit*)* nonnull @__quantum__qis__y__ctl, %Qubit* %[[VAL_5]], %Qubit* %[[VAL_8]], %Qubit* %[[VAL_11]])
// CHECK:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_5]])
// CHECK:         %[[VAL_12:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 2)
// CHECK:         %[[VAL_13:.*]] = bitcast i8* %[[VAL_12]] to %Qubit**
// CHECK:         %[[VAL_14:.*]] = load %Qubit*, %Qubit** %[[VAL_13]], align 8
// CHECK:         %[[VAL_15:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 3)
// CHECK:         %[[VAL_16:.*]] = bitcast i8* %[[VAL_15]] to %Qubit**
// CHECK:         %[[VAL_17:.*]] = load %Qubit*, %Qubit** %[[VAL_16]], align 8
// CHECK:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_17]])
// CHECK:         tail call void (i64, void (%Array*, %Qubit*)*, ...) @invokeWithControlQubits(i64 2, void (%Array*, %Qubit*)* nonnull @__quantum__qis__z__ctl, %Qubit* %[[VAL_14]], %Qubit* %[[VAL_17]], %Qubit* %[[VAL_11]])
// CHECK:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_17]])
