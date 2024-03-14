/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake %s | cudaq-opt --canonicalize --cse | cudaq-translate --convert-to=qir | FileCheck %s

#include <cudaq.h>

struct Stuart {
   void operator() () __qpu__ {
      cudaq::qarray<5> qreg;
      y<cudaq::ctrl>(!qreg[0], qreg[1], qreg[4]);
      z<cudaq::ctrl>(qreg[2], !qreg[3], qreg[4]);
   }
};

// CHECK-LABEL: define void @__nvqpp__mlirgen__Stuart() local_unnamed_addr {
// CHECK:         %[[VAL_0:.*]] = tail call target("qir#Array") @__quantum__rt__qubit_allocate_array(i64 5)
// CHECK:         %[[VAL_1:.*]] = tail call target("qir#Qubit") @__quantum__rt__array_get_qubit_element(target("qir#Array") %[[VAL_0]], i64 0)
// CHECK:         %[[VAL_2:.*]] = tail call target("qir#Qubit") @__quantum__rt__array_get_qubit_element(target("qir#Array") %[[VAL_0]], i64 1)
// CHECK:         %[[VAL_3:.*]] = tail call target("qir#Qubit") @__quantum__rt__array_get_qubit_element(target("qir#Array") %[[VAL_0]], i64 4)
// CHECK:         tail call void @__quantum__qis__x(target("qir#Qubit") %[[VAL_1]])
// CHECK:         tail call void (i64, ptr, ...) @invokeWithControlQubits(i64 2, ptr nonnull @__quantum__qis__y__ctl, target("qir#Qubit") %[[VAL_1]], target("qir#Qubit") %[[VAL_2]], target("qir#Qubit") %[[VAL_3]])
// CHECK:         tail call void @__quantum__qis__x(target("qir#Qubit") %[[VAL_1]])
// CHECK:         %[[VAL_4:.*]] = tail call target("qir#Qubit") @__quantum__rt__array_get_qubit_element(target("qir#Array") %[[VAL_0]], i64 2)
// CHECK:         %[[VAL_5:.*]] = tail call target("qir#Qubit") @__quantum__rt__array_get_qubit_element(target("qir#Array") %[[VAL_0]], i64 3)
// CHECK:         tail call void @__quantum__qis__x(target("qir#Qubit") %[[VAL_5]])
// CHECK:         tail call void (i64, ptr, ...) @invokeWithControlQubits(i64 2, ptr nonnull @__quantum__qis__z__ctl, target("qir#Qubit") %[[VAL_4]], target("qir#Qubit") %[[VAL_5]], target("qir#Qubit") %[[VAL_3]])
// CHECK:         tail call void @__quantum__qis__x(target("qir#Qubit") %[[VAL_5]])
// CHECK:         tail call void @__quantum__rt__qubit_release_array(target("qir#Array") %[[VAL_0]])
// CHECK:         ret void
// CHECK:       }

