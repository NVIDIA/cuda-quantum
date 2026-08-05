/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --add-dealloc | \
// RUN:   cudaq-translate --convert-to=qir:base | FileCheck %s

#include <cudaq.h>

struct Defeatism {
   void operator()() __disable_quantum_optimization__ __qpu__ {
    cudaq::qubit q0, q1;
    h(q0);
    h(q0);
  }
};

// CHECK-LABEL: define void @__nvqpp__mlirgen__Defeatism()
// CHECK:         %[[VAL_0:.*]] = tail call ptr @__quantum__rt__qubit_allocate_array(i64 2)
// CHECK:         %[[VAL_1:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 0)
// CHECK:         %[[VAL_2:.*]] = load ptr, ptr %[[VAL_1]], align 8
// CHECK:         tail call void @__quantum__qis__h(ptr %[[VAL_2]])
// CHECK:         tail call void @__quantum__qis__h(ptr %[[VAL_2]])
// CHECK:         tail call void @__quantum__rt__qubit_release_array(ptr %[[VAL_0]])
