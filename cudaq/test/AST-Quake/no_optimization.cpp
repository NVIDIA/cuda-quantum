/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --add-dealloc | \
// RUN:   cudaq-translate --convert-to=qir-base | FileCheck %s

#include <cudaq.h>

struct Defeatism {
  void operator()() __disable_quantum_optimization__ __qpu__ {
    cudaq::qubit q0, q1;
    h(q0);
    h(q0);
  }
};

// clang-format off
// CHECK-LABEL: define void @__nvqpp__mlirgen__Defeatism()
// CHECK:         tail call void @__quantum__qis__h(ptr
// CHECK:         tail call void @__quantum__qis__h(ptr
// CHECK-NOT:     @__quantum__qis__h
// CHECK:         ret void
