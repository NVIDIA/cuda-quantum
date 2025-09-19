/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --emit-qir %s && cat qir_gen.qir.ll | \
// RUN: FileCheck %s && rm qir_gen.qir.ll

#include <cudaq.h>
#include <iostream>

struct branching {
  void operator()() __qpu__ { 
    cudaq::qvector q(3);
    
    h(q.front());
    x<cudaq::ctrl>(q[0],q[2]);
    if (mz(q[1]))
        h(q.front());
    else {
        h(q.back());
    }
  }
};

// clang-format off
// CHECK-LABEL:   define void @__nvqpp__mlirgen__branching() {
// CHECK:   %1 = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 3)
// CHECK:   %2 = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %1, i64 0)
// CHECK:   %3 = load %Qubit*, %Qubit** %2, align 8
// CHECK:   tail call void @__quantum__qis__h(%Qubit* %3)
// CHECK:   %4 = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %1, i64 2)
// CHECK:   %5 = load %Qubit*, %Qubit** %4, align 8
// CHECK:   tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %3, %Qubit* %5)
// CHECK:   %6 = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %1, i64 1)
// CHECK:   %7 = load %Qubit*, %Qubit** %6, align 8
// CHECK:   %8 = tail call %Result* @__quantum__qis__mz(%Qubit* %7)
// CHECK:   %9 = bitcast %Result* %8 to i1*
// CHECK:   %10 = load i1, i1* %9, align 1
// CHECK:   %11 = select i1 %10, %Qubit* %3, %Qubit* %5
// CHECK:   tail call void @__quantum__qis__h(%Qubit* %11)
// CHECK:   tail call void @__quantum__rt__qubit_release_array(%Array* %1)
// CHECK:   ret void
// CHECK: }
// clang-format on

int main() {
  auto state1 = cudaq::get_state(branching{});
  state1.dump();
}