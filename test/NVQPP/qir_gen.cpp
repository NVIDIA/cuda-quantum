/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --emit-qir %s && cat qir_gen.qir.ll | \
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
// CHECK-LABEL:   define void @__nvqpp__mlirgen__branching()
// CHECK:   %[[VAL_0:.*]] = select i1 %{{.*}}, %Qubit* %{{.*}}, %Qubit* %{{.*}}
// CHECK:   tail call void @__quantum__qis__h(%Qubit* %[[VAL_0]])
// clang-format on

int main() {
  auto state1 = cudaq::get_state(branching{});
  state1.dump();
}