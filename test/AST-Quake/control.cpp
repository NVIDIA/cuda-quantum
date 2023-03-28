/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct heisenbergU {
  void operator()(cudaq::qreg<> &q) __qpu__ {
    auto nQubits = q.size();
    for (int step = 0; step < 100; ++step) {
      // Fixme need a way to apply ar Rn to all qubits
      for (int j = 0; j < nQubits; j++)
        rx(-.01, q[j]);
      for (int i = 0; i < nQubits - 1; i++) {
        cudaq::compute_action([&]() { x<cudaq::ctrl>(q[i], q[i + 1]); },
                              [&]() { rz(-.01, q[i + 1]); });
      }
    }
  }
};

struct ctrlHeisenberg {
  void operator()(int nQubits) __qpu__ {
    cudaq::qubit ctrl1, ctrl2;
    cudaq::qreg q(nQubits);
    cudaq::control(heisenbergU{}, {ctrl1, ctrl2}, q);
  }
};

int main() { ctrlHeisenberg{}(5); }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__heisenbergU(

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ctrlHeisenberg(
// CHECK-SAME:        %{{.*}}: i32) attributes
// CHECK:           %[[VAL_2:.*]] = quake.alloca : !quake.qref
// CHECK:           %[[VAL_3:.*]] = quake.alloca : !quake.qref
// CHECK:           %[[VAL_8:.*]] = quake.concat %[[VAL_2]], %[[VAL_3]] : (!quake.qref, !quake.qref) -> !quake.qvec<2>
// CHECK:           quake.apply @__nvqpp__mlirgen__heisenbergU[%[[VAL_8]] : !quake.qvec<2>] %{{.*}} : (!quake.qvec<?>) -> ()
// CHECK:           return

