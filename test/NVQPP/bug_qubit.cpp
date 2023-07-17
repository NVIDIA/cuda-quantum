/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This code is from Issue 251.

// RUN: nvq++ --enable-mlir -v %s --target quantinuum --emulate -o %t.x && %t.x | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --promote-qubit-allocation | FileCheck --check-prefixes=MLIR %s

// CHECK: Test: { 0:{{[0-9]+}} 1:{{[0-9]+}} }

#include <cudaq.h>
#include <iostream>

struct ak2 {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
  }
};

// MLIR-LABEL:   func.func @__nvqpp__mlirgen__ak2()
// MLIR-NOT:       quake.alloca !quake.ref
// MLIR:           %[[VAL_0:.*]] = quake.alloca !quake.veq<1>
// MLIR-NEXT:      %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<1>) -> !quake.ref

int main() {
  auto counts = cudaq::sample(ak2{});
  std::cout << "Test: ";
  counts.dump();
  return 0;
}
