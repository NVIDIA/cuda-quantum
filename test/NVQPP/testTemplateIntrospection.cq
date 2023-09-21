/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o out_testTemplateI.x && ./out_testTemplateI.x | FileCheck %s

#include <cudaq.h>

template <std::size_t N>
struct ghz {
  void operator()() __qpu__ {
    cudaq::qreg<N> q;
    h(q[0]);
    // .. etc
  }
};

// CHECK: 1: module { func.func @__nvqpp__mlirgen__ghzILm3EE() attributes
// CHECK: 2: module { func.func @__nvqpp__mlirgen__ghzILm4EE() attributes

int main() {
  printf("1: %s", cudaq::get_quake(ghz<3>{}).c_str());
  printf("2: %s", cudaq::get_quake(ghz<4>{}).c_str());
  return 0;
}
