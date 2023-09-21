/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Simple test to make sure the tool is built and has basic functionality.

// RUN: cudaq-quake --emit-llvm-file %s | FileCheck %s
// RUN: FileCheck --check-prefixes=CHECK-LLVM %s < lambda.ll

// CHECK-LABEL: module attributes {quake.mangled_name_map = {
// CHECK-SAME: __nvqpp__mlirgen__{{.*}} = "_ZZ4mainENK3$_0clEv"
// CHECK-LABEL: func.func @__nvqpp__mlirgen__
// CHECK-SAME: 4main
// CHECK: quake.h
// CHECK: quake.mz

// CHECK-LLVM: define {{(dso_local )?}}noundef i32 @main
// CHECK-LLVM: void @"_ZZ4mainENK3$_0clEv"(ptr

#include <cudaq.h>

int main() {

  auto superposition = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
  };

  auto counts = cudaq::sample(superposition);
  counts.dump();
}
