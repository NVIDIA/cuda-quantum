/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// clang-format off
// RUN: nvq++ -std=c++17 --enable-mlir --opt-plugin %cudaq_lib_dir/CustomPassPlugin.so --opt-pass 'func.func(cudaq-custom-pass)' %s -o %t && %t | FileCheck %s
// RUN: nvq++ --enable-mlir --opt-plugin %cudaq_lib_dir/CustomPassPlugin.so --opt-pass 'func.func(cudaq-custom-pass)'  %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

void kernel() __qpu__ {
  cudaq::qarray<2> q;
  h(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
  mz(q);
}

int main() {
  auto result = cudaq::sample(1000, kernel);
  for (auto &&[bits, counts] : result) {
    std::cout << bits << '\n';
  }
  return 0;
}

// The custom pass replace H with S, hence not a Bell state anymore.
// CHECK: 00
