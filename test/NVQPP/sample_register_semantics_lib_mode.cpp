/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %basename_t.x && ./%basename_t.x | FileCheck %s
// RUN: nvq++ --target quantinuum               --emulate %s -o %basename_t.x && ./%basename_t.x | FileCheck %s

#include <cudaq.h>
#include <iostream>

void bar() __qpu__ {
  cudaq::qubit a;
  cudaq::qubit b;
  x(a);
  auto ret_b = mz(b); // automatically handled
  auto ret_a = mz(a, "ret_a"); // can set manually
}

int main() {
  auto result = cudaq::sample(bar);
  result.dump();

  auto regNames = result.register_names();
  if (std::find(regNames.begin(), regNames.end(), "ret_b") != regNames.end()) std::cout << "SUCCESS\n"; 
  if (std::find(regNames.begin(), regNames.end(), "ret_a") != regNames.end()) std::cout << "SUCCESS\n"; 

  return 0;
}

// CHECK: SUCCESS
// CHECK: SUCCESS