/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ sample_implicit_measurements_dump.cpp && ./a.out`

#include <cudaq.h>

int main() {
  // [Begin Kernel C++]
  auto kernel = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
  };
  // [End Kernel C++]
  // [Begin Sample C++]
  cudaq::sample(kernel).dump();
  // [End Sample C++]
  return 0;
}

