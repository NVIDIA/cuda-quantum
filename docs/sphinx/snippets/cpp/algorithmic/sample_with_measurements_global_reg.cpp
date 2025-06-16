/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ sample_with_measurements_global_reg.cpp && ./a.out`

#include <cudaq.h>
#include <stdio.h>

int main() {
  // [Begin Kernel C++]
  auto kernel = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    mz(b);
    mz(a);
  };
  // [End Kernel C++]

  // [Begin Sample C++]
  printf("Default - no explicit measurements\n");
  cudaq::sample(kernel).dump();

  cudaq::sample_options options;
  options.explicit_measurements = true;
  printf("\nSetting `explicit_measurements` option\n");
  cudaq::sample(options, kernel).dump();
  // [End Sample C++]
  return 0;
}

