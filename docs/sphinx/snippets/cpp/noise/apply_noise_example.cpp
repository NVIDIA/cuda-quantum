/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ apply_noise_example.cpp && ./a.out`

#include <cudaq.h>

int main() {
  auto kernel = []() __qpu__ {
    cudaq::qubit q, r;
    cudaq::apply_noise<cudaq::depolarization2>(/*probability=*/0.1, q, r);
  };

  return 0;
}

