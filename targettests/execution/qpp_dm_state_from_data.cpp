/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s --target density-matrix-cpu -o %t && %t | FileCheck %s

#include <cudaq.h>

#include <iomanip>
#include <iostream>

struct prepare_from_state {
  void operator()(cudaq::state *initialState) __qpu__ {
    cudaq::qvector qubits(initialState);
  }
};

int main() {
  cudaq::complex_matrix densityMatrix(2, 2);
  densityMatrix[{0, 0}] = 0.5;
  densityMatrix[{0, 1}] = std::complex<double>(0.0, 0.25);
  densityMatrix[{1, 0}] = std::complex<double>(0.0, -0.25);
  densityMatrix[{1, 1}] = 0.5;

  auto initialState = cudaq::state::from_data(densityMatrix);
  auto result =
      cudaq::observe(prepare_from_state{}, cudaq::spin::y(0), &initialState);

  std::cout << std::fixed << std::setprecision(1)
            << "expectation = " << result.expectation() << '\n';

  // CHECK: expectation = -0.5
  return 0;
}
