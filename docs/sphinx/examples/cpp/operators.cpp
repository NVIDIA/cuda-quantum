/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Spin]
#include <cudaq.h>

auto operator= 2 * cudaq::spin::x(0) * cudaq::spin::y(1) * cudaq::spin::x(2) -
    3 * cudaq::spin::z(0) * cudaq::spin::z(1) * cudaq::spin::y(2);
// [End Spin]

// [Begin Pauli]
auto words = {"XYZ", "IXX"};
auto coefficients = {0.432, 0.324};

__qpu__ void kernel(std::vector<std::string> words,
                    std::vector<double> coefficients) {
  cudaq::qvector qvector(3);
  for (int i = 0; i < coefficients.size(); i++) {
    exp_pauli(coefficients[i], qvector, words[i]);
  }
}
// [End Pauli]
