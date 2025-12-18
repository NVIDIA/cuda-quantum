/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t |& FileCheck %s -check-prefix=FAIL
// clang-format on

#include <cudaq.h>
#include <iostream>

auto complex_test = [](double real, double imag)
                        __qpu__ { return std::complex<double>(real, imag); };

int main() {
  const auto results = cudaq::run(3, complex_test, 1.0, 2.0);

  std::cout << "Custom struct results:\n";
  for (const auto &r : results) {
    std::cout << "Real: " << r.real() << ", Imag: " << r.imag() << "\n";
  }

  return 0;
}

// FAIL: unsupported return type from entry-point kernel
