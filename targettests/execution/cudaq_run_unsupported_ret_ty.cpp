/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && CUDAQ_LOG_LEVEL=debug %t |& FileCheck %s -check-prefix=FAIL -dump-input=always
// clang-format on

#include <cudaq.h>
#include <iostream>

auto complex_test = [](double real, double imag)
                        __qpu__ { return std::complex<double>(real, imag); };

int main() {
  std::vector<std::complex<double>> results;
  try {
    results = cudaq::run(3, complex_test, 1.0, 2.0);
  } catch (const std::runtime_error &e) {
    std::cout << "Caught runtime error" << std::endl;
    std::cout << "Runtime error: " << e.what() << std::endl;
    throw e;
  }

  std::cout << "Custom struct results:\n";
  for (const auto &r : results) {
    std::cout << "Real: " << r.real() << ", Imag: " << r.imag() << "\n";
  }

  return 0;
}

// FAIL: unsupported return type from entry-point kernel
