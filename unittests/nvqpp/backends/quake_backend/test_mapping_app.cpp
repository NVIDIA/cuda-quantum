/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

#include <iostream>

// A scalar return requires no classical allocation, so canonicalization may
// erase the `cc.scope` around the `run` kernel.

struct bool_return_mapping {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    return true;
  }
};

struct int_return_mapping {
  int operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    return 2 + 3;
  }
};

struct float_return_mapping {
  float operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    return 1.5f;
  }
};

int main() {
  const auto boolResults = cudaq::run(1, bool_return_mapping{});
  if (boolResults.size() != 1 || !boolResults.front()) {
    std::cerr << "Scalar bool return failed.\n";
    return 1;
  }

  const auto intResults = cudaq::run(1, int_return_mapping{});
  if (intResults.size() != 1 || intResults.front() != 5) {
    std::cerr << "Scalar int return failed.\n";
    return 1;
  }

  const auto floatResults = cudaq::run(1, float_return_mapping{});
  if (floatResults.size() != 1 || floatResults.front() != 1.5f) {
    std::cerr << "Scalar float return failed.\n";
    return 1;
  }

  std::cout << "Mapped scalar returns passed.\n";
  return 0;
}
