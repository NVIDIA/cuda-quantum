/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>
#include <iostream>

// expected-error@+1{{union types are not allowed in kernels}}
union test_union {
  int foo;
  short story;
};

int main() {
  // expected-error@+1{{failed to generate type}}
  auto kernel = [](int num_qubits) __qpu__ {
    cudaq::qvector q(num_qubits);
    h(q);
    int sum = 0;
    for (auto i = 0; i < num_qubits; i++)
      if (mz(q[i]))
        sum++;
    union test_union lookatme {};
    lookatme.foo = sum;
    return lookatme;
  };

  auto results = cudaq::run(5, kernel, 4);
  std::cout << "# results: " << results.size() << "\n";
  for (const auto &result : results)
    printf("Result: %d\n", result.foo);
  return 0;
}
