/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// REQUIRES: c++20

// clang-format off
// RUN: nvq++ -fenable-cudaq-run %cpp_std --target remote-mqpu %s -o %t && %t | FileCheck %s
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>
#include <iostream>

struct rwpe {
  double operator()(int n_iter, double mu, double sigma) __qpu__ {
    int iteration = 0;

    // Allocate the qubits
    cudaq::qvector q(2);

    // Alias them
    auto &aux = q.front();
    auto &target = q.back();

    x(q[1]);

    while (iteration < n_iter) {
      h(aux);
      rz(1.0 - (mu / sigma), aux);
      rz(.25 / sigma, target);
      x<cudaq::ctrl>(aux, target);
      rz(-.25 / sigma, target);
      x<cudaq::ctrl>(aux, target);
      h(aux);
      if (mz(aux)) {
        x(aux);
        mu += sigma * .6065;
      } else {
        mu -= sigma * .6065;
      }

      sigma *= .7951;
      iteration += 1;
    }

    return 2. * mu;
  }
};

__qpu__ int nullary_test() {
  unsigned result = 0;
  cudaq::qvector v(8);
  h(v);
  z(v);
  for (int i = 0; i < 8; i++) {
    bool w = mz(v[i]);
    result |= ((unsigned)w) << (8 - 1 - i);
  }
  return result;
}

__qpu__ int unary_test(int count) {
  unsigned result = 0;
  cudaq::qvector v(count);
  h(v);
  z(v);
  for (int i = 0; i < count; i++) {
    bool w = mz(v[i]);
    result |= ((unsigned)w) << (count - 1 - i);
  }
  return result;
}

int main() {
  int c = 0;
  {
    std::vector<int> results =
        cudaq::run<int>(100, std::function<int()>{nullary_test});
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }

  {
    std::vector<int> results =
        cudaq::run<int>(50, std::function<int(int)>{unary_test}, 4);
    c = 0;
    if (results.size() != 50) {
      printf("FAILED! Expected 50 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }
  cudaq::set_random_seed(123);
  const std::size_t shots = 100;
  auto phases = cudaq::run<double>(
      shots, std::function<double(int, double, double)>{rwpe{}}, 24, 0.7951,
      0.6065);

  if (phases.size() != shots) {
    printf("FAILED! Expected %lu shots. Got %lu\n", shots, phases.size());
  } else {
    c = 0;
    for (auto phase : phases) {
      printf("%d: %lf\n", c++, phase);
    }
    printf("success!\n");
  }

  return 0;
}

// CHECK: success!
// CHECK: success!
// CHECK: success!
