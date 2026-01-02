/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>
#include <iostream>

struct rwpe {
  double operator()(const int n_iter, double mu, double sigma) __qpu__ {
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
        mu += sigma * .6065;
      } else {
        mu -= sigma * .6065;
      }

      sigma *= .7951;
      iteration += 1;
      // Reset qubit for reuse
      reset(aux);
    }

    return 2. * mu;
  }
};

struct returnTrue {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    return mz(q);
  }
};

struct returnFalse {
  bool operator()() __qpu__ {
    cudaq::qubit q, r;
    x(q);
    return mz(q) && mz(r);
  }
};

struct returnInt {
  int operator()(int iters) __qpu__ {
    cudaq::qubit q;
    int count = 0;
    for (int i = 0; i < iters; ++i) {
      h(q);
      if (mz(q)) {
        count++;
        x(q);
      }
    }
    return count;
  }
};

int main() {
  cudaq::set_random_seed(123);
  int n_iterations = 24;
  double mu = 0.7951, sigma = 0.6065;
  auto phase = rwpe{}(n_iterations, mu, sigma);

  REMOTE_TEST_ASSERT(std::abs(phase - 0.49) < 0.05);

  REMOTE_TEST_ASSERT(returnTrue{}());

  REMOTE_TEST_ASSERT(!returnFalse{}());
  const int oneCount = returnInt{}(1000);
  std::cout << "One count = " << oneCount << "\n";
  // We expect ~ 50% one.
  REMOTE_TEST_ASSERT(oneCount > 100);
  return 0;
}
