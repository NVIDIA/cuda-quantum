/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ random_walk_qpe.cpp -o qpe.x && ./qpe.x
// ```

#include <cudaq.h>

// Here we demonstrate an algorithm expressed as a CUDA Quantum kernel
// that incorporates non-trivial control flow and conditional
// quantum instruction invocation.

struct rwpe {
  double operator()(const int n_iter, double mu, double sigma) __qpu__ {
    int iteration = 0;

    // Allocate the qubits
    cudaq::qreg q(2);

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

int main() {
  int n_iterations = 24;
  double mu = 0.7951, sigma = 0.6065;
  auto phase = rwpe{}(n_iterations, mu, sigma);
  printf("Phase = %lf\n", phase);
}
