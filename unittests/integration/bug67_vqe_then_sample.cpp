/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <cudaq/algorithms/gradients/central_difference.h>
#include <cudaq/optimizers.h>
#include <cudaq/platform.h>

#ifndef CUDAQ_BACKEND_DM

CUDAQ_TEST(VqeThenSample, checkBug67) {

  struct ansatz {
    const int n_qubits;
    const int n_layers;
    void operator()(std::vector<double> theta) __qpu__ {
      using namespace cudaq::spin;
      cudaq::qvector q(n_qubits);

      // Prepare the initial state by superposition
      h(q);

      int N = q.size();
      // Loop over all the layers
      for (int i = 0; i < n_layers; ++i) {

        for (std::size_t j = 0; j < q.size(); ++j) {

          x<cudaq::ctrl>(q[j], q[(j + 1) % N]);
          rz(2.0 * theta[i], q[(j + 1) % N]);
          x<cudaq::ctrl>(q[j], q[(j + 1) % N]);
        }

        for (std::size_t j = 0; j < q.size(); ++j) {
          // Apply the mixer Hamiltonian (rx rotations)
          rx(2.0 * theta[i + n_layers], q[j]); // this is gamma
        }
      }
    }
  };

  using namespace cudaq::spin;
  cudaq::spin_op Hp = 0.5 * z(0) * z(1) + 0.5 * z(1) * z(2) +
                      0.5 * z(0) * z(3) + 0.5 * z(2) * z(3);

  int n_qubits = 4;
  int n_layers = 2;
  int n_params = 2 * n_layers;

  cudaq::optimizers::lbfgs optimizer;
  optimizer.initial_parameters = std::vector<double>{-.75, 1.15, -1.15, -.75};
  cudaq::gradients::central_difference gradient(ansatz{n_qubits, n_layers});

  auto [opt_val, opt_params] =
      cudaq::vqe(ansatz{n_qubits, n_layers}, gradient, Hp, optimizer, n_params);
  printf("theta = (%lf, %lf, %lf, %lf) \n", opt_params[0], opt_params[1],
         opt_params[2], opt_params[3]);

  // Print out the final measurement after optimization
  auto counts = cudaq::sample(ansatz{n_qubits, n_layers}, opt_params);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
  for (auto &[k, v] : counts) {
    EXPECT_TRUE(k == "0101" || k == "1010");
  }
}

#endif
