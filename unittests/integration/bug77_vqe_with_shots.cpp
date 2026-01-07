/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

#if !defined(CUDAQ_BACKEND_DM) && !defined(CUDAQ_BACKEND_STIM)
CUDAQ_TEST(VqeWithShots, checkBug77) {

  struct ansatz {
    const int n_qubits;
    const int n_layers;
    void operator()(std::vector<double> theta) __qpu__ {

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

  int n_qubits = 4;
  int n_layers = 2;
  int n_params = 2 * n_layers;

  // Problem Hamiltonian
  cudaq::spin_op Hp = 0.5 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1) +
                      0.5 * cudaq::spin_op::z(1) * cudaq::spin_op::z(2) +
                      0.5 * cudaq::spin_op::z(0) * cudaq::spin_op::z(3) +
                      0.5 * cudaq::spin_op::z(2) * cudaq::spin_op::z(3);

  // Optimizer
  cudaq::optimizers::cobyla optimizer; // gradient-free
  optimizer.max_eval = 100;
  // Set initial values for the parameters
  optimizer.initial_parameters =
      std::vector<double>{1.118643, 1.011415, -1.011415, 2.022654};

  // Call the optimizer
  auto [opt_val, opt_params] =
      cudaq::vqe(ansatz{n_qubits, n_layers}, Hp, optimizer, n_params);

  // Print the optimized value and the parameters
  printf("Optimal value = %lf\n", opt_val);
  printf("Optimal params = (%lf, %lf, %lf, %lf) \n", opt_params[0],
         opt_params[1], opt_params[2], opt_params[3]);

  EXPECT_NEAR(opt_val, -2.0, 1e-2);
}
#endif
