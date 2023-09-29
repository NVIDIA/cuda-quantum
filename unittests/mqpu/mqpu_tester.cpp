/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <gtest/gtest.h>
#include <random>

TEST(MQPUTester, checkSimple) {
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  double result = cudaq::observe<cudaq::parallel::thread>(ansatz, h, 0.59);
  EXPECT_NEAR(result, -1.7487, 1e-3);
  printf("Get energy directly as double %.16lf\n", result);
}

TEST(MQPUTester, checkLarge) {

  // This will warm up the GPUs, we don't time this
  auto &platform = cudaq::get_platform();
  printf("Num QPUs %lu\n", platform.num_qpus());
  int nQubits = 12;
  int nTerms = 1000; /// Scale this on multiple gpus to see speed up
  auto H = cudaq::spin_op::random(nQubits, nTerms, std::mt19937::default_seed);

  printf("Total Terms = %lu\n", H.num_terms());
  auto kernel = [](const int n_qubits, const int layers,
                   std::vector<int> cnot_pairs,
                   std::vector<double> params) __qpu__ {
    // Allocate the qubits
    cudaq::qvector q(n_qubits);

    // We can only handle 1d vectors so
    // count the params manually
    int param_counter = 0;
    for (int i = 0; i < n_qubits; i++) {
      rx(params[param_counter], q[i]);
      rz(params[param_counter + 1], q[i]);
      param_counter = param_counter + 2;
    }

    for (std::size_t i = 0; i < cnot_pairs.size(); i += 2) {
      x<cudaq::ctrl>(q[cnot_pairs[i]], q[cnot_pairs[i + 1]]);
    }

    // Apply layers of rotation+entangling
    for (int i = 1; i < layers; i++) {
      // Apply rotation layer
      for (int j = 0; j < n_qubits; j++) {
        rz(params[param_counter], q[j]);
        rx(params[param_counter + 1], q[j]);
        rz(params[param_counter + 2], q[j]);
        param_counter = param_counter + 3;
      }

      // Apply entangling layer
      for (std::size_t p = 0; p < cnot_pairs.size(); p += 2) {
        x<cudaq::ctrl>(q[cnot_pairs[i]], q[cnot_pairs[i + 1]]);
      }
    }
  };

  int nLayers = 2;
  auto execParams = cudaq::random_vector(
      -M_PI, M_PI, nQubits * (3 * nLayers + 2), std::mt19937::default_seed);

  std::vector<int> cnot_pairs(nQubits);
  std::iota(cnot_pairs.begin(), cnot_pairs.end(), 0);
  std::mt19937 g{std::mt19937::default_seed + 1};
  std::shuffle(cnot_pairs.begin(), cnot_pairs.end(), g);

  auto t1 = std::chrono::high_resolution_clock::now();
  cudaq::observe<cudaq::parallel::thread>(kernel, H, nQubits, nLayers,
                                          cnot_pairs, execParams);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms_double = t2 - t1;
  printf("Time %lf s\n", ms_double.count() * 1e-3);
}
