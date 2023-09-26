/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

struct deuteron_n3_ansatz {
  void operator()(double x0, double x1) __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    ry(x0, q[1]);
    ry(x1, q[2]);
    x<cudaq::ctrl>(q[2], q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    ry(-x0, q[1]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  }
};

CUDAQ_TEST(ObserveResult, checkSimple) {

  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  double result = cudaq::observe(ansatz, h, 0.59);
  EXPECT_NEAR(result, -1.7487, 1e-3);
  printf("Get energy directly as double %lf\n", result);

  auto obs_res = cudaq::observe(ansatz, h, 0.59);
  EXPECT_NEAR(obs_res.exp_val_z(), -1.7487, 1e-3);
  printf("Energy from observe_result %lf\n", obs_res.exp_val_z());

  // Observe using options w/ noise model. Note that the noise model is only
  // honored when using the Density Matrix backend.
  int shots = 252;
  cudaq::set_random_seed(13);
  cudaq::depolarization_channel depol(1.);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, depol);
  auto obs_opt =
      cudaq::observe({.shots = shots, .noise = noise}, ansatz, h, 0.59);
  // Verify that the number of shots requested was honored
  auto tmpCounts = obs_opt.raw_data();
  for (auto spinOpName : tmpCounts.register_names()) {
    if (spinOpName == cudaq::GlobalRegisterName)
      continue; // Ignore the global register
    std::size_t totalShots = 0;
    for (auto &[bitstr, counts] : tmpCounts.to_map(spinOpName))
      totalShots += counts;
    EXPECT_EQ(totalShots, shots);
  }

  printf("\n\nLAST ONE!\n");
  auto obs_res2 = cudaq::observe(100000, ansatz, h, 0.59);
  EXPECT_NEAR(obs_res2.exp_val_z(), -1.7, 1e-1);
  printf("Energy from observe_result with shots %lf\n", obs_res2.exp_val_z());
  obs_res2.dump();

  for (const auto &term : h) // td::size_t i = 0; i < h.num_terms(); i++)
    if (!term.is_identity())
      printf("Fine-grain data access: %s = %lf\n", term.to_string().data(),
             obs_res2.exp_val_z(term));

  auto x0x1Counts = obs_res2.counts(x(0) * x(1));
  x0x1Counts.dump();
  EXPECT_TRUE(x0x1Counts.size() == 4);
}

CUDAQ_TEST(ObserveResult, checkExpValBug) {

  auto kernel = []() __qpu__ {
    cudaq::qreg qubits(3);
    rx(0.531, qubits[0]);
    ry(0.9, qubits[1]);
    rx(0.3, qubits[2]);
    cz(qubits[0], qubits[1]);
    ry(-0.4, qubits[0]);
    cz(qubits[1], qubits[2]);
  };
  using namespace cudaq::spin;

  auto hamiltonian = z(0) + z(1);

  auto result = cudaq::observe(kernel, hamiltonian);
  auto exp = result.exp_val_z(z(0));
  printf("exp %lf \n", exp);
  EXPECT_NEAR(exp, .79, 1e-1);

  exp = result.exp_val_z(z(1));
  printf("exp %lf \n", exp);
  EXPECT_NEAR(exp, .62, 1e-1);

  exp = result.exp_val_z(z(0) * i(1));
  printf("exp %lf \n", exp);
  EXPECT_NEAR(exp, .79, 1e-1);
}
