// /*******************************************************************************
//  * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "cudaq/algorithms/evolve.h"
#include "cudaq/dynamics_integrators.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

TEST(EvolveAPITester, checkSimple) {
  const std::map<int, int> dims = {{0, 2}};
  cudaq::product_operator<cudaq::matrix_operator> ham1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_operator::x(0));
  cudaq::operator_sum<cudaq::matrix_operator> ham(ham1);

  constexpr int numSteps = 10;
  cudaq::Schedule schedule(cudaq::linspace(0.0, 1.0, numSteps));

  cudaq::product_operator<cudaq::matrix_operator> pauliZ_t =
      cudaq::spin_operator::z(0);
  cudaq::operator_sum<cudaq::matrix_operator> pauliZ(pauliZ_t);
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

  auto integrator = std::make_shared<cudaq::runge_kutta>();
  integrator->dt = 0.001;
  integrator->order = 1;
  auto result = cudaq::evolve(ham, dims, schedule, initialState, integrator, {},
                              {pauliZ}, true);
  // TODO: enable runge_kutta (fixing the dependency to cudm types)
  // EXPECT_TRUE(result.get_expectation_values().has_value());
  // EXPECT_EQ(result.get_expectation_values().value().size(), numSteps);
  // std::vector<double> theoryResults;
  // for (const auto &t : schedule) {
  //   const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t);
  //   theoryResults.emplace_back(expected);
  // }

  // int count = 0;
  // for (auto expVals : result.get_expectation_values().value()) {
  //   EXPECT_EQ(expVals.size(), 1);
  //   EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  // }
}
