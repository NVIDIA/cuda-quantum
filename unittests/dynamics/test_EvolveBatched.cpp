// /*******************************************************************************
//  * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "CuDensityMatState.h"
#include "common/EigenDense.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/algorithms/integrator.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>

TEST(BatchedEvolveTester, checkSimple) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham_1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham1(ham_1);

  cudaq::product_op<cudaq::matrix_handler> ham_2 =
      (2.0 * M_PI * 0.2 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham2(ham_2);

  constexpr int numSteps = 10;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);
  auto initialState1 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  auto initialState2 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto results = cudaq::__internal__::evolveBatched(
      {ham1, ham2}, dims, schedule, {initialState1, initialState2}, integrator,
      {}, {pauliZ}, cudaq::IntermediateResultSave::ExpectationValue);

  EXPECT_EQ(results.size(), 2);
  EXPECT_TRUE(results[0].expectation_values.has_value());
  EXPECT_EQ(results[0].expectation_values.value().size(), numSteps);
  EXPECT_TRUE(results[1].expectation_values.has_value());
  EXPECT_EQ(results[1].expectation_values.value().size(), numSteps);

  std::vector<double> theoryResults1;
  std::vector<double> theoryResults2;
  for (const auto &t : schedule) {
    const double expected1 = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    const double expected2 = std::cos(2 * 2.0 * M_PI * 0.2 * t.real());
    theoryResults1.emplace_back(expected1);
    theoryResults2.emplace_back(expected2);
  }

  int count = 0;
  for (auto expVals : results[0].expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults1[count++], 1e-3);
  }

  count = 0;
  for (auto expVals : results[1].expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults2[count++], 1e-3);
  }
}

TEST(BatchedEvolveTester, checkMultiTerms) {
  constexpr int cavity_levels = 10;
  const cudaq::dimension_map dims = {{0, 2}, {1, cavity_levels}};
  auto a = cudaq::boson_op::annihilate(1);
  auto a_dag = cudaq::boson_op::create(1);

  auto sm = cudaq::boson_op::annihilate(0);
  auto sm_dag = cudaq::boson_op::create(0);

  cudaq::product_op<cudaq::matrix_handler> atom_occ_op_t =
      cudaq::matrix_handler::number(0);
  cudaq::sum_op<cudaq::matrix_handler> atom_occ_op(atom_occ_op_t);

  cudaq::product_op<cudaq::matrix_handler> cavity_occ_op_t =
      cudaq::matrix_handler::number(1);
  cudaq::sum_op<cudaq::matrix_handler> cavity_occ_op(cavity_occ_op_t);

  auto hamiltonian1 = 2 * M_PI * atom_occ_op + 2 * M_PI * cavity_occ_op +
                      2 * M_PI * 0.1 * (sm * a_dag + sm_dag * a);

  auto hamiltonian2 = 2 * M_PI * atom_occ_op + 2 * M_PI * cavity_occ_op +
                      2 * M_PI * 0.2 * (sm * a_dag + sm_dag * a);

  Eigen::Vector2cd qubit_state;
  qubit_state << 1.0, 0.0;
  Eigen::VectorXcd cavity_state = Eigen::VectorXcd::Zero(cavity_levels);
  const int num_photons = 5;
  cavity_state[num_photons] = 1.0;
  Eigen::VectorXcd initial_state_vec =
      Eigen::kroneckerProduct(cavity_state, qubit_state);
  constexpr int num_steps = 101;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, num_steps);
  cudaq::schedule schedule(steps, {"t"});
  auto initialState = cudaq::state::from_data(
      std::make_pair(initial_state_vec.data(), initial_state_vec.size()));
  cudaq::integrators::runge_kutta integrator(4, 0.001);

  auto results = cudaq::__internal__::evolveBatched(
      {hamiltonian1, hamiltonian2}, dims, schedule,
      {initialState, initialState}, integrator, {},
      {cavity_occ_op, atom_occ_op},
      cudaq::IntermediateResultSave::ExpectationValue);

  EXPECT_EQ(results.size(), 2);

  for (const auto &result : results) {
    EXPECT_TRUE(result.expectation_values.has_value());
    EXPECT_EQ(result.expectation_values.value().size(), num_steps);

    int count = 0;
    for (auto expVals : result.expectation_values.value()) {
      EXPECT_EQ(expVals.size(), 2);
      std::cout << expVals[0] << " | ";
      std::cout << expVals[1] << "\n";
      // This should be an exchanged interaction
      EXPECT_NEAR((double)expVals[0] + (double)expVals[1], num_photons, 1e-2);
    }
  }

  // The second one is twice as fast (as the interation strength is doubled)
  for (std::size_t i = 0; i < num_steps / 2; ++i) {
    auto expVals1 = results[0].expectation_values.value()[2 * i][1];
    auto expVals2 = results[1].expectation_values.value()[i][1];
    EXPECT_NEAR((double)expVals1, (double)expVals2, 1e-3);
  }
}
