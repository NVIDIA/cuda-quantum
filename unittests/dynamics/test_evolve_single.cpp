// /*******************************************************************************
//  * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "common/EigenDense.h"
#include "cudaq/evolution.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>

TEST(EvolveTester, checkSimple) {
  const std::map<int, int> dims = {{0, 2}};
  const std::string op_id = "pauli_x";
  auto func = [](std::vector<int> dimensions,
                 std::map<std::string, std::complex<double>> _none) {
    if (dimensions.size() != 1)
      throw std::invalid_argument("Must have a singe dimension");
    if (dimensions[0] != 2)
      throw std::invalid_argument("Must have dimension 2");
    auto mat = cudaq::matrix_2(2, 2);
    mat[{1, 0}] = 1.0;
    mat[{0, 1}] = 1.0;
    return mat;
  };
  cudaq::matrix_operator::define(op_id, {-1}, func);
  auto ham = cudaq::product_operator<cudaq::matrix_operator>(
      2.0 * M_PI * 0.1, cudaq::matrix_operator(op_id, {0}));
  constexpr int numSteps = 10;
  cudaq::Schedule schedule(cudaq::linspace(0.0, 1.0, numSteps));

  cudaq::matrix_operator::define(
      "pauli_z", {-1},
      [](std::vector<int> dimensions,
         std::map<std::string, std::complex<double>> _none) {
        if (dimensions.size() != 1)
          throw std::invalid_argument("Must have a singe dimension");
        if (dimensions[0] != 2)
          throw std::invalid_argument("Must have dimension 2");
        auto mat = cudaq::matrix_2(2, 2);
        mat[{0, 0}] = 1.0;
        mat[{1, 1}] = -1.0;
        return mat;
      });
  auto pauliZ = cudaq::product_operator<cudaq::matrix_operator>(
      std::complex<double>{1.0, 0.0}, cudaq::matrix_operator("pauli_z", {0}));
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  auto result = cudaq::evolve_single(ham, dims, schedule, initialState, {},
                                     {&pauliZ}, true);
  EXPECT_TRUE(result.get_expectation_values().has_value());
  EXPECT_EQ(result.get_expectation_values().value().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t);
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.get_expectation_values().value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveTester, checkCompositeSystem) {
  constexpr int cavity_levels = 10;
  const std::map<int, int> dims = {{0, 2}, {1, cavity_levels}};
  auto a = cudaq::matrix_operator::annihilate(1);
  auto a_dag = cudaq::matrix_operator::create(1);

  auto sm = cudaq::matrix_operator::annihilate(0);
  auto sm_dag = cudaq::matrix_operator::create(0);

  auto atom_occ_op = cudaq::matrix_operator::number(0);
  auto cavity_occ_op = cudaq::matrix_operator::number(1);
  auto hamiltonian = 2 * M_PI * atom_occ_op + 2 * M_PI * cavity_occ_op +
                     2 * M_PI * 0.25 * (sm * a_dag + sm_dag * a);
  // auto matrix = hamiltonian.to_matrix(dims);
  // std::cout << "Matrix:\n" << matrix.dump() << "\n";
  Eigen::Vector2cd qubit_state;
  qubit_state << 1.0, 0.0;
  Eigen::VectorXcd cavity_state = Eigen::VectorXcd::Zero(cavity_levels);
  const int num_photons = 5;
  cavity_state[num_photons] = 1.0;
  Eigen::VectorXcd initial_state_vec =
      Eigen::kroneckerProduct(cavity_state, qubit_state);
  std::cout << "Initial state: \n" << initial_state_vec << "\n";
  constexpr int num_steps = 201;
  cudaq::Schedule schedule(cudaq::linspace(0.0, 10, num_steps));
  auto initialState = cudaq::state::from_data(
      std::make_pair(initial_state_vec.data(), initial_state_vec.size()));

  auto result = cudaq::evolve_single(hamiltonian, dims, schedule, initialState,
                                     {}, {&cavity_occ_op, &atom_occ_op}, true);
  EXPECT_TRUE(result.get_expectation_values().has_value());
  EXPECT_EQ(result.get_expectation_values().value().size(), num_steps);
  // std::vector<double> theoryResults;
  // for (const auto &t : schedule) {
  //   const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t);
  //   theoryResults.emplace_back(expected);
  // }

  int count = 0;
  for (auto expVals : result.get_expectation_values().value()) {
    EXPECT_EQ(expVals.size(), 2);
    std::cout << expVals[0] << "\n";
    // EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}
