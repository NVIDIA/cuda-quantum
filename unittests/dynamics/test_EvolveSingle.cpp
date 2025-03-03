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
#include "cudaq/dynamics_integrators.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>

TEST(EvolveTester, checkSimple) {
  const std::map<int, int> dims = {{0, 2}};
  cudaq::product_operator<cudaq::matrix_operator> ham1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_operator::x(0));
  cudaq::operator_sum<cudaq::matrix_operator> ham(ham1);

  constexpr int numSteps = 10;
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 1.0, numSteps)) {
    steps.emplace_back(t, 0.0);
  }
  cudaq::Schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &val) { return val; });

  cudaq::product_operator<cudaq::matrix_operator> pauliZ_t =
      cudaq::spin_operator::z(0);
  cudaq::operator_sum<cudaq::matrix_operator> pauliZ(pauliZ_t);
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

  auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
  integrator->dt = 0.001;
  integrator->order = 1;
  auto result = cudaq::__internal__::evolveSingle(
      ham, dims, schedule, initialState, *integrator, {}, {pauliZ}, true);
  EXPECT_NE(result.get_expectation_values().size(), 0);
  EXPECT_EQ(result.get_expectation_values().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.get_expectation_values()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveTester, checkSimpleRK4) {
  const std::map<int, int> dims = {{0, 2}};
  cudaq::product_operator<cudaq::matrix_operator> ham1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_operator::x(0));
  cudaq::operator_sum<cudaq::matrix_operator> ham(ham1);

  constexpr int numSteps = 10;
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 1.0, numSteps)) {
    steps.emplace_back(t, 0.0);
  }
  cudaq::Schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &val) { return val; });

  cudaq::product_operator<cudaq::matrix_operator> pauliZ_t =
      cudaq::spin_operator::z(0);
  cudaq::operator_sum<cudaq::matrix_operator> pauliZ(pauliZ_t);
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

  auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
  integrator->dt = 0.001;
  integrator->order = 4;
  auto result = cudaq::__internal__::evolveSingle(
      ham, dims, schedule, initialState, *integrator, {}, {pauliZ}, true);
  EXPECT_NE(result.get_expectation_values().size(), 0);
  EXPECT_EQ(result.get_expectation_values().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.get_expectation_values()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveTester, checkDensityMatrixSimple) {
  const std::map<int, int> dims = {{0, 2}};
  cudaq::product_operator<cudaq::matrix_operator> ham1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_operator::x(0));
  cudaq::operator_sum<cudaq::matrix_operator> ham(ham1);

  constexpr int numSteps = 10;
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 1.0, numSteps)) {
    steps.emplace_back(t, 0.0);
  }
  cudaq::Schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &val) { return val; });

  cudaq::product_operator<cudaq::matrix_operator> pauliZ_t =
      cudaq::spin_operator::z(0);
  cudaq::operator_sum<cudaq::matrix_operator> pauliZ(pauliZ_t);
  auto initialState = cudaq::state::from_data(
      std::vector<std::complex<double>>{1.0, 0.0, 0.0, 0.0});

  auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
  integrator->dt = 0.001;
  integrator->order = 1;
  auto result = cudaq::__internal__::evolveSingle(
      ham, dims, schedule, initialState, *integrator, {}, {pauliZ}, true);
  EXPECT_NE(result.get_expectation_values().size(), 0);
  EXPECT_EQ(result.get_expectation_values().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.get_expectation_values()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveTester, checkCompositeSystem) {
  constexpr int cavity_levels = 10;
  const std::map<int, int> dims = {{0, 2}, {1, cavity_levels}};
  auto a = cudaq::boson_operator::annihilate(1);
  auto a_dag = cudaq::boson_operator::create(1);

  auto sm = cudaq::boson_operator::annihilate(0);
  auto sm_dag = cudaq::boson_operator::create(0);

  cudaq::product_operator<cudaq::matrix_operator> atom_occ_op_t =
      cudaq::matrix_operator::number(0);
  cudaq::operator_sum<cudaq::matrix_operator> atom_occ_op(atom_occ_op_t);

  cudaq::product_operator<cudaq::matrix_operator> cavity_occ_op_t =
      cudaq::matrix_operator::number(1);
  cudaq::operator_sum<cudaq::matrix_operator> cavity_occ_op(cavity_occ_op_t);

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
  constexpr int num_steps = 21;
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 1.0, num_steps)) {
    steps.emplace_back(t, 0.0);
  }
  cudaq::Schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &val) { return val; });
  auto initialState = cudaq::state::from_data(
      std::make_pair(initial_state_vec.data(), initial_state_vec.size()));
  auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
  integrator->dt = 0.001;
  integrator->order = 4;

  auto result = cudaq::__internal__::evolveSingle(
      hamiltonian, dims, schedule, initialState, *integrator, {},
      {cavity_occ_op, atom_occ_op}, true);
  EXPECT_NE(result.get_expectation_values().size(), 0);
  EXPECT_EQ(result.get_expectation_values().size(), num_steps);

  int count = 0;
  for (auto expVals : result.get_expectation_values()) {
    EXPECT_EQ(expVals.size(), 2);
    std::cout << expVals[0] << " | ";
    std::cout << expVals[1] << "\n";
    // This should be an exchanged interaction
    EXPECT_NEAR((double)expVals[0] + (double)expVals[1], num_photons, 1e-2);
  }
}

TEST(EvolveTester, checkCompositeSystemWithCollapse) {
  constexpr int cavity_levels = 10;
  const std::map<int, int> dims = {{0, 2}, {1, cavity_levels}};
  auto a = cudaq::boson_operator::annihilate(1);
  auto a_dag = cudaq::boson_operator::create(1);

  auto sm = cudaq::boson_operator::annihilate(0);
  auto sm_dag = cudaq::boson_operator::create(0);

  cudaq::product_operator<cudaq::matrix_operator> atom_occ_op_t =
      cudaq::matrix_operator::number(0);
  cudaq::operator_sum<cudaq::matrix_operator> atom_occ_op(atom_occ_op_t);

  cudaq::product_operator<cudaq::matrix_operator> cavity_occ_op_t =
      cudaq::matrix_operator::number(1);
  cudaq::operator_sum<cudaq::matrix_operator> cavity_occ_op(cavity_occ_op_t);

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
  Eigen::MatrixXcd rho0 = initial_state_vec * initial_state_vec.transpose();
  std::cout << "Initial rho:\n" << rho0 << "\n";
  constexpr int num_steps = 11;
  std::vector<std::complex<double>> timeSteps;
  for (double t : cudaq::linspace(0.0, 1.0, num_steps)) {
    timeSteps.emplace_back(t, 0.0);
  }
  cudaq::Schedule schedule(
      timeSteps, {"t"},
      [](const std::string &, const std::complex<double> &val) { return val; });
  auto initialState =
      cudaq::state::from_data(std::make_pair(rho0.data(), rho0.size()));
  auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
  integrator->dt = 0.001;
  integrator->order = 4;
  constexpr double decayRate = 0.1;
  cudaq::product_operator<cudaq::matrix_operator> collapsedOp_t =
      std::sqrt(decayRate) * a;
  cudaq::operator_sum<cudaq::matrix_operator> collapsedOp(collapsedOp_t);
  cudaq::evolve_result result = cudaq::__internal__::evolveSingle(
      hamiltonian, dims, schedule, initialState, *integrator, {collapsedOp},
      {cavity_occ_op, atom_occ_op}, true);
  EXPECT_NE(result.get_expectation_values().size(), 0);
  EXPECT_EQ(result.get_expectation_values().size(), num_steps);

  int count = 0;
  for (auto expVals : result.get_expectation_values()) {
    EXPECT_EQ(expVals.size(), 2);
    const double totalParticleCount = expVals[0] + expVals[1];
    const auto time = timeSteps[count++];
    const double expectedResult =
        num_photons * std::exp(-decayRate * time.real());
    std::cout << "t = " << time << "; particle count = " << totalParticleCount
              << " vs " << expectedResult << "\n";
    EXPECT_NEAR(totalParticleCount, expectedResult, 0.1);
  }
}

TEST(EvolveTester, checkScalarTd) {
  const std::map<int, int> dims = {{0, 10}};

  constexpr int numSteps = 101;
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 10.0, numSteps)) {
    steps.emplace_back(t, 0.0);
  }
  cudaq::Schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &val) { return val; });

  auto function = [](const std::unordered_map<std::string, std::complex<double>>
                         &parameters) {
    auto entry = parameters.find("t");
    if (entry == parameters.end())
      throw std::runtime_error("Cannot find value of expected parameter");
    return 1.0;
  };
  cudaq::product_operator<cudaq::matrix_operator> ham1 =
      cudaq::scalar_operator(function) * cudaq::boson_operator::number(0);
  cudaq::operator_sum<cudaq::matrix_operator> ham(ham1);
  cudaq::product_operator<cudaq::matrix_operator> obs1 =
      cudaq::boson_operator::number(0);
  cudaq::operator_sum<cudaq::matrix_operator> obs(obs1);
  const double decayRate = 0.1;
  cudaq::product_operator<cudaq::matrix_operator> collapseOp1 =
      std::sqrt(decayRate) * cudaq::boson_operator::annihilate(0);
  cudaq::operator_sum<cudaq::matrix_operator> collapseOp(collapseOp1);
  Eigen::VectorXcd initial_state_vec = Eigen::VectorXcd::Zero(10);
  initial_state_vec[9] = 1.0;
  Eigen::MatrixXcd rho0 = initial_state_vec * initial_state_vec.transpose();
  auto initialState =
      cudaq::state::from_data(std::make_pair(rho0.data(), rho0.size()));
  auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
  integrator->dt = 0.001;
  integrator->order = 4;
  auto result =
      cudaq::__internal__::evolveSingle(ham, dims, schedule, initialState,
                                        *integrator, {collapseOp}, {obs}, true);
  EXPECT_NE(result.get_expectation_values().size(), 0);
  EXPECT_EQ(result.get_expectation_values().size(), numSteps);
  std::vector<double> theoryResults;
  int idx = 0;
  for (const auto &t : schedule) {
    const double expected = 9.0 * std::exp(-decayRate * steps[idx++].real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.get_expectation_values()) {
    EXPECT_EQ(expVals.size(), 1);
    std::cout << "Result = " << (double)expVals[0]
              << "; expected = " << theoryResults[count] << "\n";
    EXPECT_NEAR((double)expVals[0], theoryResults[count], 1e-3);
    count++;
  }
}