// /*******************************************************************************
//  * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates. *
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

TEST(EvolveTester, checkSimple) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);

  constexpr int numSteps = 10;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

  cudaq::integrators::runge_kutta integrator(1, 0.001);
  auto result = cudaq::__internal__::evolveSingle(
      ham, dims, schedule, initialState, integrator, {}, {pauliZ},
      cudaq::IntermediateResultSave::All);
  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveTester, checkSimpleRK4) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);

  constexpr int numSteps = 10;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

  cudaq::integrators::runge_kutta integrator(4, 0.001);
  auto result = cudaq::__internal__::evolveSingle(
      ham, dims, schedule, initialState, integrator, {}, {pauliZ},
      cudaq::IntermediateResultSave::All);
  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveTester, checkDensityMatrixSimple) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);

  constexpr int numSteps = 10;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);
  auto initialState = cudaq::state::from_data(
      std::vector<std::complex<double>>{1.0, 0.0, 0.0, 0.0});

  cudaq::integrators::runge_kutta integrator(1, 0.001);
  auto result = cudaq::__internal__::evolveSingle(
      ham, dims, schedule, initialState, integrator, {}, {pauliZ},
      cudaq::IntermediateResultSave::All);
  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveTester, checkCompositeSystem) {
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
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, num_steps);
  cudaq::schedule schedule(steps, {"t"});
  auto initialState = cudaq::state::from_data(
      std::make_pair(initial_state_vec.data(), initial_state_vec.size()));
  cudaq::integrators::runge_kutta integrator(4, 0.001);

  auto result = cudaq::__internal__::evolveSingle(
      hamiltonian, dims, schedule, initialState, integrator, {},
      {cavity_occ_op, atom_occ_op}, cudaq::IntermediateResultSave::All);
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

TEST(EvolveTester, checkCompositeSystemWithCollapse) {
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
  std::vector<double> timeSteps = cudaq::linspace(0.0, 1.0, num_steps);
  cudaq::schedule schedule(timeSteps, {"t"});
  auto initialState =
      cudaq::state::from_data(std::make_pair(rho0.data(), rho0.size()));
  cudaq::integrators::runge_kutta integrator(4, 0.001);
  constexpr double decayRate = 0.1;
  cudaq::product_op<cudaq::matrix_handler> collapsedOp_t =
      std::sqrt(decayRate) * a;
  cudaq::sum_op<cudaq::matrix_handler> collapsedOp(collapsedOp_t);
  cudaq::evolve_result result = cudaq::__internal__::evolveSingle(
      hamiltonian, dims, schedule, initialState, integrator, {collapsedOp},
      {cavity_occ_op, atom_occ_op}, cudaq::IntermediateResultSave::All);
  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), num_steps);

  int count = 0;
  for (auto expVals : result.expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 2);
    const double totalParticleCount = expVals[0] + expVals[1];
    const auto time = timeSteps[count++];
    const double expectedResult = num_photons * std::exp(-decayRate * time);
    std::cout << "t = " << time << "; particle count = " << totalParticleCount
              << " vs " << expectedResult << "\n";
    EXPECT_NEAR(totalParticleCount, expectedResult, 0.1);
  }
}

TEST(EvolveTester, checkScalarTd) {
  const cudaq::dimension_map dims = {{0, 10}};

  constexpr int numSteps = 101;
  std::vector<double> steps = cudaq::linspace(0.0, 10.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  auto function = [](const std::unordered_map<std::string, std::complex<double>>
                         &parameters) {
    auto entry = parameters.find("t");
    if (entry == parameters.end())
      throw std::runtime_error("Cannot find value of expected parameter");
    return 1.0;
  };
  cudaq::product_op<cudaq::matrix_handler> ham1 =
      cudaq::scalar_operator(function) * cudaq::boson_op::number(0);
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);
  cudaq::product_op<cudaq::matrix_handler> obs1 = cudaq::boson_op::number(0);
  cudaq::sum_op<cudaq::matrix_handler> obs(obs1);
  const double decayRate = 0.1;
  cudaq::product_op<cudaq::matrix_handler> collapseOp1 =
      std::sqrt(decayRate) * cudaq::boson_op::annihilate(0);
  cudaq::sum_op<cudaq::matrix_handler> collapseOp(collapseOp1);
  Eigen::VectorXcd initial_state_vec = Eigen::VectorXcd::Zero(10);
  initial_state_vec[9] = 1.0;
  Eigen::MatrixXcd rho0 = initial_state_vec * initial_state_vec.transpose();
  auto initialState =
      cudaq::state::from_data(std::make_pair(rho0.data(), rho0.size()));
  cudaq::integrators::runge_kutta integrator(4, 0.001);
  auto result = cudaq::__internal__::evolveSingle(
      ham, dims, schedule, initialState, integrator, {collapseOp}, {obs},
      cudaq::IntermediateResultSave::All);
  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), numSteps);
  std::vector<double> theoryResults;
  int idx = 0;
  for (const auto &t : schedule) {
    const double expected = 9.0 * std::exp(-decayRate * steps[idx++]);
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    std::cout << "Result = " << (double)expVals[0]
              << "; expected = " << theoryResults[count] << "\n";
    EXPECT_NEAR((double)expVals[0], theoryResults[count], 1e-3);
    count++;
  }
}

TEST(EvolveTester, checkSimpleNoIntermediateResults) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);

  constexpr int numSteps = 10;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

  cudaq::integrators::runge_kutta integrator(1, 0.001);
  auto result = cudaq::__internal__::evolveSingle(
      ham, dims, schedule, initialState, integrator, {}, {pauliZ},
      cudaq::IntermediateResultSave::None);

  // Verify final expectation value only (no intermediate results)
  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), 1);
  EXPECT_EQ(result.expectation_values.value()[0].size(), 1);

  const double finalTime = steps.back();
  const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * finalTime);
  EXPECT_NEAR(result.expectation_values.value()[0][0], expected, 1e-3);
}

TEST(EvolveTester, checkDensityMatrixNoIntermediateResults) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);

  constexpr int numSteps = 10;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);
  auto initialState = cudaq::state::from_data(
      std::vector<std::complex<double>>{1.0, 0.0, 0.0, 0.0});

  cudaq::integrators::runge_kutta integrator(1, 0.001);
  auto result = cudaq::__internal__::evolveSingle(
      ham, dims, schedule, initialState, integrator, {}, {pauliZ},
      cudaq::IntermediateResultSave::None);

  // Verify final expectation value only (no intermediate results)
  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), 1);
  EXPECT_EQ(result.expectation_values.value()[0].size(), 1);

  const double finalTime = steps.back();
  const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * finalTime);
  EXPECT_NEAR(result.expectation_values.value()[0][0], expected, 1e-3);
}
