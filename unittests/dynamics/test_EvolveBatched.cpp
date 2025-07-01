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

TEST(BatchedEvolveTester, checkDifferentOperators) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham_1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::x(0)); // X
  cudaq::sum_op<cudaq::matrix_handler> ham1(ham_1);

  cudaq::product_op<cudaq::matrix_handler> ham_2 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::z(0)); // Z
  cudaq::sum_op<cudaq::matrix_handler> ham2(ham_2);

  constexpr int numSteps = 10;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);

  cudaq::product_op<cudaq::matrix_handler> pauliX_t = cudaq::spin_op::x(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliX(pauliX_t);
  auto initialState1 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  auto initialState2 = cudaq::state::from_data(
      std::vector<std::complex<double>>{M_SQRT1_2, M_SQRT1_2});

  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto results = cudaq::__internal__::evolveBatched(
      {ham1, ham2}, dims, schedule, {initialState1, initialState2}, integrator,
      {}, {pauliZ, pauliX}, cudaq::IntermediateResultSave::ExpectationValue);

  EXPECT_EQ(results.size(), 2);
  EXPECT_TRUE(results[0].expectation_values.has_value());
  EXPECT_EQ(results[0].expectation_values.value().size(), numSteps);
  EXPECT_TRUE(results[1].expectation_values.has_value());
  EXPECT_EQ(results[1].expectation_values.value().size(), numSteps);

  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : results[0].expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 2);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }

  count = 0;
  for (auto expVals : results[1].expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 2);
    EXPECT_NEAR((double)expVals[1], theoryResults[count++], 1e-3);
  }
}

TEST(BatchedEvolveTester, checkBatchedCollapseOps) {
  // Batching the decay rates
  std::vector<double> decayRates = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4};
  constexpr int N = 10;
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto hamiltonian = cudaq::boson_op::number(0);
  const cudaq::dimension_map dimensions{{0, N}};
  std::vector<std::complex<double>> psi0_(N, 0.0);
  psi0_.back() = 1.0;
  auto psi0 = cudaq::state::from_data(psi0_);
  std::vector<cudaq::sum_op<cudaq::matrix_handler>> batchedHams;
  std::vector<std::vector<cudaq::sum_op<cudaq::matrix_handler>>>
      batchedCollapsedOps;
  std::vector<cudaq::state> initialStates;
  for (const auto &decayRate : decayRates) {
    // Same hamiltonian, but different collapse operators
    batchedHams.emplace_back(cudaq::sum_op<cudaq::matrix_handler>(hamiltonian));
    batchedCollapsedOps.emplace_back(
        std::vector<cudaq::sum_op<cudaq::matrix_handler>>{
            cudaq::sum_op<cudaq::matrix_handler>(
                std::sqrt(decayRate) * cudaq::boson_op::annihilate(0))});
    initialStates.emplace_back(psi0);
  }

  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto results = cudaq::__internal__::evolveBatched(
      batchedHams, dimensions, schedule, initialStates, integrator,
      batchedCollapsedOps, {cudaq::sum_op<cudaq::matrix_handler>(hamiltonian)},
      cudaq::IntermediateResultSave::ExpectationValue);
  EXPECT_EQ(results.size(), decayRates.size());
  std::vector<std::vector<double>> theoryResults;
  for (const auto &t : schedule) {
    std::vector<double> expectedResults;
    for (const auto &decayRate : decayRates) {
      expectedResults.emplace_back((N - 1) * std::exp(-decayRate * t.real()));
    }
    theoryResults.emplace_back(expectedResults);
  }

  for (std::size_t i = 0; i < results.size(); ++i) {
    EXPECT_TRUE(results[i].expectation_values.has_value());
    EXPECT_EQ(results[i].expectation_values.value().size(), numSteps);

    int count = 0;
    for (auto expVals : results[i].expectation_values.value()) {
      EXPECT_EQ(expVals.size(), 1);
      EXPECT_NEAR((double)expVals[0], theoryResults[count++][i], 1e-3);
    }
  }
}

TEST(BatchedEvolveTester, checkTimeDependentHamiltonian) {
  const cudaq::dimension_map dims = {{0, 2}};
  std::vector<double> resonanceFreqs = {0.05, 0.1, 0.15, 0.2,
                                        0.25, 0.3, 0.35, 0.4};
  auto td_function =
      [](const std::unordered_map<std::string, std::complex<double>>
             &parameters,
         double f) { return f; };

  std::vector<cudaq::sum_op<cudaq::matrix_handler>> batchedHams;
  std::vector<cudaq::state> initialStates;
  for (const auto &resonanceFreq : resonanceFreqs) {
    batchedHams.emplace_back(cudaq::sum_op<cudaq::matrix_handler>(
        2.0 * M_PI *
        cudaq::scalar_operator(
            [td_function, resonanceFreq](
                const std::unordered_map<std::string, std::complex<double>>
                    &parameters) {
              return td_function(parameters, resonanceFreq);
            }) *
        cudaq::spin_op::x(0)));
    initialStates.emplace_back(
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0}));
  }

  constexpr int numSteps = 100;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);

  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto results = cudaq::__internal__::evolveBatched(
      batchedHams, dims, schedule, initialStates, integrator, {}, {pauliZ},
      cudaq::IntermediateResultSave::ExpectationValue);

  EXPECT_EQ(results.size(), resonanceFreqs.size());
  std::vector<std::vector<double>> theoryResults;
  for (const auto &t : schedule) {
    std::vector<double> expectedResults;
    for (const auto &resonanceFreq : resonanceFreqs) {
      expectedResults.emplace_back(
          std::cos(2 * 2.0 * M_PI * resonanceFreq * t.real()));
    }
    theoryResults.emplace_back(expectedResults);
  }

  for (std::size_t i = 0; i < results.size(); ++i) {
    EXPECT_TRUE(results[i].expectation_values.has_value());
    EXPECT_EQ(results[i].expectation_values.value().size(), numSteps);

    int count = 0;
    for (auto expVals : results[i].expectation_values.value()) {
      EXPECT_EQ(expVals.size(), 1);
      EXPECT_NEAR((double)expVals[0], theoryResults[count++][i], 1e-3);
    }
  }
}

TEST(BatchedEvolveTester, checkTimeDependentCollapsedOps) {
  constexpr int N = 10;
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto hamiltonian = cudaq::boson_op::number(0);
  const cudaq::dimension_map dimensions{{0, N}};
  std::vector<std::complex<double>> psi0_(N, 0.0);
  psi0_.back() = 1.0;
  auto psi0 = cudaq::state::from_data(psi0_);
  std::vector<double> decayRates = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4};

  auto td_function =
      [](const std::unordered_map<std::string, std::complex<double>>
             &parameters,
         double decay_rate) {
        auto entry = parameters.find("t");
        if (entry == parameters.end())
          throw std::runtime_error("Cannot find value of expected parameter");
        const auto t = entry->second.real();
        const auto result = std::sqrt(decay_rate * std::exp(-t));
        return result;
      };

  std::vector<cudaq::sum_op<cudaq::matrix_handler>> batchedHams;
  std::vector<std::vector<cudaq::sum_op<cudaq::matrix_handler>>>
      batchedCollapsedOps;
  std::vector<cudaq::state> initialStates;
  for (const auto &decayRate : decayRates) {
    // Same hamiltonian, but different collapse operators
    batchedHams.emplace_back(cudaq::sum_op<cudaq::matrix_handler>(hamiltonian));
    batchedCollapsedOps.emplace_back(
        std::vector<cudaq::sum_op<cudaq::matrix_handler>>{
            cudaq::sum_op<cudaq::matrix_handler>(
                cudaq::scalar_operator(
                    [td_function, decayRate](
                        const std::unordered_map<
                            std::string, std::complex<double>> &parameters) {
                      return td_function(parameters, decayRate);
                    }) *
                cudaq::boson_op::annihilate(0))});
    initialStates.emplace_back(psi0);
  }

  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto results = cudaq::__internal__::evolveBatched(
      batchedHams, dimensions, schedule, initialStates, integrator,
      batchedCollapsedOps, {cudaq::sum_op<cudaq::matrix_handler>(hamiltonian)},
      cudaq::IntermediateResultSave::ExpectationValue);

  EXPECT_EQ(results.size(), decayRates.size());
  std::vector<std::vector<double>> theoryResults;
  for (const auto &t : schedule) {
    std::vector<double> expectedResults;
    for (const auto &decayRate : decayRates) {
      expectedResults.emplace_back(
          (N - 1) * std::exp(-decayRate * (1.0 - std::exp(-t.real()))));
    }
    theoryResults.emplace_back(expectedResults);
  }

  for (std::size_t i = 0; i < results.size(); ++i) {
    EXPECT_TRUE(results[i].expectation_values.has_value());
    EXPECT_EQ(results[i].expectation_values.value().size(), numSteps);

    int count = 0;
    for (auto expVals : results[i].expectation_values.value()) {
      EXPECT_EQ(expVals.size(), 1);
      EXPECT_NEAR((double)expVals[0], theoryResults[count++][i], 1e-3);
    }
  }
}
