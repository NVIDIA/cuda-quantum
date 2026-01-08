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

TEST(BatchedEvolveTester, checkBatchedDifferentCollapseOps) {
  auto annihilate_matrix =
      [](const std::vector<int64_t> &dimensions,
         const std::unordered_map<std::string, std::complex<double>>
             &parameters) -> cudaq::complex_matrix {
    std::size_t dimension = dimensions[0];
    auto annihilate = cudaq::complex_matrix(dimension, dimension);
    for (std::size_t i = 0; i + 1 < dimension; i++) {
      annihilate[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1));
    }
    return annihilate;
  };

  cudaq::matrix_handler::define("my_annihilate_op", {-1}, annihilate_matrix);

  auto annihilate_op = [](std::size_t degree) {
    return cudaq::matrix_handler::instantiate("my_annihilate_op", {degree});
  };

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
  int i = 0;
  for (const auto &decayRate : decayRates) {
    // Same hamiltonian, but different collapse operators
    batchedHams.emplace_back(cudaq::sum_op<cudaq::matrix_handler>(hamiltonian));
    // We alternately use boson annihilate and custom annihilate
    // operator to test the batching of different collapse operators.
    if (i % 2 == 0) {
      batchedCollapsedOps.emplace_back(
          std::vector<cudaq::sum_op<cudaq::matrix_handler>>{
              cudaq::sum_op<cudaq::matrix_handler>(
                  std::sqrt(decayRate) * cudaq::boson_op::annihilate(0))});
    } else {
      batchedCollapsedOps.emplace_back(
          std::vector<cudaq::sum_op<cudaq::matrix_handler>>{
              cudaq::sum_op<cudaq::matrix_handler>(std::sqrt(decayRate) *
                                                   annihilate_op(0))});
    }
    i++;
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

TEST(BatchedEvolveTester, checkCallbackTensorOpSimple) {
  auto tensorFunction =
      [](const std::vector<int64_t> &dimensions,
         const std::unordered_map<std::string, std::complex<double>>
             &parameters) -> cudaq::complex_matrix {
    cudaq::complex_matrix mat(2, 2);
    mat[{0, 0}] = 0.0;
    mat[{0, 1}] = 1.0;
    mat[{1, 0}] = 1.0;
    mat[{1, 1}] = 0.0;
    return mat;
  };

  cudaq::matrix_handler::define("CustomPauliX", {2}, tensorFunction);
  const std::vector<double> resonanceFreqs = {0.05, 0.1, 0.15, 0.2,
                                              0.25, 0.3, 0.35, 0.4};
  std::vector<cudaq::sum_op<cudaq::matrix_handler>> batchedHams;
  std::vector<cudaq::state> initialStates;
  for (const auto &resonanceFreq : resonanceFreqs) {
    batchedHams.emplace_back(cudaq::sum_op<cudaq::matrix_handler>(
        2.0 * M_PI * resonanceFreq *
        cudaq::matrix_handler::instantiate("CustomPauliX", {0})));
    initialStates.emplace_back(
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0}));
  }

  const cudaq::dimension_map dims = {{0, 2}};
  constexpr int numSteps = 100;
  std::vector<double> steps = cudaq::linspace(0.0, 4.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::integrators::runge_kutta integrator(4);
  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);

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

TEST(BatchedEvolveTester, checkCallbackTensorOpDifferentFuncs) {
  auto tensorFunction =
      [](const std::vector<int64_t> &dimensions,
         const std::unordered_map<std::string, std::complex<double>>
             &parameters,
         double resonantFreq) -> cudaq::complex_matrix {
    cudaq::complex_matrix mat(2, 2);
    mat[{0, 0}] = 0.0;
    mat[{0, 1}] = 2.0 * M_PI * resonantFreq;
    mat[{1, 0}] = 2.0 * M_PI * resonantFreq;
    mat[{1, 1}] = 0.0;
    return mat;
  };

  const std::vector<double> resonanceFreqs = {0.05, 0.1, 0.15, 0.2,
                                              0.25, 0.3, 0.35, 0.4};
  std::vector<cudaq::sum_op<cudaq::matrix_handler>> batchedHams;
  std::vector<cudaq::state> initialStates;
  int count = 0;
  for (const auto &resonanceFreq : resonanceFreqs) {
    const std::string opName = "CustomPauliX_" + std::to_string(count++);
    cudaq::matrix_handler::define(
        opName, {2},
        [&](const std::vector<int64_t> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters) {
          return tensorFunction(dimensions, parameters, resonanceFreq);
        });

    batchedHams.emplace_back(cudaq::sum_op<cudaq::matrix_handler>(
        cudaq::matrix_handler::instantiate(opName, {0})));
    initialStates.emplace_back(
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0}));
  }

  const cudaq::dimension_map dims = {{0, 2}};
  constexpr int numSteps = 100;
  std::vector<double> steps = cudaq::linspace(0.0, 4.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::integrators::runge_kutta integrator(4);
  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);

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
      std::cout << "Freq = " << resonanceFreqs[i]
                << "; Result = " << (double)expVals[0]
                << "; Expected = " << theoryResults[count - 1][i] << "\n";
    }
  }
}

TEST(BatchedEvolveTester, checkSuperopSimple) {
  const cudaq::dimension_map dims = {{0, 2}};
  const std::vector<double> resonanceFreqs = {0.05, 0.1, 0.15, 0.2,
                                              0.25, 0.3, 0.35, 0.4};
  std::vector<cudaq::super_op> sups;
  std::vector<cudaq::state> initialStates;
  for (const auto &resonanceFreq : resonanceFreqs) {
    cudaq::product_op<cudaq::matrix_handler> ham_ =
        (2.0 * M_PI * resonanceFreq * cudaq::spin_op::x(0));
    cudaq::sum_op<cudaq::matrix_handler> ham(ham_);
    cudaq::super_op sup;
    // Apply `-iH * psi` superop
    sup +=
        cudaq::super_op::left_multiply(std::complex<double>(0.0, -1.0) * ham);
    sups.emplace_back(sup);
    initialStates.emplace_back(
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0}));
  }

  cudaq::integrators::runge_kutta integrator(4);
  constexpr int numSteps = 10;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  cudaq::product_op<cudaq::matrix_handler> pauliZ_t = cudaq::spin_op::z(0);
  cudaq::sum_op<cudaq::matrix_handler> pauliZ(pauliZ_t);
  auto results = cudaq::__internal__::evolveBatched(
      sups, dims, schedule, initialStates, integrator, {pauliZ},
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
      std::cout << "Freq = " << resonanceFreqs[i]
                << "; Result = " << (double)expVals[0]
                << "; Expected = " << theoryResults[count - 1][i] << "\n";
    }
  }
}

TEST(BatchedEvolveTester, checkSuperopMasterEquation) {
  constexpr int N = 10;
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto ham = cudaq::boson_op::number(0);
  const cudaq::dimension_map dimensions{{0, N}};
  std::vector<std::complex<double>> rho0_(N * N, 0.0);
  rho0_.back() = 1.0;
  const std::vector<double> decayRates{0.05, 0.1, 0.15, 0.2,
                                       0.25, 0.3, 0.35, 0.4};

  std::vector<cudaq::super_op> batchedSups;
  std::vector<cudaq::state> initialStates;

  for (const auto &decayRate : decayRates) {
    // Same hamiltonian, but different collapse operators
    cudaq::super_op sup;
    // Apply `-i[H, rho]` superop
    sup +=
        cudaq::super_op::left_multiply(std::complex<double>(0.0, -1.0) * ham);
    sup +=
        cudaq::super_op::right_multiply(std::complex<double>(0.0, 1.0) * ham);

    auto td_function =
        [decayRate](const std::unordered_map<std::string, std::complex<double>>
                        &parameters) {
          auto entry = parameters.find("t");
          if (entry == parameters.end())
            throw std::runtime_error("Cannot find value of expected parameter");
          const auto t = entry->second.real();
          const auto result = std::sqrt(decayRate * std::exp(-t));
          return result;
        };

    auto L =
        cudaq::scalar_operator(td_function) * cudaq::boson_op::annihilate(0);
    auto L_dagger =
        cudaq::scalar_operator(td_function) * cudaq::boson_op::create(0);
    // Lindblad terms
    // L * rho * L_dagger
    sup += cudaq::super_op::left_right_multiply(L, L_dagger);
    // -0.5 * L_dagger * L * rho
    sup += cudaq::super_op::left_multiply(-0.5 * L_dagger * L);
    // -0.5 * rho * L_dagger * L
    sup += cudaq::super_op::right_multiply(-0.5 * L_dagger * L);

    batchedSups.emplace_back(sup);
    initialStates.emplace_back(cudaq::state::from_data(rho0_));
  }

  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto results = cudaq::__internal__::evolveBatched(
      batchedSups, dimensions, schedule, initialStates, integrator,
      {cudaq::sum_op<cudaq::matrix_handler>(ham)},
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
      std::cout << "Decay Rate = " << decayRates[i]
                << "; Result = " << (double)expVals[0]
                << "; Expected = " << theoryResults[count - 1][i] << "\n";
    }
  }
}

TEST(BatchedEvolveTester, checkParamSweep) {
  const cudaq::dimension_map dims = {{0, 3}};
  std::vector<double> amplitudes = cudaq::linspace(10.0, 30.0, 128);
  std::vector<double> dragAmplitudes = cudaq::linspace(100, 150, 128);
  const double sigma = 0.01;         // sigma of the Gaussian pulse
  const double cutoff = 4.0 * sigma; // total length of drive pulse

  const auto gaussian =
      [sigma,
       cutoff](const std::unordered_map<std::string, std::complex<double>>
                   &parameters) {
        auto entry = parameters.find("t");
        if (entry == parameters.end())
          throw std::runtime_error("Cannot find value of expected parameter");
        const auto t = entry->second.real();
        const auto val =
            (std::exp(-((t - cutoff / 2) / sigma) * ((t - cutoff / 2) / sigma) /
                      2) -
             std::exp(-(cutoff / sigma) * (cutoff / sigma) / 8)) /
            (1 - std::exp(-(cutoff / sigma) * (cutoff / sigma) / 8));
        return val;
      };

  const auto drag_gaussian =
      [sigma,
       cutoff](const std::unordered_map<std::string, std::complex<double>>
                   &parameters) {
        auto entry = parameters.find("t");
        if (entry == parameters.end())
          throw std::runtime_error("Cannot find value of expected parameter");
        const auto t = entry->second.real();
        const auto val = -((t - cutoff / 2) / sigma) *
                         std::exp(-((t - cutoff / 2) / sigma) *
                                      ((t - cutoff / 2) / sigma) / 2 +
                                  0.5);
        return val;
      };
  constexpr int numSteps = 201;
  std::vector<double> steps = cudaq::linspace(0.0, cutoff, numSteps);
  cudaq::schedule schedule(steps, {"t"});
  cudaq::state targetState =
      cudaq::state::from_data(std::vector<std::complex<double>>{
          1.0 / std::sqrt(2.0),
          std::complex<double>(0.0, -1.0 / std::sqrt(2.0)), 0.0});
  std::vector<cudaq::sum_op<cudaq::matrix_handler>> batchedHams;
  std::vector<cudaq::state> initialStates;
  for (const auto &amplitude : amplitudes) {
    for (const auto &dragAmplitude : dragAmplitudes) {
      batchedHams.emplace_back(-cudaq::sum_op<cudaq::matrix_handler>(
          amplitude * cudaq::scalar_operator(gaussian) *
              (cudaq::boson::create(0) + cudaq::boson::annihilate(0)) -
          std::complex<double>(0.0, 1.0) * dragAmplitude *
              cudaq::scalar_operator(drag_gaussian) *
              (cudaq::boson::annihilate(0) - cudaq::boson::create(0))));
      initialStates.emplace_back(cudaq::state::from_data(
          std::vector<std::complex<double>>{1.0, 0.0, 0.0}));
    }
  }

  cudaq::integrators::runge_kutta integrator(4);
  auto results = cudaq::__internal__::evolveBatched(
      batchedHams, dims, schedule, initialStates, integrator, {}, {},
      cudaq::IntermediateResultSave::None);
  EXPECT_EQ(results.size(), amplitudes.size() * dragAmplitudes.size());
  int count = 0;
  double maxOverlap = 0.0;

  for (const auto &amplitude : amplitudes) {
    for (const auto &dragAmplitude : dragAmplitudes) {
      const auto &result = results[count++];
      EXPECT_TRUE(result.states.has_value());
      EXPECT_TRUE(result.states.value().size() == 1);
      const auto overlap = targetState.overlap(result.states.value()[0]);
      EXPECT_GT(overlap.real(), 0.0);
      if (overlap.real() > maxOverlap) {
        maxOverlap = overlap.real();
      }
    }
  }

  EXPECT_GT(maxOverlap, 0.98); // Expect a high overlap with the target state
}

TEST(BatchedEvolveTester, checkIntermediateResultSaveAll) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham_1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham1(ham_1);

  cudaq::product_op<cudaq::matrix_handler> ham_2 =
      (2.0 * M_PI * 0.2 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham2(ham_2);

  constexpr int numSteps = 5; // Use fewer steps for this test
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
      {}, {pauliZ}, cudaq::IntermediateResultSave::All);

  EXPECT_EQ(results.size(), 2);

  // Check that both results have states and expectation values
  for (const auto &result : results) {
    EXPECT_TRUE(result.states.has_value());
    EXPECT_TRUE(result.expectation_values.has_value());

    // For IntermediateResultSave::All, we should have states for each time step
    EXPECT_EQ(result.states.value().size(), numSteps);
    EXPECT_EQ(result.expectation_values.value().size(), numSteps);

    // Check that each time step has expectation values
    for (const auto &expVals : result.expectation_values.value()) {
      EXPECT_EQ(expVals.size(), 1); // One observable (pauliZ)
    }
  }
}

TEST(BatchedEvolveTester, checkIntermediateResultSaveNoneWithObservables) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham_1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham1(ham_1);

  cudaq::product_op<cudaq::matrix_handler> ham_2 =
      (2.0 * M_PI * 0.2 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham2(ham_2);

  constexpr int numSteps = 3; // Use fewer steps for this test
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
      {}, {pauliZ}, cudaq::IntermediateResultSave::None);

  EXPECT_EQ(results.size(), 2);

  // Check that both results have final states and final expectation values
  for (const auto &result : results) {
    EXPECT_TRUE(result.states.has_value());
    EXPECT_TRUE(result.expectation_values.has_value());

    // For IntermediateResultSave::None, we should have only final state
    EXPECT_EQ(result.states.value().size(), 1);
    EXPECT_EQ(result.expectation_values.value().size(), 1);

    // Check that final expectation value is computed
    EXPECT_EQ(result.expectation_values.value()[0].size(),
              1); // One observable (pauliZ)
  }
}

// Test to reproduce coefficient mismatch bug when batched operators are sorted
// by degrees but coefficients are taken from unsorted ops.
// Bug: CuDensityMatOpConverter.cpp:528-529 uses ops[termIdx] instead of
// batchedProductTerms[i][termIdx]
TEST(BatchedEvolveTester, checkCoefficientMismatchAfterSorting) {
  // Use 2 qubits (degrees 0 and 1) to ensure sorting changes term order
  const cudaq::dimension_map dims = {{0, 2}, {1, 2}};

  // We'll construct two Hamiltonians where terms are ordered such that sorting
  // by degrees will reorder them. Then we use X operators (non-diagonal) so
  // that different coefficients lead to measurably different evolution.
  //
  // Hamiltonian structure (before sorting):
  //   term[0]: coeff1 * X(1)  -> degrees = {1}
  //   term[1]: coeff0 * X(0)  -> degrees = {0}
  //
  // After stable_sort by degrees:
  //   term[0]: coeff0 * X(0)  -> degrees = {0}
  //   term[1]: coeff1 * X(1)  -> degrees = {1}
  //
  // Bug: coeffs[i] = ops[i][termIdx].get_coefficient() uses unsorted index
  //      but prodTerms[i] = batchedProductTerms[i][termIdx] uses sorted index
  //      This causes coefficient mismatch!

  // Batch 1: H1 = f1 * X(1) + f0 * X(0) where f1=0.2, f0=0.1
  //          After sorting: H1 = f0 * X(0) + f1 * X(1)
  // Batch 2: H2 = g1 * X(1) + g0 * X(0) where g1=0.4, g0=0.3
  //          After sorting: H2 = g0 * X(0) + g1 * X(1)
  //
  // With the bug, when processing sorted term[0] (X(0)):
  //   - prodTerms uses X(0) (correct)
  //   - coeffs uses ops[0] (unsorted) which gives f1=0.2 for batch1, g1=0.4 for
  //   batch2
  //     instead of f0=0.1, g0=0.3

  const double f0 = 0.1, f1 = 0.2;
  const double g0 = 0.3, g1 = 0.4;

  // Use time-dependent callbacks to trigger the non-constant coefficient path
  auto make_td_coeff = [](double val) {
    return cudaq::scalar_operator(
        [val](const std::unordered_map<std::string, std::complex<double>>
                  &parameters) { return std::complex<double>(val, 0.0); });
  };

  // Hamiltonian 1: terms added in order degree1, degree0
  cudaq::sum_op<cudaq::matrix_handler> ham1 =
      make_td_coeff(2.0 * M_PI * f1) * cudaq::spin_op::x(1) +
      make_td_coeff(2.0 * M_PI * f0) * cudaq::spin_op::x(0);

  // Hamiltonian 2: terms added in order degree1, degree0
  cudaq::sum_op<cudaq::matrix_handler> ham2 =
      make_td_coeff(2.0 * M_PI * g1) * cudaq::spin_op::x(1) +
      make_td_coeff(2.0 * M_PI * g0) * cudaq::spin_op::x(0);

  constexpr int numSteps = 50;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  // Initial state: |00> = (1, 0, 0, 0) in computational basis
  // For 2 qubits: |00>, |01>, |10>, |11>
  auto initialState = cudaq::state::from_data(
      std::vector<std::complex<double>>{1.0, 0.0, 0.0, 0.0});

  // Observables: Z(0) and Z(1) measured separately
  cudaq::sum_op<cudaq::matrix_handler> obsZ0(cudaq::spin_op::z(0));
  cudaq::sum_op<cudaq::matrix_handler> obsZ1(cudaq::spin_op::z(1));

  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto results = cudaq::__internal__::evolveBatched(
      {ham1, ham2}, dims, schedule, {initialState, initialState}, integrator,
      {}, {obsZ0, obsZ1}, cudaq::IntermediateResultSave::ExpectationValue);

  EXPECT_EQ(results.size(), 2);

  // For independent X rotations on each qubit:
  // H = omega0 * X(0) + omega1 * X(1)
  // <Z(0)>(t) = cos(2 * omega0 * t)
  // <Z(1)>(t) = cos(2 * omega1 * t)
  //
  // Batch 1: omega0 = 2*pi*f0, omega1 = 2*pi*f1
  //   <Z(0)>(t) = cos(4*pi*f0*t) = cos(4*pi*0.1*t)
  //   <Z(1)>(t) = cos(4*pi*f1*t) = cos(4*pi*0.2*t)
  //
  // Batch 2: omega0 = 2*pi*g0, omega1 = 2*pi*g1
  //   <Z(0)>(t) = cos(4*pi*g0*t) = cos(4*pi*0.3*t)
  //   <Z(1)>(t) = cos(4*pi*g1*t) = cos(4*pi*0.4*t)
  //
  // With the bug (coefficients swapped):
  // Batch 1: omega0 = 2*pi*f1, omega1 = 2*pi*f0 (swapped!)
  //   <Z(0)>(t) = cos(4*pi*0.2*t)  <- wrong!
  //   <Z(1)>(t) = cos(4*pi*0.1*t)  <- wrong!

  // Check batch 1 results
  {
    EXPECT_TRUE(results[0].expectation_values.has_value());
    const auto &expValsList = results[0].expectation_values.value();
    EXPECT_EQ(expValsList.size(), numSteps);

    int count = 0;
    for (auto expVals : expValsList) {
      EXPECT_EQ(expVals.size(), 2); // Two observables
      double t = steps[count];

      // Expected values with CORRECT coefficients
      double expectedZ0 = std::cos(4.0 * M_PI * f0 * t);
      double expectedZ1 = std::cos(4.0 * M_PI * f1 * t);

      // What we'd get with WRONG (swapped) coefficients
      double wrongZ0 = std::cos(4.0 * M_PI * f1 * t);
      double wrongZ1 = std::cos(4.0 * M_PI * f0 * t);

      double actualZ0 = (double)expVals[0];
      double actualZ1 = (double)expVals[1];

      // If bug exists, actualZ0 would be close to wrongZ0 instead of expectedZ0
      bool matchesCorrect = (std::abs(actualZ0 - expectedZ0) < 0.05) &&
                            (std::abs(actualZ1 - expectedZ1) < 0.05);
      bool matchesWrong = (std::abs(actualZ0 - wrongZ0) < 0.05) &&
                          (std::abs(actualZ1 - wrongZ1) < 0.05);

      if (t > 0.1) { // Skip early times where values might be similar
        if (matchesWrong && !matchesCorrect) {
          std::cout << "BUG DETECTED at t=" << t << ": Batch 1\n";
          std::cout << "  <Z(0)> actual=" << actualZ0
                    << ", expected=" << expectedZ0 << ", wrong=" << wrongZ0
                    << "\n";
          std::cout << "  <Z(1)> actual=" << actualZ1
                    << ", expected=" << expectedZ1 << ", wrong=" << wrongZ1
                    << "\n";
        }
      }

      EXPECT_NEAR(actualZ0, expectedZ0, 0.05)
          << "Batch 1, t=" << t << ": <Z(0)> mismatch - coefficient bug?";
      EXPECT_NEAR(actualZ1, expectedZ1, 0.05)
          << "Batch 1, t=" << t << ": <Z(1)> mismatch - coefficient bug?";

      count++;
    }
  }

  // Check batch 2 results
  {
    EXPECT_TRUE(results[1].expectation_values.has_value());
    const auto &expValsList = results[1].expectation_values.value();
    EXPECT_EQ(expValsList.size(), numSteps);

    int count = 0;
    for (auto expVals : expValsList) {
      EXPECT_EQ(expVals.size(), 2);
      double t = steps[count];

      double expectedZ0 = std::cos(4.0 * M_PI * g0 * t);
      double expectedZ1 = std::cos(4.0 * M_PI * g1 * t);

      double actualZ0 = (double)expVals[0];
      double actualZ1 = (double)expVals[1];

      EXPECT_NEAR(actualZ0, expectedZ0, 0.05)
          << "Batch 2, t=" << t << ": <Z(0)> mismatch - coefficient bug?";
      EXPECT_NEAR(actualZ1, expectedZ1, 0.05)
          << "Batch 2, t=" << t << ": <Z(1)> mismatch - coefficient bug?";

      count++;
    }
  }
}
