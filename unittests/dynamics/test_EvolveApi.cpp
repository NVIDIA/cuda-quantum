// /*******************************************************************************
//  * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include <cmath>
#include <complex>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

TEST(EvolveAPITester, checkSimple) {
  const cudaq::dimension_map dims = {{0, 2}};
  auto ham = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
  constexpr int numSteps = 10;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  cudaq::integrators::runge_kutta integrator(1, 0.001);
  auto result = cudaq::evolve(ham, dims, schedule, initialState, integrator, {},
                              {cudaq::spin_op::z(0)},
                              cudaq::IntermediateResultSave::ExpectationValue);
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

TEST(EvolveAPITester, checkCavityModel) {
  constexpr int N = 10;
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto hamiltonian = cudaq::boson_op::number(0);
  const cudaq::dimension_map dimensions{{0, N}};
  std::vector<std::complex<double>> psi0_(N, 0.0);
  psi0_.back() = 1.0;
  auto psi0 = cudaq::state::from_data(psi0_);
  constexpr double decay_rate = 0.1;
  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto result = cudaq::evolve(
      hamiltonian, dimensions, schedule, psi0, integrator,
      {std::sqrt(decay_rate) * cudaq::boson_op::annihilate(0)}, {hamiltonian},
      cudaq::IntermediateResultSave::ExpectationValue);
  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = (N - 1) * std::exp(-decay_rate * t.real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveAPITester, checkTimeDependent) {
  constexpr int N = 10;
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto hamiltonian = cudaq::boson_op::number(0);
  const cudaq::dimension_map dimensions{{0, N}};
  std::vector<std::complex<double>> psi0_(N, 0.0);
  psi0_.back() = 1.0;
  auto psi0 = cudaq::state::from_data(psi0_);
  constexpr double decay_rate = 0.1;

  auto td_function =
      [decay_rate](const std::unordered_map<std::string, std::complex<double>>
                       &parameters) {
        auto entry = parameters.find("t");
        if (entry == parameters.end())
          throw std::runtime_error("Cannot find value of expected parameter");
        const auto t = entry->second.real();
        const auto result = std::sqrt(decay_rate * std::exp(-t));
        return result;
      };

  auto collapseOperator =
      cudaq::scalar_operator(td_function) * cudaq::boson_op::annihilate(0);
  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto result = cudaq::evolve(hamiltonian, dimensions, schedule, psi0,
                              integrator, {collapseOperator}, {hamiltonian},
                              cudaq::IntermediateResultSave::ExpectationValue);
  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected =
        (N - 1) * std::exp(-decay_rate * (1.0 - std::exp(-t.real())));
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    std::cout << "Result = " << (double)expVals[0] << "; expected "
              << theoryResults[count] << "\n";
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveAPITester, checkBatchedSimple) {
  const cudaq::dimension_map dimensions = {{0, 2}};
  auto hamiltonian = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
  // Initial state: ground state
  std::vector<std::complex<double>> initial_state_zero = {1.0, 0.0};
  std::vector<std::complex<double>> initial_state_one = {0.0, 1.0};

  auto psi0 = cudaq::state::from_data(initial_state_zero);
  auto psi1 = cudaq::state::from_data(initial_state_one);

  // Create a schedule of time steps from 0 to 10 with 101 points
  int numSteps = 101;
  std::vector<double> steps = cudaq::linspace(0.0, 10.0, 101);
  cudaq::schedule schedule(steps);

  // Runge-`Kutta` integrator with a time step of 0.01 and order 4
  cudaq::integrators::runge_kutta integrator(4, 0.01);

  // Run the simulation without collapse operators (ideal evolution)
  auto results = cudaq::evolve(hamiltonian, dimensions, schedule, {psi0, psi1},
                               integrator, {}, {cudaq::spin_op::z(0)},
                               cudaq::IntermediateResultSave::ExpectationValue);
  EXPECT_EQ(results.size(), 2);
  EXPECT_TRUE(results[0].expectation_values.has_value());
  EXPECT_EQ(results[0].expectation_values.value().size(), numSteps);
  EXPECT_TRUE(results[1].expectation_values.has_value());
  EXPECT_EQ(results[1].expectation_values.value().size(), numSteps);

  std::vector<double> theoryResults0;
  std::vector<double> theoryResults1;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    theoryResults0.emplace_back(expected);
    theoryResults1.emplace_back(-expected);
  }

  int count = 0;
  for (auto expVals : results[0].expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults0[count++], 1e-3);
  }

  count = 0;
  for (auto expVals : results[1].expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults1[count++], 1e-3);
  }
}

TEST(EvolveAPITester, checkCavityModelBatchedState) {
  constexpr int N = 10;
  constexpr int numStates = 4;
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto hamiltonian = cudaq::boson_op::number(0);
  const cudaq::dimension_map dimensions{{0, N}};
  std::vector<cudaq::state> initialStates;
  for (int i = 0; i < numStates; ++i) {
    std::vector<std::complex<double>> psi0_(N, 0.0);
    psi0_[N - i - 1] = 1.0;
    initialStates.emplace_back(cudaq::state::from_data(psi0_));
  }

  constexpr double decay_rate = 0.1;

  auto td_function =
      [decay_rate](const std::unordered_map<std::string, std::complex<double>>
                       &parameters) {
        auto entry = parameters.find("t");
        if (entry == parameters.end())
          throw std::runtime_error("Cannot find value of expected parameter");
        const auto t = entry->second.real();
        const auto result = std::sqrt(decay_rate * std::exp(-t));
        return result;
      };

  auto collapseOperator =
      cudaq::scalar_operator(td_function) * cudaq::boson_op::annihilate(0);
  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto results = cudaq::evolve(hamiltonian, dimensions, schedule, initialStates,
                               integrator, {collapseOperator}, {hamiltonian},
                               cudaq::IntermediateResultSave::ExpectationValue);
  EXPECT_EQ(results.size(), numStates);
  for (int i = 0; i < numStates; ++i) {
    auto &result = results[i];
    const auto numPhotons = N - i;
    std::cout << "Checking results for the initial state with " << numPhotons
              << " photons.\n";
    EXPECT_TRUE(result.expectation_values.has_value());
    EXPECT_EQ(result.expectation_values.value().size(), numSteps);
    std::vector<double> theoryResults;
    for (const auto &t : schedule) {
      const double expected =
          (numPhotons - 1) *
          std::exp(-decay_rate * (1.0 - std::exp(-t.real())));
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.expectation_values.value()) {
      EXPECT_EQ(expVals.size(), 1);
      std::cout << "Result = " << (double)expVals[0] << "; expected "
                << theoryResults[count] << "\n";
      EXPECT_NEAR((double)expVals[0], theoryResults[count++], 0.01);
    }
  }
}

TEST(EvolveAPITester, checkNoIntermediateStates) {
  const cudaq::dimension_map dims = {{0, 2}};
  auto ham = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
  constexpr int numSteps = 10;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  cudaq::integrators::runge_kutta integrator(1, 0.001);
  auto result = cudaq::evolve(ham, dims, schedule, initialState, integrator, {},
                              {cudaq::spin_op::z(0)},
                              cudaq::IntermediateResultSave::ExpectationValue);
  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), numSteps);
  EXPECT_TRUE(result.states.has_value());
  // Only the final state is saved, so we expect one state.
  EXPECT_EQ(result.states.value().size(), 1);
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

TEST(EvolveAPITester, checkBatchedNoIntermediateStates) {
  const cudaq::dimension_map dimensions = {{0, 2}};
  auto hamiltonian = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
  // Initial state: ground state
  std::vector<std::complex<double>> initial_state_zero = {1.0, 0.0};
  std::vector<std::complex<double>> initial_state_one = {0.0, 1.0};

  auto psi0 = cudaq::state::from_data(initial_state_zero);
  auto psi1 = cudaq::state::from_data(initial_state_one);

  // Create a schedule of time steps from 0 to 10 with 101 points
  int numSteps = 101;
  std::vector<double> steps = cudaq::linspace(0.0, 10.0, 101);
  cudaq::schedule schedule(steps);

  // Runge-`Kutta` integrator with a time step of 0.01 and order 4
  cudaq::integrators::runge_kutta integrator(4, 0.01);

  // Run the simulation without collapse operators (ideal evolution)
  auto results = cudaq::evolve(hamiltonian, dimensions, schedule, {psi0, psi1},
                               integrator, {}, {cudaq::spin_op::z(0)},
                               cudaq::IntermediateResultSave::ExpectationValue);
  EXPECT_EQ(results.size(), 2);
  EXPECT_TRUE(results[0].expectation_values.has_value());
  EXPECT_EQ(results[0].expectation_values.value().size(), numSteps);
  EXPECT_TRUE(results[1].expectation_values.has_value());
  EXPECT_EQ(results[1].expectation_values.value().size(), numSteps);
  // We expect only the final state to be saved, so we should have one state for
  // each result.
  EXPECT_TRUE(results[0].states.has_value());
  EXPECT_EQ(results[0].states.value().size(), 1);
  EXPECT_TRUE(results[1].states.has_value());
  EXPECT_EQ(results[1].states.value().size(), 1);

  std::vector<double> theoryResults0;
  std::vector<double> theoryResults1;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    theoryResults0.emplace_back(expected);
    theoryResults1.emplace_back(-expected);
  }

  int count = 0;
  for (auto expVals : results[0].expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults0[count++], 1e-3);
  }

  count = 0;
  for (auto expVals : results[1].expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults1[count++], 1e-3);
  }
}

TEST(EvolveAPITester, checkSuperopSimple) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham_ =
      2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
  cudaq::sum_op<cudaq::matrix_handler> ham(ham_);
  constexpr int numSteps = 10;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  cudaq::integrators::runge_kutta integrator(1, 0.001);
  cudaq::super_op sup;
  // Apply `-iH * psi` superop
  sup += cudaq::super_op::left_multiply(std::complex<double>(0.0, -1.0) * ham);

  const auto checkResult = [&](const auto &result) {
    std::vector<double> theoryResults;
    for (const auto &t : schedule) {
      const double expected = std::cos(4.0 * M_PI * 0.1 * t.real());
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.expectation_values.value()) {
      EXPECT_EQ(expVals.size(), 1);
      std::cout << "Result = " << (double)expVals[0] << "; expected "
                << theoryResults[count] << "\n";
      EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
    }
  };

  {
    // Observables as initializer list
    auto result = cudaq::evolve(
        sup, dims, schedule, initialState, integrator, {cudaq::spin_op::z(0)},
        cudaq::IntermediateResultSave::ExpectationValue);
    checkResult(result);
  }

  {
    // Observables as vector
    std::vector<decltype(cudaq::spin_op::z(0))> observables = {
        cudaq::spin_op::z(0)};
    auto result = cudaq::evolve(
        sup, dims, schedule, initialState, integrator, observables,
        cudaq::IntermediateResultSave::ExpectationValue);
    checkResult(result);
  }
}

TEST(EvolveAPITester, checkSuperopMasterEquation) {
  constexpr int N = 10;
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto ham = cudaq::boson_op::number(0);
  const cudaq::dimension_map dimensions{{0, N}};
  std::vector<std::complex<double>> rho0_(N * N, 0.0);
  rho0_.back() = 1.0;
  auto initialState = cudaq::state::from_data(rho0_);
  cudaq::integrators::runge_kutta integrator(4, 0.01);
  cudaq::super_op sup;
  // Apply `-i[H, rho]` superop
  sup += cudaq::super_op::left_multiply(std::complex<double>(0.0, -1.0) * ham);
  sup += cudaq::super_op::right_multiply(std::complex<double>(0.0, 1.0) * ham);

  constexpr double decay_rate = 0.1;

  auto td_function =
      [decay_rate](const std::unordered_map<std::string, std::complex<double>>
                       &parameters) {
        auto entry = parameters.find("t");
        if (entry == parameters.end())
          throw std::runtime_error("Cannot find value of expected parameter");
        const auto t = entry->second.real();
        const auto result = std::sqrt(decay_rate * std::exp(-t));
        return result;
      };

  auto L = cudaq::scalar_operator(td_function) * cudaq::boson_op::annihilate(0);
  auto L_dagger =
      cudaq::scalar_operator(td_function) * cudaq::boson_op::create(0);
  // Lindblad terms
  // L * rho * L_dagger
  sup += cudaq::super_op::left_right_multiply(L, L_dagger);
  // -0.5 * L_dagger * L * rho
  sup += cudaq::super_op::left_multiply(-0.5 * L_dagger * L);
  // -0.5 * rho * L_dagger * L
  sup += cudaq::super_op::right_multiply(-0.5 * L_dagger * L);

  auto result = cudaq::evolve(sup, dimensions, schedule, initialState,
                              integrator, {cudaq::boson_op::number(0)},
                              cudaq::IntermediateResultSave::ExpectationValue);

  EXPECT_TRUE(result.expectation_values.has_value());
  EXPECT_EQ(result.expectation_values.value().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected =
        (N - 1) * std::exp(-decay_rate * (1.0 - std::exp(-t.real())));
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.expectation_values.value()) {
    EXPECT_EQ(expVals.size(), 1);
    std::cout << "Result = " << (double)expVals[0] << "; expected "
              << theoryResults[count] << "\n";
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveAPITester, checkSuperopMasterEquationInputStateEnum) {
  const cudaq::dimension_map dims = {{0, 2}};
  auto ham = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
  cudaq::super_op sup;
  // Apply `-i[H, rho]` superop
  sup += cudaq::super_op::left_multiply(std::complex<double>(0.0, -1.0) * ham);
  sup += cudaq::super_op::right_multiply(std::complex<double>(0.0, 1.0) * ham);
  constexpr int numSteps = 10;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto initialState = cudaq::InitialState::ZERO;
  cudaq::integrators::runge_kutta integrator(1, 0.001);
  auto result = cudaq::evolve(ham, dims, schedule, initialState, integrator, {},
                              {cudaq::spin_op::z(0)},
                              cudaq::IntermediateResultSave::ExpectationValue);
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

TEST(EvolveAPITester, checkCavityModelSuperOpBatchedState) {
  constexpr int N = 10;
  constexpr int numStates = 4;
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  const cudaq::dimension_map dimensions{{0, N}};
  std::vector<cudaq::state> initialStates;
  for (int i = 0; i < numStates; ++i) {
    std::vector<std::complex<double>> psi0_(N, 0.0);
    psi0_[N - i - 1] = 1.0;
    initialStates.emplace_back(cudaq::state::from_data(psi0_));
  }

  auto ham = cudaq::boson_op::number(0);
  cudaq::super_op sup;
  // Apply `-i[H, rho]` superop
  sup += cudaq::super_op::left_multiply(std::complex<double>(0.0, -1.0) * ham);
  sup += cudaq::super_op::right_multiply(std::complex<double>(0.0, 1.0) * ham);

  constexpr double decay_rate = 0.1;

  auto td_function =
      [decay_rate](const std::unordered_map<std::string, std::complex<double>>
                       &parameters) {
        auto entry = parameters.find("t");
        if (entry == parameters.end())
          throw std::runtime_error("Cannot find value of expected parameter");
        const auto t = entry->second.real();
        const auto result = std::sqrt(decay_rate * std::exp(-t));
        return result;
      };

  auto L = cudaq::scalar_operator(td_function) * cudaq::boson_op::annihilate(0);
  auto L_dagger =
      cudaq::scalar_operator(td_function) * cudaq::boson_op::create(0);
  // Lindblad terms
  // L * rho * L_dagger
  sup += cudaq::super_op::left_right_multiply(L, L_dagger);
  // -0.5 * L_dagger * L * rho
  sup += cudaq::super_op::left_multiply(-0.5 * L_dagger * L);
  // -0.5 * rho * L_dagger * L
  sup += cudaq::super_op::right_multiply(-0.5 * L_dagger * L);

  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto results = cudaq::evolve(sup, dimensions, schedule, initialStates,
                               integrator, {cudaq::boson_op::number(0)},
                               cudaq::IntermediateResultSave::ExpectationValue);
  EXPECT_EQ(results.size(), numStates);
  for (int i = 0; i < numStates; ++i) {
    auto &result = results[i];
    const auto numPhotons = N - i;
    std::cout << "Checking results for the initial state with " << numPhotons
              << " photons.\n";
    EXPECT_TRUE(result.expectation_values.has_value());
    EXPECT_EQ(result.expectation_values.value().size(), numSteps);
    std::vector<double> theoryResults;
    for (const auto &t : schedule) {
      const double expected =
          (numPhotons - 1) *
          std::exp(-decay_rate * (1.0 - std::exp(-t.real())));
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.expectation_values.value()) {
      EXPECT_EQ(expVals.size(), 1);
      std::cout << "Result = " << (double)expVals[0] << "; expected "
                << theoryResults[count] << "\n";
      EXPECT_NEAR((double)expVals[0], theoryResults[count++], 0.01);
    }
  }
}

TEST(EvolveAPITester, checkCallbackTensorOp) {
  auto tensorFunction =
      [](const std::vector<int64_t> &dimensions,
         const std::unordered_map<std::string, std::complex<double>>
             &parameters) -> cudaq::complex_matrix {
    if (dimensions.empty())
      throw std::runtime_error("Empty dimensions vector received!");

    cudaq::complex_matrix mat(2, 2);
    mat[{0, 0}] = 0.0;
    mat[{0, 1}] = 2.0 * M_PI * 0.1;
    mat[{1, 0}] = 2.0 * M_PI * 0.1;
    mat[{1, 1}] = 0.0;
    return mat;
  };

  cudaq::matrix_handler::define("CustomX", {2}, tensorFunction);
  auto ham = cudaq::matrix_handler::instantiate("CustomX", {0});
  const cudaq::dimension_map dims = {{0, 2}};
  auto psi0 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

  constexpr int numSteps = 100;
  std::vector<double> steps = cudaq::linspace(0.0, 4.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});

  cudaq::integrators::runge_kutta integrator(4);

  auto result = cudaq::evolve(ham, dims, schedule, psi0, integrator, {},
                              {cudaq::spin_op::z(0)},
                              cudaq::IntermediateResultSave::ExpectationValue);

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
