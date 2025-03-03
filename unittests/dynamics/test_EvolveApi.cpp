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
#include <complex>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

TEST(EvolveAPITester, checkSimple) {
  const std::map<int, int> dims = {{0, 2}};
  auto ham = 2.0 * M_PI * 0.1 * cudaq::spin_operator::x(0);
  constexpr int numSteps = 10;
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 1.0, numSteps)) {
    steps.emplace_back(t, 0.0);
  }
  cudaq::Schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &val) { return val; });
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
  integrator->order = 1;
  integrator->dt = 0.001;
  auto result = cudaq::evolve(ham, dims, schedule, initialState, integrator, {},
                              {cudaq::spin_operator::z(0)}, true);
  EXPECT_TRUE(result.get_expectation_values().has_value());
  EXPECT_EQ(result.get_expectation_values().value().size(), numSteps);

  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t.real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.get_expectation_values().value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveAPITester, checkCavityModel) {
  constexpr int N = 10;
  constexpr int numSteps = 101;
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 1.0, numSteps)) {
    steps.emplace_back(t, 0.0);
  }
  cudaq::Schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &val) { return val; });
  auto hamiltonian = cudaq::boson_operator::number(0);
  const std::map<int, int> dimensions{{0, N}};
  std::vector<std::complex<double>> psi0_(N, 0.0);
  psi0_.back() = 1.0;
  auto psi0 = cudaq::state::from_data(psi0_);
  constexpr double decay_rate = 0.1;
  auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
  integrator->dt = 0.01;
  auto result = cudaq::evolve(
      hamiltonian, dimensions, schedule, psi0, integrator,
      {std::sqrt(decay_rate) * cudaq::boson_operator::annihilate(0)},
      {hamiltonian}, true);
  EXPECT_TRUE(result.get_expectation_values().has_value());
  EXPECT_EQ(result.get_expectation_values().value().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected = (N - 1) * std::exp(-decay_rate * t.real());
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.get_expectation_values().value()) {
    EXPECT_EQ(expVals.size(), 1);
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}

TEST(EvolveAPITester, checkTimeDependent) {
  constexpr int N = 10;
  constexpr int numSteps = 101;
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 1.0, numSteps)) {
    steps.emplace_back(t, 0.0);
  }
  cudaq::Schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &val) { return val; });
  auto hamiltonian = cudaq::boson_operator::number(0);
  const std::map<int, int> dimensions{{0, N}};
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

  auto collapseOperator = cudaq::scalar_operator(td_function) *
                          cudaq::boson_operator::annihilate(0);
  auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
  integrator->dt = 0.01;
  auto result =
      cudaq::evolve(hamiltonian, dimensions, schedule, psi0, integrator,
                    {collapseOperator}, {hamiltonian}, true);
  EXPECT_TRUE(result.get_expectation_values().has_value());
  EXPECT_EQ(result.get_expectation_values().value().size(), numSteps);
  std::vector<double> theoryResults;
  for (const auto &t : schedule) {
    const double expected =
        (N - 1) * std::exp(-decay_rate * (1.0 - std::exp(-t.real())));
    theoryResults.emplace_back(expected);
  }

  int count = 0;
  for (auto expVals : result.get_expectation_values().value()) {
    EXPECT_EQ(expVals.size(), 1);
    std::cout << "Result = " << (double)expVals[0] << "; expected "
              << theoryResults[count] << "\n";
    EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
  }
}
