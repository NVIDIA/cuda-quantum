/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/algorithms/evolve.h"
#include "cudaq/dynamics_integrators.h"
#include <cudaq.h>
#include <gtest/gtest.h>

TEST(DynamicsAsyncTester, checkSimple) {
  auto &platform = cudaq::get_platform();
  printf("Num QPUs %lu\n", platform.num_qpus());
  auto jobHandle1 = []() {
    const std::map<int, int> dims = {{0, 2}};
    auto ham = 2.0 * M_PI * 0.1 * cudaq::spin_operator::x(0);
    constexpr int numSteps = 10;
    cudaq::Schedule schedule(cudaq::linspace(0.0, 1.0, numSteps));
    auto initialState =
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
    auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
    integrator->order = 1;
    integrator->dt = 0.001;
    auto resultFuture1 = cudaq::evolve_async(
        ham, dims, schedule, initialState, integrator,
        std::vector<cudaq::product_operator<cudaq::spin_operator>>{},
        std::vector<cudaq::product_operator<cudaq::spin_operator>>{
            cudaq::spin_operator::z(0)},
        true, {}, 0);
    std::cout << "Launched evolve job on QPU 0\n";
    return resultFuture1;
  }();

  auto jobHandle2 = []() {
    constexpr int N = 10;
    constexpr int numSteps = 101;
    const auto steps = cudaq::linspace(0, 10, numSteps);
    cudaq::Schedule schedule(steps);
    auto hamiltonian = cudaq::boson_operator::number(0);
    const std::map<int, int> dimensions{{0, N}};
    std::vector<std::complex<double>> psi0_(N, 0.0);
    psi0_.back() = 1.0;
    auto psi0 = cudaq::state::from_data(psi0_);
    constexpr double decay_rate = 0.1;
    auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
    integrator->dt = 0.01;
    auto resultFuture = cudaq::evolve_async(
        hamiltonian, dimensions, schedule, psi0, integrator,
        std::vector<cudaq::product_operator<cudaq::boson_operator>>{
            std::sqrt(decay_rate) * cudaq::boson_operator::annihilate(0)},
        std::vector<cudaq::product_operator<cudaq::boson_operator>>{
            hamiltonian},
        true, {}, 1);
    std::cout << "Launched evolve job on QPU 1\n";
    return resultFuture;
  }();

  std::cout << "Wait for all the evolve jobs complete...\n";
  {
    auto result = jobHandle1.get();
    std::cout << "Checking the results from QPU 0\n";
    constexpr int numSteps = 10;
    EXPECT_NE(result.get_expectation_values().size(), 0);
    EXPECT_EQ(result.get_expectation_values().size(), numSteps);
    std::vector<double> theoryResults;
    for (const auto &t : cudaq::linspace(0.0, 1.0, numSteps)) {
      const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t);
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.get_expectation_values()) {
      EXPECT_EQ(expVals.size(), 1);
      EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
    }
  }
  {
    auto result = jobHandle2.get();
    std::cout << "Checking the results from QPU 1\n";
    constexpr int N = 10;
    constexpr double decay_rate = 0.1;
    constexpr int numSteps = 101;
    const auto steps = cudaq::linspace(0, 10, numSteps);
    EXPECT_NE(result.get_expectation_values().size(), 0);
    EXPECT_EQ(result.get_expectation_values().size(), numSteps);
    std::vector<double> theoryResults;
    for (const auto &t : steps) {
      const double expected = (N - 1) * std::exp(-decay_rate * t);
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.get_expectation_values()) {
      EXPECT_EQ(expVals.size(), 1);
      EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
    }
  }
}

TEST(DynamicsAsyncTester, checkInitializerArgs) {
  auto &platform = cudaq::get_platform();
  printf("Num QPUs %lu\n", platform.num_qpus());
  auto jobHandle1 = []() {
    const std::map<int, int> dims = {{0, 2}};
    auto ham = 2.0 * M_PI * 0.1 * cudaq::spin_operator::x(0);
    constexpr int numSteps = 10;
    cudaq::Schedule schedule(cudaq::linspace(0.0, 1.0, numSteps));
    auto initialState =
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
    auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
    integrator->order = 1;
    integrator->dt = 0.001;
    auto resultFuture1 =
        cudaq::evolve_async(ham, dims, schedule, initialState, integrator, {},
                            {cudaq::spin_operator::z(0)}, true, {}, 0);
    std::cout << "Launched evolve job on QPU 0\n";
    return resultFuture1;
  }();

  auto jobHandle2 = []() {
    constexpr int N = 10;
    constexpr int numSteps = 101;
    const auto steps = cudaq::linspace(0, 10, numSteps);
    cudaq::Schedule schedule(steps);
    auto hamiltonian = cudaq::boson_operator::number(0);
    const std::map<int, int> dimensions{{0, N}};
    std::vector<std::complex<double>> psi0_(N, 0.0);
    psi0_.back() = 1.0;
    auto psi0 = cudaq::state::from_data(psi0_);
    constexpr double decay_rate = 0.1;
    auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
    integrator->dt = 0.01;
    auto resultFuture = cudaq::evolve_async(
        hamiltonian, dimensions, schedule, psi0, integrator,
        {std::sqrt(decay_rate) * cudaq::boson_operator::annihilate(0)},
        {hamiltonian}, true, {}, 1);
    std::cout << "Launched evolve job on QPU 1\n";
    return resultFuture;
  }();

  std::cout << "Wait for all the evolve jobs complete...\n";
  {
    auto result = jobHandle1.get();
    std::cout << "Checking the results from QPU 0\n";
    constexpr int numSteps = 10;
    EXPECT_NE(result.get_expectation_values().size(), 0);
    EXPECT_EQ(result.get_expectation_values().size(), numSteps);
    std::vector<double> theoryResults;
    for (const auto &t : cudaq::linspace(0.0, 1.0, numSteps)) {
      const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t);
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.get_expectation_values()) {
      EXPECT_EQ(expVals.size(), 1);
      EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
    }
  }
  {
    auto result = jobHandle2.get();
    std::cout << "Checking the results from QPU 1\n";
    constexpr int N = 10;
    constexpr double decay_rate = 0.1;
    constexpr int numSteps = 101;
    const auto steps = cudaq::linspace(0, 10, numSteps);
    EXPECT_NE(result.get_expectation_values().size(), 0);
    EXPECT_EQ(result.get_expectation_values().size(), numSteps);
    std::vector<double> theoryResults;
    for (const auto &t : steps) {
      const double expected = (N - 1) * std::exp(-decay_rate * t);
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.get_expectation_values()) {
      EXPECT_EQ(expVals.size(), 1);
      EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
    }
  }
}