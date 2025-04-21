/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq.h"
#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include <gtest/gtest.h>

TEST(DynamicsAsyncTester, checkSimple) {
  auto &platform = cudaq::get_platform();
  printf("Num QPUs %lu\n", platform.num_qpus());
  auto jobHandle1 = []() {
    const cudaq::dimension_map dims = {{0, 2}};
    auto ham = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
    constexpr int numSteps = 10;
    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, 1.0, numSteps)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    auto initialState =
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
    cudaq::integrators::runge_kutta integrator(/*order=*/1,
                                               /*max_step_size*/ 0.001);
    auto resultFuture1 = cudaq::evolve_async(
        ham, dims, schedule, initialState, integrator,
        std::vector<cudaq::spin_op_term>{},
        std::vector<cudaq::spin_op_term>{cudaq::spin_op::z(0)}, true, {}, 0);
    std::cout << "Launched evolve job on QPU 0\n";
    return resultFuture1;
  }();

  auto jobHandle2 = []() {
    constexpr int N = 10;
    constexpr int numSteps = 101;
    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0, 10, numSteps)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    auto hamiltonian = cudaq::boson_op::number(0);
    const cudaq::dimension_map dimensions{{0, N}};
    std::vector<std::complex<double>> psi0_(N, 0.0);
    psi0_.back() = 1.0;
    auto psi0 = cudaq::state::from_data(psi0_);
    constexpr double decay_rate = 0.1;
    cudaq::integrators::runge_kutta integrator(/*order=*/4,
                                               /*max_step_size*/ 0.01);
    auto resultFuture = cudaq::evolve_async(
        hamiltonian, dimensions, schedule, psi0, integrator,
        std::vector<cudaq::boson_op_term>{std::sqrt(decay_rate) *
                                          cudaq::boson_op::annihilate(0)},
        std::vector<cudaq::boson_op_term>{hamiltonian}, true, {}, 1);
    std::cout << "Launched evolve job on QPU 1\n";
    return resultFuture;
  }();

  std::cout << "Wait for all the evolve jobs complete...\n";
  {
    auto result = jobHandle1.get();
    std::cout << "Checking the results from QPU 0\n";
    constexpr int numSteps = 10;
    EXPECT_TRUE(result.expectation_values.has_value());
    EXPECT_EQ(result.expectation_values.value().size(), numSteps);
    std::vector<double> theoryResults;
    for (const auto &t : cudaq::linspace(0.0, 1.0, numSteps)) {
      const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t);
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.expectation_values.value()) {
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
    EXPECT_TRUE(result.expectation_values.has_value());
    EXPECT_EQ(result.expectation_values.value().size(), numSteps);
    std::vector<double> theoryResults;
    for (const auto &t : steps) {
      const double expected = (N - 1) * std::exp(-decay_rate * t);
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.expectation_values.value()) {
      EXPECT_EQ(expVals.size(), 1);
      EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
    }
  }
}

TEST(DynamicsAsyncTester, checkInitializerArgs) {
  auto &platform = cudaq::get_platform();
  printf("Num QPUs %lu\n", platform.num_qpus());
  auto jobHandle1 = []() {
    const cudaq::dimension_map dims = {{0, 2}};
    auto ham = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
    constexpr int numSteps = 10;
    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, 1.0, numSteps)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });

    auto initialState =
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
    cudaq::integrators::runge_kutta integrator(1, 0.001);
    auto resultFuture1 =
        cudaq::evolve_async(ham, dims, schedule, initialState, integrator, {},
                            {cudaq::spin_op::z(0)}, true, {}, 0);
    std::cout << "Launched evolve job on QPU 0\n";
    return resultFuture1;
  }();

  auto jobHandle2 = []() {
    constexpr int N = 10;
    constexpr int numSteps = 101;
    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0, 10, numSteps)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    auto hamiltonian = cudaq::boson_op::number(0);
    const cudaq::dimension_map dimensions{{0, N}};
    std::vector<std::complex<double>> psi0_(N, 0.0);
    psi0_.back() = 1.0;
    auto psi0 = cudaq::state::from_data(psi0_);
    constexpr double decay_rate = 0.1;
    cudaq::integrators::runge_kutta integrator(4, 0.01);
    auto resultFuture = cudaq::evolve_async(
        hamiltonian, dimensions, schedule, psi0, integrator,
        {std::sqrt(decay_rate) * cudaq::boson_op::annihilate(0)}, {hamiltonian},
        true, {}, 1);
    std::cout << "Launched evolve job on QPU 1\n";
    return resultFuture;
  }();

  std::cout << "Wait for all the evolve jobs complete...\n";
  {
    auto result = jobHandle1.get();
    std::cout << "Checking the results from QPU 0\n";
    constexpr int numSteps = 10;
    EXPECT_TRUE(result.expectation_values.has_value());
    EXPECT_EQ(result.expectation_values.value().size(), numSteps);
    std::vector<double> theoryResults;
    for (const auto &t : cudaq::linspace(0.0, 1.0, numSteps)) {
      const double expected = std::cos(2 * 2.0 * M_PI * 0.1 * t);
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.expectation_values.value()) {
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
    EXPECT_TRUE(result.expectation_values.has_value());
    EXPECT_EQ(result.expectation_values.value().size(), numSteps);
    std::vector<double> theoryResults;
    for (const auto &t : steps) {
      const double expected = (N - 1) * std::exp(-decay_rate * t);
      theoryResults.emplace_back(expected);
    }

    int count = 0;
    for (auto expVals : result.expectation_values.value()) {
      EXPECT_EQ(expVals.size(), 1);
      EXPECT_NEAR((double)expVals[0], theoryResults[count++], 1e-3);
    }
  }
}
