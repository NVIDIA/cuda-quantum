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

TEST(BatchedEvolveAPITester, checkHamiltonianInitializerList) {

  const cudaq::dimension_map dims = {{0, 2}};
  auto ham1 = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
  auto ham2 = 2.0 * M_PI * 0.2 * cudaq::spin_op::x(0);

  constexpr int numSteps = 10;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});
  auto initialState1 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  auto initialState2 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

  cudaq::integrators::runge_kutta integrator(4, 0.01);

  const auto checkResults = [&](const auto &results) {
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
  };

  {
    // Observables as initializer list
    auto results = cudaq::evolve(
        {ham1, ham2}, dims, schedule, {initialState1, initialState2},
        integrator, {}, {cudaq::spin_op::z(0)},
        cudaq::IntermediateResultSave::ExpectationValue);
    checkResults(results);
  }
  {
    // Observables as vector
    std::vector<decltype(cudaq::spin_op::z(0))> observables{
        cudaq::spin_op::z(0)};
    auto results = cudaq::evolve(
        {ham1, ham2}, dims, schedule, {initialState1, initialState2},
        integrator, {}, observables,
        cudaq::IntermediateResultSave::ExpectationValue);
    checkResults(results);
  }
}

TEST(BatchedEvolveAPITester, checkHamiltonianVectorType) {
  const cudaq::dimension_map dims = {{0, 2}};
  auto ham1 = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
  auto ham2 = 2.0 * M_PI * 0.2 * cudaq::spin_op::x(0);
  std::vector<cudaq::product_op<cudaq::matrix_handler>> hamiltonians = {ham1,
                                                                        ham2};
  constexpr int numSteps = 10;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, numSteps);
  cudaq::schedule schedule(steps, {"t"});
  auto initialState1 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  auto initialState2 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

  cudaq::integrators::runge_kutta integrator(4, 0.01);

  const auto checkResults = [&](const auto &results) {
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
  };

  {
    // Observables as initializer list
    auto results = cudaq::evolve(
        hamiltonians, dims, schedule, {initialState1, initialState2},
        integrator, {}, {cudaq::spin_op::z(0)},
        cudaq::IntermediateResultSave::ExpectationValue);
    checkResults(results);
  }
  {
    // Observables as vector
    std::vector<decltype(cudaq::spin_op::z(0))> observables{
        cudaq::spin_op::z(0)};
    auto results = cudaq::evolve(
        hamiltonians, dims, schedule, {initialState1, initialState2},
        integrator, {}, observables,
        cudaq::IntermediateResultSave::ExpectationValue);
    checkResults(results);
  }
}

TEST(BatchedEvolveAPITester, checkFullMasterEquation) {
  // Batching the decay rates
  std::vector<double> decayRates = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4};
  const auto batchSize = decayRates.size();
  constexpr int N = 10;
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto hamiltonian = cudaq::boson_op::number(0);
  const cudaq::dimension_map dimensions{{0, N}};
  std::vector<std::complex<double>> psi0_(N, 0.0);
  psi0_.back() = 1.0;
  auto psi0 = cudaq::state::from_data(psi0_);
  std::vector<cudaq::product_op<cudaq::matrix_handler>> batchedHams(
      batchSize,
      cudaq::boson_op::number(0)); // Hamiltonian is the same for all batches
  std::vector<cudaq::state> initialStates(
      batchSize, psi0); // Initial state is the same for all batches

  std::vector<std::vector<cudaq::product_op<cudaq::boson_handler>>>
      batchedCollapsedOps;
  for (const auto &decayRate : decayRates) {
    batchedCollapsedOps.emplace_back(
        std::vector<cudaq::product_op<cudaq::boson_handler>>{
            std::sqrt(decayRate) * cudaq::boson_op::annihilate(0)});
  }

  cudaq::integrators::runge_kutta integrator(4, 0.01);
  auto results = cudaq::evolve(batchedHams, dimensions, schedule, initialStates,
                               integrator, batchedCollapsedOps, {hamiltonian},
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

TEST(BatchedEvolveAPITester, checkBatchedSuperOps) {
  const cudaq::dimension_map dims = {{0, 2}};
  const std::vector<double> resonanceFreqs = {0.05, 0.1, 0.15, 0.2,
                                              0.25, 0.3, 0.35, 0.4};
  std::vector<cudaq::super_op> sups;
  std::vector<cudaq::state> initialStates;
  for (const auto &resonanceFreq : resonanceFreqs) {
    // Apply `-iH * psi` superop
    sups.emplace_back(cudaq::super_op::left_multiply(
        std::complex<double>(0.0, -1.0) * 2.0 * M_PI * resonanceFreq *
        cudaq::spin_op::x(0)));
    initialStates.emplace_back(
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0}));
  }

  cudaq::integrators::runge_kutta integrator(4);
  constexpr int numSteps = 10;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});

  const auto checkResults = [&](const auto &results) {
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
  };

  {
    // Observables as initializer list
    auto results = cudaq::evolve(
        sups, dims, schedule, initialStates, integrator, {cudaq::spin_op::z(0)},
        cudaq::IntermediateResultSave::ExpectationValue);
    checkResults(results);
  }
  {
    // Observables as vector
    std::vector<decltype(cudaq::spin_op::z(0))> observables{
        cudaq::spin_op::z(0)};
    auto results = cudaq::evolve(
        sups, dims, schedule, initialStates, integrator, observables,
        cudaq::IntermediateResultSave::ExpectationValue);
    checkResults(results);
  }
}

TEST(BatchedEvolveAPITester, checkBatchSizeMasterEquation) {
  std::vector<double> decayRates = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4};
  const auto batchSize = decayRates.size();
  constexpr int N = 10;
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  auto hamiltonian = cudaq::boson_op::number(0);
  const cudaq::dimension_map dimensions{{0, N}};
  std::vector<std::complex<double>> psi0_(N, 0.0);
  psi0_.back() = 1.0;
  auto psi0 = cudaq::state::from_data(psi0_);
  std::vector<cudaq::product_op<cudaq::matrix_handler>> batchedHams(
      batchSize,
      cudaq::boson_op::number(0)); // Hamiltonian is the same for all batches
  std::vector<cudaq::state> initialStates(
      batchSize, psi0); // Initial state is the same for all batches

  std::vector<std::vector<cudaq::product_op<cudaq::boson_handler>>>
      batchedCollapsedOps;
  for (const auto &decayRate : decayRates) {
    batchedCollapsedOps.emplace_back(
        std::vector<cudaq::product_op<cudaq::boson_handler>>{
            std::sqrt(decayRate) * cudaq::boson_op::annihilate(0)});
  }

  cudaq::integrators::runge_kutta integrator(4, 0.01);

  {
    // Check invalid batch size
    // Zero
    EXPECT_THROW(cudaq::evolve(batchedHams, dimensions, schedule, initialStates,
                               integrator, batchedCollapsedOps, {hamiltonian},
                               cudaq::IntermediateResultSave::ExpectationValue,
                               /*batchSize*/ 0),
                 std::invalid_argument);
  }

  const auto checkResults = [&](const auto &results) {
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
  };

  // Valid batch size
  {
    auto results =
        cudaq::evolve(batchedHams, dimensions, schedule, initialStates,
                      integrator, batchedCollapsedOps, {hamiltonian},
                      cudaq::IntermediateResultSave::ExpectationValue,
                      /*batchSize*/ 4);
    checkResults(results);
  }
  {
    auto results =
        cudaq::evolve(batchedHams, dimensions, schedule, initialStates,
                      integrator, batchedCollapsedOps, {hamiltonian},
                      cudaq::IntermediateResultSave::ExpectationValue,
                      /*batchSize*/ 1);
    checkResults(results);
  }
  {
    // More than number of work items
    auto results =
        cudaq::evolve(batchedHams, dimensions, schedule, initialStates,
                      integrator, batchedCollapsedOps, {hamiltonian},
                      cudaq::IntermediateResultSave::ExpectationValue,
                      /*batchSize*/ 10);
    checkResults(results);
  }
}

TEST(BatchedEvolveAPITester, checkBatchSizeParamSuperOp) {
  const cudaq::dimension_map dims = {{0, 2}};
  const std::vector<double> resonanceFreqs = {0.05, 0.1, 0.15, 0.2,
                                              0.25, 0.3, 0.35, 0.4};
  std::vector<cudaq::super_op> sups;
  std::vector<cudaq::state> initialStates;
  for (const auto &resonanceFreq : resonanceFreqs) {
    // Apply `-iH * psi` superop
    sups.emplace_back(cudaq::super_op::left_multiply(
        std::complex<double>(0.0, -1.0) * 2.0 * M_PI * resonanceFreq *
        cudaq::spin_op::x(0)));
    initialStates.emplace_back(
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0}));
  }

  cudaq::integrators::runge_kutta integrator(4);
  constexpr int numSteps = 10;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});

  {
    // Check invalid batch size
    // Zero
    EXPECT_THROW(cudaq::evolve(sups, dims, schedule, initialStates, integrator,
                               {cudaq::spin_op::z(0)},
                               cudaq::IntermediateResultSave::ExpectationValue,
                               /*batchSize*/ 0),
                 std::invalid_argument);
  }

  const auto checkResults = [&](const auto &results) {
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
  };
  // Valid batch size
  {
    auto results = cudaq::evolve(
        sups, dims, schedule, initialStates, integrator, {cudaq::spin_op::z(0)},
        cudaq::IntermediateResultSave::ExpectationValue, /*batchSize*/ 4);
    checkResults(results);
  }
  {
    auto results = cudaq::evolve(
        sups, dims, schedule, initialStates, integrator, {cudaq::spin_op::z(0)},
        cudaq::IntermediateResultSave::ExpectationValue, /*batchSize*/ 1);
    checkResults(results);
  }
  {
    // More than number of work items
    auto results = cudaq::evolve(
        sups, dims, schedule, initialStates, integrator, {cudaq::spin_op::z(0)},
        cudaq::IntermediateResultSave::ExpectationValue, /*batchSize*/ 10);
    checkResults(results);
  }
}

TEST(BatchedEvolveAPITester, checkBatchSizeCheckMasterEquation) {
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  // These 2 hamiltonian operators cannot be batched.
  auto ham1 = cudaq::boson_op::number(0) + cudaq::boson_op::number(1);
  auto ham2 = cudaq::boson_op::number(0) +
              cudaq::boson_op::number(0) * cudaq::boson_op::number(1);
  const cudaq::dimension_map dimensions{{0, 2}, {1, 2}};
  std::vector<std::complex<double>> psi0_(4, 0.0);
  psi0_[0] = 1.0;
  auto psi0 = cudaq::state::from_data(psi0_);
  cudaq::integrators::runge_kutta integrator(4, 0.01);

  {
    // Set the batch size of 2, which is invalid since we cannot batch these 2
    // Hamiltonians
    EXPECT_THROW(
        cudaq::evolve({ham1, ham2}, dimensions, schedule, {psi0, psi0},
                      integrator, {},
                      {cudaq::boson_op::number(0), cudaq::boson_op::number(1)},
                      cudaq::IntermediateResultSave::ExpectationValue,
                      /*batchSize*/ 2),
        std::runtime_error);

    EXPECT_NO_THROW(cudaq::evolve(
        {ham1, ham2}, dimensions, schedule, {psi0, psi0}, integrator, {},
        {cudaq::boson_op::number(0), cudaq::boson_op::number(1)},
        cudaq::IntermediateResultSave::ExpectationValue));
    EXPECT_NO_THROW(cudaq::evolve(
        {ham1, ham2}, dimensions, schedule, {psi0, psi0}, integrator, {},
        {cudaq::boson_op::number(0), cudaq::boson_op::number(1)},
        cudaq::IntermediateResultSave::ExpectationValue,
        /*batchSize*/ 1));
  }
}

TEST(BatchedEvolveAPITester, checkBatchSizeCheckSuperOperator) {
  constexpr int numSteps = 101;
  cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
  // These 2 hamiltonian operators cannot be batched.
  auto ham1 = cudaq::boson_op::number(0) + cudaq::boson_op::number(1);
  auto ham2 = cudaq::boson_op::number(0) +
              cudaq::boson_op::number(0) * cudaq::boson_op::number(1);

  const std::vector<cudaq::super_op> sups{
      cudaq::super_op::left_multiply(std::complex<double>(0.0, -1.0) * ham1),
      cudaq::super_op::left_multiply(std::complex<double>(0.0, -1.0) * ham2)};
  const cudaq::dimension_map dimensions{{0, 2}, {1, 2}};
  std::vector<std::complex<double>> psi0_(4, 0.0);
  psi0_[0] = 1.0;
  auto psi0 = cudaq::state::from_data(psi0_);
  cudaq::integrators::runge_kutta integrator(4, 0.01);

  {
    // Set the batch size of 2, which is invalid since we cannot batch these 2
    // super-operators
    EXPECT_THROW(
        cudaq::evolve(sups, dimensions, schedule, {psi0, psi0}, integrator,
                      {cudaq::boson_op::number(0), cudaq::boson_op::number(1)},
                      cudaq::IntermediateResultSave::ExpectationValue,
                      /*batchSize*/ 2),
        std::runtime_error);

    EXPECT_NO_THROW(
        cudaq::evolve(sups, dimensions, schedule, {psi0, psi0}, integrator,
                      {cudaq::boson_op::number(0), cudaq::boson_op::number(1)},
                      cudaq::IntermediateResultSave::ExpectationValue));
    EXPECT_NO_THROW(
        cudaq::evolve(sups, dimensions, schedule, {psi0, psi0}, integrator,
                      {cudaq::boson_op::number(0), cudaq::boson_op::number(1)},
                      cudaq::IntermediateResultSave::ExpectationValue,
                      /*batchSize*/ 1));
  }
}
