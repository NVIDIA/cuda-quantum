/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include <cudaq.h>
#include <gtest/gtest.h>
#include <iostream>

class TestEnvironment : public ::testing::Environment {
protected:
  void SetUp() override { cudaq::mpi::initialize(); }
  void TearDown() override { cudaq::mpi::finalize(); }
};

::testing::Environment *const dynamics_test_env =
    AddGlobalTestEnvironment(new TestEnvironment);

TEST(DynamicsBatchingMpi, checkSaveModes) {
  EXPECT_TRUE(cudaq::mpi::is_initialized());
  std::cout << "Number of ranks = " << cudaq::mpi::num_ranks() << "\n";
  EXPECT_GT(cudaq::mpi::num_ranks(), 1);

  const auto rank = cudaq::mpi::rank();
  const auto num_ranks = cudaq::mpi::num_ranks();

  // Create 4 distinct initial states
  std::vector<cudaq::state> initial_states;
  for (int i = 0; i < 4; i++) {
    const double theta = i * M_PI / 8;
    std::vector<std::complex<double>> state_data = {std::cos(theta),
                                                    std::sin(theta)};
    initial_states.push_back(cudaq::state::from_data(state_data));
  }
  // Simple single-qubit Hamiltonian
  cudaq::dimension_map dimensions{{0, 2}};
  auto hamiltonian = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);

  const auto batch_size = initial_states.size();
  constexpr int num_steps = 11;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, num_steps);
  cudaq::schedule schedule(steps);

  // Test helpers to run tests for different save modes
  const auto runTest = [&](auto saveMode) {
    cudaq::integrators::runge_kutta integrator;

    // Use save all to test split batched state across MPI ranks
    auto evolve_results =
        cudaq::evolve(hamiltonian, dimensions, schedule, initial_states,
                      integrator, {}, {cudaq::spin_op::z(0)}, saveMode);
    return evolve_results;
  };
  {
    // CHECK: cudaq::IntermediateResultSave::All
    auto evolve_results = runTest(cudaq::IntermediateResultSave::All);
    // In distributed mode, each rank gets a subset of results
    const auto expected_local_results = batch_size / num_ranks;
    EXPECT_EQ(evolve_results.size(), expected_local_results);

    // Verify each result
    for (size_t i = 0; i < evolve_results.size(); ++i) {
      const auto &result = evolve_results[i];
      auto final_state = result.states->back();
      // State should be a 2-element vector (single qubit)
      std::vector<std::complex<double>> state_array(2);
      final_state.to_host(state_array.data(), state_array.size());

      // State should be approximately normalized
      const auto norm =
          std::sqrt(std::norm(state_array[0]) + std::norm(state_array[1]));
      EXPECT_LT(std::abs(norm - 1.0), 0.01);

      // Should have 11 intermediate states
      auto &intermediate_states = result.states.value();
      std::cout << "Rank " << rank << ", Result " << i
                << ", Intermediate states count: " << intermediate_states.size()
                << ", Number of expectation values: "
                << result.expectation_values->size() << "\n";
      EXPECT_EQ(intermediate_states.size(), num_steps);
      EXPECT_EQ(result.expectation_values->size(), num_steps);
    }
  }

  {
    auto evolve_results =
        runTest(cudaq::IntermediateResultSave::ExpectationValue);
    // In distributed mode, each rank gets a subset of results
    const auto expected_local_results = batch_size / num_ranks;
    EXPECT_EQ(evolve_results.size(), expected_local_results);
    // Verify each result
    for (size_t i = 0; i < evolve_results.size(); ++i) {
      const auto &result = evolve_results[i];
      auto final_state = result.states->back();
      // State should be a 2-element vector (single qubit)
      std::vector<std::complex<double>> state_array(2);
      final_state.to_host(state_array.data(), state_array.size());

      // State should be approximately normalized
      const auto norm =
          std::sqrt(std::norm(state_array[0]) + std::norm(state_array[1]));
      EXPECT_LT(std::abs(norm - 1.0), 0.01);

      // Should have only final state stored
      auto &intermediate_states = result.states.value();
      std::cout << "Rank " << rank << ", Result " << i
                << ", Intermediate states count: " << intermediate_states.size()
                << ", Number of expectation values: "
                << result.expectation_values->size() << "\n";
      EXPECT_EQ(intermediate_states.size(), 1); // Only final state stored
      EXPECT_EQ(result.expectation_values->size(), num_steps);
    }
  }

  {
    auto evolve_results = runTest(cudaq::IntermediateResultSave::None);
    // In distributed mode, each rank gets a subset of results
    const auto expected_local_results = batch_size / num_ranks;
    EXPECT_EQ(evolve_results.size(), expected_local_results);

    // Verify each result
    for (size_t i = 0; i < evolve_results.size(); ++i) {
      const auto &result = evolve_results[i];
      auto final_state = result.states->back();
      // State should be a 2-element vector (single qubit)
      std::vector<std::complex<double>> state_array(2);
      final_state.to_host(state_array.data(), state_array.size());

      // State should be approximately normalized
      const auto norm =
          std::sqrt(std::norm(state_array[0]) + std::norm(state_array[1]));
      EXPECT_LT(std::abs(norm - 1.0), 0.01);

      // Should have only final state and final expectation values stored
      auto &intermediate_states = result.states.value();
      std::cout << "Rank " << rank << ", Result " << i
                << ", Intermediate states count: " << intermediate_states.size()
                << ", Number of expectation values: "
                << result.expectation_values->size() << "\n";
      EXPECT_EQ(intermediate_states.size(), 1); // Only final state stored
      EXPECT_EQ(result.expectation_values->size(),
                1); // Only final expectation value stored
    }
  }
}
