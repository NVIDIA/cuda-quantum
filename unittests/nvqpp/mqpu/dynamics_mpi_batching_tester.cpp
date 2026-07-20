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

class TestEnvironment : public ::testing::Environment {
protected:
  void SetUp() override { cudaq::mpi::initialize(); }
  void TearDown() override { cudaq::mpi::finalize(); }
};

::testing::Environment *const dynamics_test_env =
    AddGlobalTestEnvironment(new TestEnvironment);

TEST(DynamicsBatchingMpi, checkSaveModes) {
  EXPECT_TRUE(cudaq::mpi::is_initialized());
  EXPECT_GT(cudaq::mpi::num_ranks(), 1);

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
      EXPECT_EQ(intermediate_states.size(), 1); // Only final state stored
      EXPECT_EQ(result.expectation_values->size(),
                1); // Only final expectation value stored
    }
  }
}

TEST(DynamicsBatchingMpi, checkDifferentBatchSizes) {
  EXPECT_TRUE(cudaq::mpi::is_initialized());
  EXPECT_GT(cudaq::mpi::num_ranks(), 1);

  const auto num_ranks = cudaq::mpi::num_ranks();
  const auto rank = cudaq::mpi::rank();

  // Simple single-qubit Hamiltonian
  cudaq::dimension_map dimensions{{0, 2}};
  auto hamiltonian = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);

  constexpr int num_steps = 11;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, num_steps);
  cudaq::schedule schedule(steps);

  for (auto batch_size : {2, 4, 6, 8}) {
    if (batch_size % num_ranks != 0) {
      // Skip batch sizes that cannot be evenly divided across ranks
      continue;
    }
    printf("Rank %d: Testing batch size %d\n", rank, batch_size);
    std::vector<cudaq::state> initial_states;
    for (int i = 0; i < batch_size; i++) {
      const double theta = i * M_PI / (2 * batch_size);
      std::vector<std::complex<double>> state_data = {std::cos(theta),
                                                      std::sin(theta)};
      initial_states.push_back(cudaq::state::from_data(state_data));
    }

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
        EXPECT_EQ(intermediate_states.size(), 1); // Only final state stored
        EXPECT_EQ(result.expectation_values->size(),
                  1); // Only final expectation value stored
      }
    }
  }
}

TEST(DynamicsBatchingMpi, checkWithCollapseOperators) {
  EXPECT_TRUE(cudaq::mpi::is_initialized());
  EXPECT_GT(cudaq::mpi::num_ranks(), 1);

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
  // Decay operator
  const double gamma = 0.1;
  auto collapse_op = std::sqrt(gamma) * cudaq::spin_op::minus(0);

  const auto batch_size = initial_states.size();
  constexpr int num_steps = 11;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, num_steps);
  cudaq::schedule schedule(steps);

  // Test helpers to run tests for different save modes
  const auto runTest = [&](auto saveMode) {
    cudaq::integrators::runge_kutta integrator;

    // Use save all to test split batched state across MPI ranks
    auto evolve_results = cudaq::evolve(
        hamiltonian, dimensions, schedule, initial_states, integrator,
        {collapse_op}, {cudaq::spin_op::z(0)}, saveMode);
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
      // State should be a density matrix with 4 elements
      std::vector<std::complex<double>> density_matrix(4);
      final_state.to_host(density_matrix.data(), density_matrix.size());

      // State should be approximately normalized
      const auto norm = density_matrix[0] + density_matrix[3]; // Diagonal sum
      EXPECT_LT(std::abs(norm - 1.0), 0.01);

      // Should have 11 intermediate states
      auto &intermediate_states = result.states.value();
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
      // State should be density matrix with 4 elements
      std::vector<std::complex<double>> density_matrix(4);
      final_state.to_host(density_matrix.data(), density_matrix.size());

      // State should be approximately normalized
      const auto norm = density_matrix[0] + density_matrix[3];
      EXPECT_LT(std::abs(norm - 1.0), 0.01);

      // Should have only final state stored
      auto &intermediate_states = result.states.value();
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
      // State should be a density matrix with 4 elements
      std::vector<std::complex<double>> density_matrix(4);
      final_state.to_host(density_matrix.data(), density_matrix.size());

      // State should be approximately normalized
      const auto norm = density_matrix[0] + density_matrix[3];
      EXPECT_LT(std::abs(norm - 1.0), 0.01);

      // Should have only final state and final expectation values stored
      auto &intermediate_states = result.states.value();
      EXPECT_EQ(intermediate_states.size(), 1); // Only final state stored
      EXPECT_EQ(result.expectation_values->size(),
                1); // Only final expectation value stored
    }
  }
}

TEST(DynamicsBatchingMpi, checkTwoQubits) {
  EXPECT_TRUE(cudaq::mpi::is_initialized());
  EXPECT_GT(cudaq::mpi::num_ranks(), 1);

  const auto num_ranks = cudaq::mpi::num_ranks();

  // Create 4 distinct initial states
  std::vector<cudaq::state> initial_states;
  for (int i = 0; i < 4; i++) {
    const double theta0 = i * M_PI / 8;
    const double theta1 = (i + 1) * M_PI / 8;
    std::vector<std::complex<double>> state_data = {
        std::cos(theta0) * std::cos(theta1),
        std::cos(theta0) * std::sin(theta1),
        std::sin(theta0) * std::cos(theta1),
        std::sin(theta0) * std::sin(theta1)};
    initial_states.push_back(cudaq::state::from_data(state_data));
  }
  // Two qubit Hamiltonian
  cudaq::dimension_map dimensions{{0, 2}, {1, 2}};
  auto hamiltonian = 2.0 * M_PI * 0.1 *
                     (cudaq::spin_op::x(0) + cudaq::spin_op::x(1) +
                      0.5 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1));

  const auto batch_size = initial_states.size();
  constexpr int num_steps = 11;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, num_steps);
  cudaq::schedule schedule(steps);

  // Test helpers to run tests for different save modes
  const auto runTest = [&](auto saveMode) {
    cudaq::integrators::runge_kutta integrator;

    // Use save all to test split batched state across MPI ranks
    auto evolve_results = cudaq::evolve(
        hamiltonian, dimensions, schedule, initial_states, integrator, {},
        {cudaq::spin_op::z(0), cudaq::spin_op::z(0)}, saveMode);
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
      // State should be a 4-element vector (two qubits)
      std::vector<std::complex<double>> state_array(4);
      final_state.to_host(state_array.data(), state_array.size());

      // State should be approximately normalized
      const auto norm =
          std::sqrt(std::norm(state_array[0]) + std::norm(state_array[1]) +
                    std::norm(state_array[2]) + std::norm(state_array[3]));
      EXPECT_LT(std::abs(norm - 1.0), 0.01);

      // Should have 11 intermediate states
      auto &intermediate_states = result.states.value();
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
      // State should be a 4-element vector (two qubits)
      std::vector<std::complex<double>> state_array(4);
      final_state.to_host(state_array.data(), state_array.size());

      // State should be approximately normalized
      const auto norm =
          std::sqrt(std::norm(state_array[0]) + std::norm(state_array[1]) +
                    std::norm(state_array[2]) + std::norm(state_array[3]));
      EXPECT_LT(std::abs(norm - 1.0), 0.01);

      // Should have only final state stored
      auto &intermediate_states = result.states.value();
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
      // State should be a 4-element vector (two qubits)
      std::vector<std::complex<double>> state_array(4);
      final_state.to_host(state_array.data(), state_array.size());

      // State should be approximately normalized
      const auto norm =
          std::sqrt(std::norm(state_array[0]) + std::norm(state_array[1]) +
                    std::norm(state_array[2]) + std::norm(state_array[3]));
      EXPECT_LT(std::abs(norm - 1.0), 0.01);

      // Should have only final state and final expectation values stored
      auto &intermediate_states = result.states.value();
      EXPECT_EQ(intermediate_states.size(), 1); // Only final state stored
      EXPECT_EQ(result.expectation_values->size(),
                1); // Only final expectation value stored
    }
  }
}

TEST(DynamicsBatchingMpi, checkInvalidBatchSize) {
  EXPECT_TRUE(cudaq::mpi::is_initialized());
  EXPECT_GT(cudaq::mpi::num_ranks(), 1);

  const auto num_ranks = cudaq::mpi::num_ranks();

  // Create (num_ranks + 1) distinct initial states: this is not divisible by
  // num_ranks, so should trigger an error when trying to run with batching.
  std::vector<cudaq::state> initial_states;
  for (int i = 0; i < (num_ranks + 1); i++) {
    const double theta = i * M_PI / (num_ranks + 1);
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

  // Expect a runtime error due to invalid batch size
  EXPECT_THROW(runTest(cudaq::IntermediateResultSave::All), std::runtime_error);
  EXPECT_THROW(runTest(cudaq::IntermediateResultSave::ExpectationValue),
               std::runtime_error);
  EXPECT_THROW(runTest(cudaq::IntermediateResultSave::None),
               std::runtime_error);
}
