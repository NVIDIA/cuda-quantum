// /*******************************************************************************
//  * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "CuDensityMatUtils.h"
#include "cudaq/algorithms/integrator.h"
#include "test_Mocks.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

using namespace cudaq;

class RungeKuttaIntegratorTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;
  cudensitymatOperator_t liouvillian_;
  std::unique_ptr<cudaq::integrators::runge_kutta> integrator_;
  std::unique_ptr<CuDensityMatState> state_;

  void SetUp() override {
    // Create library handle
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));

    // Create a mock Liouvillian
    liouvillian_ = mock_liouvillian(handle_);

    // Create initial state
    state_ = std::make_unique<CuDensityMatState>(
        mock_initial_state_data().size(),
        cudaq::dynamics::createArrayGpu(mock_initial_state_data()));
    state_->initialize_cudm(handle_, mock_hilbert_space_dims(),
                            /*batchSize=*/1);
    ASSERT_NE(state_, nullptr);
    ASSERT_TRUE(state_->is_initialized());

    double t0 = 0.0;
    // Initialize the integrator (using substeps = 4, for Runge-Kutta method)
    ASSERT_NO_THROW(integrator_ =
                        std::make_unique<cudaq::integrators::runge_kutta>());
    ASSERT_NE(integrator_, nullptr);
  }

  void TearDown() override {
    // Clean up resources
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(liouvillian_));
    HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_));
  }
};

// Test Initialization
TEST_F(RungeKuttaIntegratorTest, Initialization) {
  ASSERT_NE(integrator_, nullptr);
}

TEST_F(RungeKuttaIntegratorTest, CheckEvolve) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  auto spin_op_x = cudaq::spin_op::x(0);
  cudaq::product_op<cudaq::matrix_handler> ham1 = 2.0 * M_PI * 0.1 * spin_op_x;
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);
  SystemDynamics system(dims, ham);

  for (int integratorOrder : {1, 2, 4}) {
    std::cout << "Test RK order " << integratorOrder << "\n";
    cudaq::integrators::runge_kutta integrator(integratorOrder, 0.001);
    constexpr std::size_t numDataPoints = 10;
    double t = 0.0;
    auto initialState = cudaq::state::from_data(initialStateVec);
    // initialState.dump();
    auto *simState = cudaq::state_helper::getSimulationState(&initialState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    EXPECT_TRUE(castSimState != nullptr);
    castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);
    integrator.setState(initialState, 0.0);
    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, 1.0 * numDataPoints, numDataPoints)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    cudaq::integrator_helper::init_system_dynamics(integrator, system,
                                                   schedule);
    std::vector<std::complex<double>> outputStateVec(2);
    for (std::size_t i = 1; i < numDataPoints; ++i) {
      integrator.integrate(i);
      auto [t, state] = integrator.getState();
      // std::cout << "Time = " << t << "\n";
      // state.dump();
      state.to_host(outputStateVec.data(), outputStateVec.size());
      // Check state vector norm
      EXPECT_NEAR(std::norm(outputStateVec[0]) + std::norm(outputStateVec[1]),
                  1.0, 1e-2);
      const double expValZ =
          std::norm(outputStateVec[0]) - std::norm(outputStateVec[1]);
      // Analytical results
      EXPECT_NEAR(outputStateVec[0].real(), std::cos(2.0 * M_PI * 0.1 * t),
                  1e-2);
    }
  }

  // Add test to test tensor_callback
}

// Test to verify the convergence order of integrators.
// This test uses Richardson extrapolation to estimate the order of accuracy.
// For an integrator of order p, when step size h is halved, error should
// decrease by a factor of 2^p (ratio ~ 2^p).
TEST_F(RungeKuttaIntegratorTest, ConvergenceOrderVerification) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  // Hamiltonian: H = omega * sigma_x, omega = 2*pi*0.1
  const double omega = 2.0 * M_PI * 0.1;
  auto spin_op_x = cudaq::spin_op::x(0);
  cudaq::product_op<cudaq::matrix_handler> ham1 = omega * spin_op_x;
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);
  SystemDynamics system(dims, ham);

  // Test parameters
  const double t_final = 5.0;
  constexpr std::size_t numDataPoints = 51;

  // Helper lambda to run evolution and get final state error
  auto runEvolution = [&](int order, double stepSize) -> double {
    cudaq::integrators::runge_kutta integrator(order, stepSize);
    auto initialState = cudaq::state::from_data(initialStateVec);
    auto *simState = cudaq::state_helper::getSimulationState(&initialState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);
    integrator.setState(initialState, 0.0);

    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, t_final, numDataPoints)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    cudaq::integrator_helper::init_system_dynamics(integrator, system,
                                                   schedule);

    // Integrate to final time
    integrator.integrate(t_final);
    auto [t, state] = integrator.getState();

    std::vector<std::complex<double>> outputStateVec(2);
    state.to_host(outputStateVec.data(), outputStateVec.size());

    // Analytical solution: |<0|psi>|^2 = cos^2(omega * t)
    double analytical = std::cos(omega * t_final) * std::cos(omega * t_final);
    double numerical = std::norm(outputStateVec[0]); // |<0|psi>|^2 = |psi_0|^2

    return std::abs(numerical - analytical);
  };

  // Test each integrator order with two step sizes
  // Using step sizes that are large enough to show meaningful errors
  const double h1 = 0.1;  // Larger step size
  const double h2 = 0.05; // Half of h1

  for (int order : {1, 2, 4}) {
    double error_h1 = runEvolution(order, h1);
    double error_h2 = runEvolution(order, h2);

    // Compute the error ratio when step size is halved
    double ratio = error_h1 / error_h2;

    // Estimate the convergence order: ratio = 2^p => p = log2(ratio)
    double estimated_order = std::log2(ratio);

    std::cout << "Order " << order << " integrator: "
              << "error(h=" << h1 << ")=" << error_h1 << ", error(h=" << h2
              << ")=" << error_h2 << ", ratio=" << ratio
              << ", estimated_order=" << estimated_order << "\n";

    // Verify that the estimated order is close to the expected order
    // Allow some tolerance due to numerical effects
    // Expected: order 1 -> ratio ~2, order 2 -> ratio ~4, order 4 -> ratio ~16
    double expected_ratio = std::pow(2.0, order);
    double min_acceptable_ratio =
        expected_ratio * 0.5; // At least half of expected

    EXPECT_GE(ratio, min_acceptable_ratio)
        << "Order " << order
        << " integrator shows lower than expected convergence rate. "
        << "Expected ratio >= " << min_acceptable_ratio << ", got " << ratio
        << ". Estimated order: " << estimated_order;

    // Also verify the estimated order is reasonable (within 0.5 of expected)
    EXPECT_GE(estimated_order, order - 0.5)
        << "Order " << order
        << " integrator estimated order too low: " << estimated_order
        << " (expected >= " << (order - 0.5) << ")";
  }
}

// Test that higher-order integrators produce more accurate results
// with the same step size
TEST_F(RungeKuttaIntegratorTest, AccuracyComparison) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  const double omega = 2.0 * M_PI * 0.1;
  auto spin_op_x = cudaq::spin_op::x(0);
  cudaq::product_op<cudaq::matrix_handler> ham1 = omega * spin_op_x;
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);
  SystemDynamics system(dims, ham);

  const double t_final = 5.0;
  const double stepSize = 0.1; // Use same step size for all
  constexpr std::size_t numDataPoints = 51;

  // Helper to compute error
  auto computeError = [&](int order) -> double {
    cudaq::integrators::runge_kutta integrator(order, stepSize);
    auto initialState = cudaq::state::from_data(initialStateVec);
    auto *simState = cudaq::state_helper::getSimulationState(&initialState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);
    integrator.setState(initialState, 0.0);

    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, t_final, numDataPoints)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    cudaq::integrator_helper::init_system_dynamics(integrator, system,
                                                   schedule);

    integrator.integrate(t_final);
    auto [t, state] = integrator.getState();

    std::vector<std::complex<double>> outputStateVec(2);
    state.to_host(outputStateVec.data(), outputStateVec.size());

    double analytical = std::cos(omega * t_final) * std::cos(omega * t_final);
    double numerical = std::norm(outputStateVec[0]);

    return std::abs(numerical - analytical);
  };

  double error_order1 = computeError(1);
  double error_order2 = computeError(2);
  double error_order4 = computeError(4);

  std::cout << "Accuracy comparison (step_size=" << stepSize << "):\n";
  std::cout << "  Order 1 error: " << error_order1 << "\n";
  std::cout << "  Order 2 error: " << error_order2 << "\n";
  std::cout << "  Order 4 error: " << error_order4 << "\n";

  // Order 2 should be significantly more accurate than Order 1
  EXPECT_LT(error_order2, error_order1 * 0.1)
      << "Order 2 should be at least 10x more accurate than Order 1";

  // Order 4 should be significantly more accurate than Order 2
  EXPECT_LT(error_order4, error_order2 * 0.01)
      << "Order 4 should be at least 100x more accurate than Order 2";

  // Order 2 error should be reasonable for a 2nd order method
  // With h=0.1 and t=5.0 (50 steps), error should be O(h^2) ~ 0.01 range
  EXPECT_LT(error_order2, 0.01)
      << "Order 2 error too large for a proper 2nd-order method";
}
