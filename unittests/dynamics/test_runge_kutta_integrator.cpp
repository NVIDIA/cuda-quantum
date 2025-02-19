// /*******************************************************************************
//  * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "CuDensityMatState.h"
#include "cudaq/dynamics_integrators.h"
#include "cudm_helpers.h"
#include "cudm_time_stepper.h"
#include "test_mocks.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

using namespace cudaq;

class RungeKuttaIntegratorTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;
  cudensitymatOperator_t liouvillian_;
  std::unique_ptr<runge_kutta> integrator_;
  std::unique_ptr<CuDensityMatState> state_;

  void SetUp() override {
    // Create library handle
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));

    // Create a mock Liouvillian
    liouvillian_ = mock_liouvillian(handle_);

    // Create initial state
    state_ = std::make_unique<CuDensityMatState>(
        handle_, mock_initial_state_data(), mock_hilbert_space_dims());
    ASSERT_NE(state_, nullptr);
    ASSERT_TRUE(state_->is_initialized());

    double t0 = 0.0;
    // Initialize the integrator (using substeps = 4, for Runge-Kutta method)
    ASSERT_NO_THROW(integrator_ = std::make_unique<runge_kutta>());
    ASSERT_NE(integrator_, nullptr);
    integrator_->order = 4;
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

// Integration with Euler Method (substeps = 1)
TEST_F(RungeKuttaIntegratorTest, EulerIntegration) {
  // auto integrator = std::make_unique<runge_kutta>(
  //     CuDensityMatState(handle_, mock_initial_state_data(),
  //     mock_hilbert_space_dims()), 0.0, time_stepper_, 1);
  // integrator->set_option("dt", 0.1);
  // EXPECT_NO_THROW(integrator->integrate(1.0));
}

// Integration with Midpoint Rule (substeps = 2)
TEST_F(RungeKuttaIntegratorTest, MidpointIntegration) {
  // auto midpointIntegrator = std::make_unique<runge_kutta>(
  //     CuDensityMatState(handle_, mock_initial_state_data(),
  //     mock_hilbert_space_dims()), 0.0, time_stepper_, 2);
  // midpointIntegrator->set_option("dt", 0.1);
  // EXPECT_NO_THROW(midpointIntegrator->integrate(1.0));
}

// Integration with Runge-Kutta 4 (substeps = 4, which is the default value)
TEST_F(RungeKuttaIntegratorTest, RungeKutta4Integration) {
  // integrator_->set_option("dt", 0.1);
  // EXPECT_NO_THROW(integrator_->integrate(1.0));
}

// Basic Integration Test
TEST_F(RungeKuttaIntegratorTest, BasicIntegration) {
  // auto [t_before, state_before] = integrator_->get_state();
  // integrator_->set_option("dt", 0.1);

  // EXPECT_NO_THROW(integrator_->integrate(1.0));

  // auto [t_after, state_after] = integrator_->get_state();
  // EXPECT_GT(t_after, t_before);
}

// Multiple Integration Steps
TEST_F(RungeKuttaIntegratorTest, MultipleIntegrationSteps) {
  // integrator_->set_option("dt", 0.1);
  // integrator_->integrate(0.5);
  // auto [t_mid, _] = integrator_->get_state();

  // EXPECT_EQ(t_mid, 0.5);

  // integrator_->integrate(1.0);
  // auto [t_final, __] = integrator_->get_state();

  // EXPECT_EQ(t_final, 1.0);
}

// Missing Time Step (dt)
TEST_F(RungeKuttaIntegratorTest, MissingTimeStepOption) {
  // auto integrator_missing_dt = std::make_unique<runge_kutta>(
  //     CuDensityMatState(handle_, mock_initial_state_data(),
  //     mock_hilbert_space_dims()), 0.0, time_stepper_, 2);

  // EXPECT_THROW(integrator_missing_dt->integrate(1.0), std::invalid_argument);
}

// Invalid Time Step (dt <= 0)
TEST_F(RungeKuttaIntegratorTest, InvalidTimeStepSize) {
  // integrator_->set_option("dt", -0.1);
  // EXPECT_THROW(integrator_->integrate(1.0), std::invalid_argument);
}

// Zero Integration Time
TEST_F(RungeKuttaIntegratorTest, ZeroIntegrationTime) {
  // auto [t_before, state_before] = integrator_->get_state();
  // integrator_->set_option("dt", 0.1);

  // EXPECT_NO_THROW(integrator_->integrate(0.0));

  // auto [t_after, state_after] = integrator_->get_state();
  // EXPECT_EQ(t_before, t_after);
}

// Large Time Step
TEST_F(RungeKuttaIntegratorTest, LargeTimeStep) {
  // integrator_->set_option("dt", 100);
  // EXPECT_NO_THROW(integrator_->integrate(0.0));
}

// Invalid Substeps
TEST_F(RungeKuttaIntegratorTest, InvalidSubsteps) {
  // EXPECT_THROW(std::make_unique<runge_kutta>(
  //                  CuDensityMatState(handle_, mock_initial_state_data(),
  //                             mock_hilbert_space_dims()),
  //                  0.0, time_stepper_, 3),
  //              std::invalid_argument);
}

TEST_F(RungeKuttaIntegratorTest, CheckEvolve) {
  cudm_helper helper(handle_);
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  auto spin_op_x = cudaq::spin_operator::x(0);
  cudaq::product_operator<cudaq::matrix_operator> ham1 =
      2.0 * M_PI * 0.1 * spin_op_x;
  cudaq::operator_sum<cudaq::matrix_operator> ham(ham1);
  SystemDynamics system;
  system.hamiltonian = &ham;
  system.modeExtents = dims;

  for (int integratorOrder : {1, 2, 4}) {
    std::cout << "Test RK order " << integratorOrder << "\n";
    cudaq::runge_kutta integrator;
    integrator.dt = 0.001;
    integrator.order = integratorOrder;
    constexpr std::size_t numDataPoints = 10;
    double t = 0.0;
    auto initialState = cudaq::state::from_data(initialStateVec);
    // initialState.dump();
    auto *simState = cudaq::state_helper::getSimulationState(&initialState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    EXPECT_TRUE(castSimState != nullptr);
    castSimState->initialize_cudm(handle_, dims);
    integrator.set_state(initialState, 0.0);
    cudaq::Schedule schedule(
        cudaq::linspace(0, 1.0 * numDataPoints, numDataPoints));
    integrator.set_system(system, schedule);
    std::vector<std::complex<double>> outputStateVec(2);
    for (std::size_t i = 1; i < numDataPoints; ++i) {
      integrator.integrate(i);
      auto [t, state] = integrator.get_state();
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
