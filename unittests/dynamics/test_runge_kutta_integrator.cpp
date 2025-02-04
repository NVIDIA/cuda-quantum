// /*******************************************************************************
//  * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "cudaq/runge_kutta_integrator.h"
#include "test_mocks.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

using namespace cudaq;

class RungeKuttaIntegratorTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;
  cudensitymatOperator_t liouvillian_;
  std::shared_ptr<cudm_time_stepper> time_stepper_;
  std::unique_ptr<runge_kutta_integrator<std::complex<double>>> integrator_;
  std::unique_ptr<cudm_state> state_;

  void SetUp() override {
    // Create library handle
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));

    // Create a mock Liouvillian
    liouvillian_ = mock_liouvillian(handle_);

    // Initialize the time stepper
    time_stepper_ = std::make_shared<cudm_time_stepper>(handle_, liouvillian_);
    ASSERT_NE(time_stepper_, nullptr);

    // Create initial state
    state_ = std::make_unique<cudm_state>(handle_, mock_initial_state_data(),
                                          mock_hilbert_space_dims());
    ASSERT_NE(state_, nullptr);
    ASSERT_TRUE(state_->is_initialized());

    double t0 = 0.0;
    // Initialize the integrator (using substeps = 4, for Runge-Kutta method)
    ASSERT_NO_THROW(
        integrator_ =
            std::make_unique<runge_kutta_integrator<std::complex<double>>>(
                std::move(*state_), t0, time_stepper_, 4));
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

// Integration with Euler Method (substeps = 1)
TEST_F(RungeKuttaIntegratorTest, EulerIntegration) {
  auto eulerIntegrator =
      std::make_unique<runge_kutta_integrator<std::complex<double>>>(
          cudm_state(handle_, mock_initial_state_data(),
                     mock_hilbert_space_dims()),
          0.0, time_stepper_, 1);
  eulerIntegrator->set_option("dt", 0.1);
  EXPECT_NO_THROW(eulerIntegrator->integrate(1.0));
}

// Integration with Midpoint Rule (substeps = 2)
TEST_F(RungeKuttaIntegratorTest, MidpointIntegration) {
  auto midpointIntegrator =
      std::make_unique<runge_kutta_integrator<std::complex<double>>>(
          cudm_state(handle_, mock_initial_state_data(),
                     mock_hilbert_space_dims()),
          0.0, time_stepper_, 2);
  midpointIntegrator->set_option("dt", 0.1);
  EXPECT_NO_THROW(midpointIntegrator->integrate(1.0));
}

// Integration with Runge-Kutta 4 (substeps = 4, which is the default value)
TEST_F(RungeKuttaIntegratorTest, RungeKutta4Integration) {
  integrator_->set_option("dt", 0.1);
  EXPECT_NO_THROW(integrator_->integrate(1.0));
}

// Basic Integration Test
TEST_F(RungeKuttaIntegratorTest, BasicIntegration) {
  auto [t_before, state_before] = integrator_->get_state();
  integrator_->set_option("dt", 0.1);

  EXPECT_NO_THROW(integrator_->integrate(1.0));

  auto [t_after, state_after] = integrator_->get_state();
  EXPECT_GT(t_after, t_before);
}

// Multiple Integration Steps
TEST_F(RungeKuttaIntegratorTest, MultipleIntegrationSteps) {
  integrator_->set_option("dt", 0.1);
  integrator_->integrate(0.5);
  auto [t_mid, _] = integrator_->get_state();

  EXPECT_EQ(t_mid, 0.5);

  integrator_->integrate(1.0);
  auto [t_final, __] = integrator_->get_state();

  EXPECT_EQ(t_final, 1.0);
}

// Missing Time Step (dt)
TEST_F(RungeKuttaIntegratorTest, MissingTimeStepOption) {
  auto integrator_missing_dt =
      std::make_unique<runge_kutta_integrator<std::complex<double>>>(
          cudm_state(handle_, mock_initial_state_data(),
                     mock_hilbert_space_dims()),
          0.0, time_stepper_, 2);

  EXPECT_THROW(integrator_missing_dt->integrate(1.0), std::invalid_argument);
}

// Invalid Time Step (dt <= 0)
TEST_F(RungeKuttaIntegratorTest, InvalidTimeStepSize) {
  integrator_->set_option("dt", -0.1);
  EXPECT_THROW(integrator_->integrate(1.0), std::invalid_argument);
}

// Zero Integration Time
TEST_F(RungeKuttaIntegratorTest, ZeroIntegrationTime) {
  auto [t_before, state_before] = integrator_->get_state();
  integrator_->set_option("dt", 0.1);

  EXPECT_NO_THROW(integrator_->integrate(0.0));

  auto [t_after, state_after] = integrator_->get_state();
  EXPECT_EQ(t_before, t_after);
}

// Large Time Step
TEST_F(RungeKuttaIntegratorTest, LargeTimeStep) {
  integrator_->set_option("dt", 100);
  EXPECT_NO_THROW(integrator_->integrate(0.0));
}

// Invalid Substeps
TEST_F(RungeKuttaIntegratorTest, InvalidSubsteps) {
  EXPECT_THROW(std::make_unique<runge_kutta_integrator<std::complex<double>>>(
                   cudm_state(handle_, mock_initial_state_data(),
                              mock_hilbert_space_dims()),
                   0.0, time_stepper_, 3),
               std::invalid_argument);
}
