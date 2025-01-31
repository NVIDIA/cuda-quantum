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

// Test fixture class
class RungeKuttaIntegratorTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;
  cudensitymatOperator_t liouvillian_;
  std::shared_ptr<cudm_time_stepper> time_stepper_;
  std::unique_ptr<runge_kutta_integrator> integrator_;
  std::unique_ptr<cudm_state> state_;

  void SetUp() override {
    // Create library handle
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));

    // Create a mock Liouvillian
    liouvillian_ = mock_liouvillian(handle_);

    // Initialize the time stepper
    time_stepper_ = std::make_shared<cudm_time_stepper>(handle_, liouvillian_);

    // Create initial state
    state_ = std::make_unique<cudm_state>(handle_, mock_initial_state_data(),
                                          mock_hilbert_space_dims());

    double t0 = 0.0;
    // Initialize the integrator (using substeps = 2, for mid-point rule)
    integrator_ =
        std::make_unique<runge_kutta_integrator>(*state_, t0, time_stepper_, 2);

    ASSERT_TRUE(state_->is_initialized());
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
  // ASSERT_TRUE(state_->is_initialized());
}
