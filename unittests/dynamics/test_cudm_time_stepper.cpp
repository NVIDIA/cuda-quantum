/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "test_mocks.h"
#include <cudaq/cudm_error_handling.h>
#include <cudaq/cudm_helpers.h>
#include <cudaq/cudm_state.h>
#include <cudaq/cudm_time_stepper.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

using namespace cudaq;

class CuDensityMatTimeStepperTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;
  cudensitymatOperator_t liouvillian_;
  std::unique_ptr<cudm_time_stepper> time_stepper_;
  std::unique_ptr<cudm_state> state_;

  void SetUp() override {
    // Create library handle
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));

    // Create a mock Liouvillian
    liouvillian_ = mock_liouvillian(handle_);

    // Initialize the time stepper
    time_stepper_ = std::make_unique<cudm_time_stepper>(handle_, liouvillian_);

    state_ = std::make_unique<cudm_state>(handle_, mock_initial_state_data(),
                                          mock_hilbert_space_dims());

    ASSERT_TRUE(state_->is_initialized());
  }

  void TearDown() override {
    // Clean up
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(liouvillian_));
    HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_));
  }
};

// Test initialization of cudm_time_stepper
TEST_F(CuDensityMatTimeStepperTest, Initialization) {
  ASSERT_NE(time_stepper_, nullptr);
  ASSERT_TRUE(state_->is_initialized());
  ASSERT_FALSE(state_->is_density_matrix());
}

// Test a single compute step
TEST_F(CuDensityMatTimeStepperTest, ComputeStep) {
  ASSERT_TRUE(state_->is_initialized());
  EXPECT_NO_THROW(time_stepper_->compute(*state_, 0.0, 1.0));
  ASSERT_TRUE(state_->is_initialized());
}

// Compute step when handle is uninitialized
TEST_F(CuDensityMatTimeStepperTest, ComputeStepUninitializedHandle) {
  cudm_time_stepper invalidStepper(nullptr, liouvillian_);
  EXPECT_THROW(invalidStepper.compute(*state_, 0.0, 1.0), std::runtime_error);
}

// Compute step when liouvillian is missing
TEST_F(CuDensityMatTimeStepperTest, ComputeStepNoLiouvillian) {
  cudm_time_stepper invalidStepper(handle_, nullptr);
  EXPECT_THROW(invalidStepper.compute(*state_, 0.0, 1.0), std::runtime_error);
}

// Compute step with mismatched dimensions
TEST_F(CuDensityMatTimeStepperTest, ComputeStepMistmatchedDimensions) {
  EXPECT_THROW(std::unique_ptr<cudm_state> mismatchedState =
                   std::make_unique<cudm_state>(handle_,
                                                mock_initial_state_data(),
                                                std::vector<int64_t>{3, 3}),
               std::invalid_argument);
}

// Compute step with zero step size
TEST_F(CuDensityMatTimeStepperTest, ComputeStepZeroStepSize) {
  EXPECT_THROW(time_stepper_->compute(*state_, 0.0, 0.0), std::runtime_error);
}

// Compute step with large time values
TEST_F(CuDensityMatTimeStepperTest, ComputeStepLargeTimeValues) {
  EXPECT_NO_THROW(time_stepper_->compute(*state_, 1e6, 1e3));
}
