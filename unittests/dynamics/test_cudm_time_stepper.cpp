/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>
#include <cudaq/cudm_state.h>
#include <cudaq/cudm_time_stepper.h>
#include <cudaq/cudm_helpers.h>
#include <cudaq/cudm_error_handling.h>

using namespace cudaq;

// Mock Liouvillian operator creation
cudensitymatOperator_t mock_liouvillian(cudensitymatHandle_t handle) {
    cudensitymatOperator_t liouvillian;
    std::vector<int64_t> dimensions = {2, 2};
    HANDLE_CUDM_ERROR(cudensitymatCreateOperator(handle, static_cast<int32_t>(dimensions.size()), dimensions.data(), &liouvillian));
    return liouvillian;
}

// Mock Hilbert space dimensions
std::vector<std::complex<double>> mock_initial_state_data() {
    return {
        {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}
    };
}

// Mock initial raw state data
std::vector<int64_t> mock_hilbert_space_dims() {
    return {2, 2};
}

class CuDensityMatTimeStepperTest : public ::testing::Test {
protected:
    cudensitymatHandle_t handle_;
    cudensitymatOperator_t liouvillian_;
    cudm_time_stepper *time_stepper_;
    cudm_state state_;

    CuDensityMatTimeStepperTest() : state_(mock_initial_state_data()) {};

    void SetUp() override {
        // Create library handle
        HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));

        // Create a mock Liouvillian
        liouvillian_ = mock_liouvillian(handle_);

        // Initialize the time stepper
        time_stepper_ = new cudm_time_stepper(liouvillian_, handle_);

        // Initialize the state
        state_.init_state(mock_hilbert_space_dims());

        ASSERT_TRUE(state_.is_initialized());
    }

    void TearDown() override {
        // Clean up
        HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(liouvillian_));
        HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_));
        delete time_stepper_;
    }
};

// Test initialization of cudm_time_stepper
TEST_F(CuDensityMatTimeStepperTest, Initialization) {
    ASSERT_NE(time_stepper_, nullptr);
    ASSERT_TRUE(state_.is_initialized());
    ASSERT_FALSE(state_.is_density_matrix());
}

// Test a single compute step
TEST_F(CuDensityMatTimeStepperTest, ComputeStep) {
    ASSERT_TRUE(state_.is_initialized());
    EXPECT_NO_THROW(time_stepper_->compute(state_, 0.0, 1.0));
    ASSERT_TRUE(state_.is_initialized());
}



