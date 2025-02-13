/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "test_mocks.h"
#include <cudm_error_handling.h>
#include <cudm_helpers.h>
#include <cudm_state.h>
#include <cudm_time_stepper.h>
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
  std::unique_ptr<cudm_helper> helper_;

  void SetUp() override {
    // Create library handle
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));

    // Create helper
    helper_ = std::make_unique<cudm_helper>(handle_);

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
    // HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_));
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

TEST_F(CuDensityMatTimeStepperTest, ComputeStepCheckOutput) {
  const std::vector<std::complex<double>> initialState = {
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  const std::vector<int64_t> dims = {4};
  auto inputState = std::make_unique<cudm_state>(handle_, initialState, dims);
  auto op = cudaq::matrix_operator::create(0);
  auto cudmOp =
      helper_->convert_to_cudensitymat_operator<cudaq::matrix_operator>(
          {}, op, dims); // Initialize the time stepper
  auto time_stepper = std::make_unique<cudm_time_stepper>(handle_, cudmOp);
  auto outputState = time_stepper->compute(*inputState, 0.0, 1.0);

  std::vector<std::complex<double>> outputStateVec(4);
  HANDLE_CUDA_ERROR(cudaMemcpy(
      outputStateVec.data(), outputState.get_device_pointer(),
      outputStateVec.size() * sizeof(std::complex<double>), cudaMemcpyDefault));
  // Create operator move the state up 1 step.
  const std::vector<std::complex<double>> expectedOutputState = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

  for (std::size_t i = 0; i < expectedOutputState.size(); ++i) {
    EXPECT_TRUE(std::abs(expectedOutputState[i] - outputStateVec[i]) < 1e-12);
  }
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(cudmOp));
}

TEST_F(CuDensityMatTimeStepperTest, TimeSteppingWithLindblad) {
  std::vector<std::complex<double>> initial_state;
  initial_state.resize(100, {0.0, 0.0});
  initial_state[5 * 10 + 5] = {1.0, 0.0};

  const std::vector<int64_t> dims = {10};
  auto input_state = std::make_unique<cudm_state>(handle_, initial_state, dims);

  // auto c_op_0 = cudaq::matrix_operator::annihilate(0);
  auto c_op_0 = cudaq::matrix_operator::create(0);
  auto cudm_lindblad_op =
      helper_->compute_lindblad_operator({c_op_0.to_matrix({{0, 10}})}, dims);

  auto time_stepper =
      std::make_unique<cudm_time_stepper>(handle_, cudm_lindblad_op);
  auto output_state = time_stepper_->compute(*input_state, 0.0, 1.0);

  std::cout << "Printing output_state ..." << std::endl;
  output_state.dumpDeviceData();

  std::vector<std::complex<double>> output_state_vec(100);
  HANDLE_CUDA_ERROR(
      cudaMemcpy(output_state_vec.data(), output_state.get_device_pointer(),
                 output_state_vec.size() * sizeof(std::complex<double>),
                 cudaMemcpyDefault));

  helper_->print_complex_vector(output_state_vec);

  EXPECT_TRUE(std::abs(output_state_vec[4 * 10 + 4] -
                       std::complex<double>(1.0, 0.0)) < 1e-12);

  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(cudm_lindblad_op));
}
