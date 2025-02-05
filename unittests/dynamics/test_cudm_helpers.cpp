/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq/cudm_error_handling.h>
#include <cudaq/cudm_helpers.h>
#include <cudaq/cudm_state.h>
#include <gtest/gtest.h>

// Initialize operator_sum
cudaq::operator_sum<cudaq::matrix_operator> initialize_operator_sum() {
  return cudaq::matrix_operator::create(1) + cudaq::matrix_operator::create(2);
}

class CuDensityMatTestFixture : public ::testing::Test {
protected:
  cudensitymatHandle_t handle;
  cudaStream_t stream;

  void SetUp() override {
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
    stream = 0;
  }

  void TearDown() override { cudensitymatDestroy(handle); }
};

// Test for initialize_state
TEST_F(CuDensityMatTestFixture, InitializeState) {
  std::vector<int64_t> mode_extents = {2};

  std::vector<std::complex<double>> rawData = {{1.0, 0.0}, {0.0, 0.0}};

  cudaq::cudm_state state(handle, rawData, mode_extents);

  ASSERT_TRUE(state.is_initialized());
}

// Test for scale_state
TEST_F(CuDensityMatTestFixture, ScaleState) {
  std::vector<int64_t> mode_extents = {2};

  std::vector<std::complex<double>> rawData = {{1.0, 0.0}, {0.0, 0.0}};

  cudaq::cudm_state state(handle, rawData, mode_extents);

  ASSERT_TRUE(state.is_initialized());

  EXPECT_NO_THROW(cudaq::scale_state(handle, state.get_impl(), 2.0, stream));
}

// Test for compute_lindblad_op
TEST_F(CuDensityMatTestFixture, ComputeLindbladOp) {
  std::vector<int64_t> mode_extents = {2, 2};

  cudaq::matrix_2 c_op1({1.0, 0.0, 0.0, 0.0}, {2, 2});
  cudaq::matrix_2 c_op2({0.0, 0.0, 0.0, 1.0}, {2, 2});
  std::vector<cudaq::matrix_2> c_ops = {c_op1, c_op2};

  EXPECT_NO_THROW({
    auto lindblad_op =
        cudaq::compute_lindblad_operator(handle, c_ops, mode_extents);
    ASSERT_NE(lindblad_op, nullptr);
    cudensitymatDestroyOperator(lindblad_op);
  });
}

// Test for convert_to_cudensitymat_operator
TEST_F(CuDensityMatTestFixture, ConvertToCuDensityMatOperator) {
  std::vector<int64_t> mode_extents = {2, 2};

  auto op_sum = initialize_operator_sum();

  EXPECT_NO_THROW({
    auto result = cudaq::convert_to_cudensitymat_operator(handle, {}, op_sum,
                                                          mode_extents);
    ASSERT_NE(result, nullptr);
    cudensitymatDestroyOperator(result);
  });
}

// Test invalid handle
TEST_F(CuDensityMatTestFixture, InvalidHandle) {
  cudensitymatHandle_t invalid_handle = nullptr;

  std::vector<int64_t> mode_extents = {2, 2};
  auto op_sum = initialize_operator_sum();

  EXPECT_THROW(cudaq::convert_to_cudensitymat_operator(invalid_handle, {},
                                                       op_sum, mode_extents),
               std::runtime_error);
}
