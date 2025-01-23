/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq/cudm_error_handling.h>
#include <cudaq/cudm_helpers.h>
#include <gtest/gtest.h>

// Initialize operator_sum
cudaq::operator_sum initialize_operator_sum() {
  std::vector<int> degrees = {0, 1};

  // Elementary operators
  cudaq::elementary_operator pauli_x("pauli_x", {0});
  cudaq::elementary_operator pauli_z("pauli_z", {1});
  cudaq::elementary_operator identity = cudaq::elementary_operator::identity(0);

  auto prod_op_1 = cudaq::scalar_operator(std::complex<double>(1.0, 0.0)) *
                   pauli_x * pauli_z;

  auto prod_op_2 =
      cudaq::scalar_operator(std::complex<double>(0.5, -0.5)) * identity;

  cudaq::operator_sum op_sum({prod_op_1, prod_op_2});

  return op_sum;
}

class CuDensityMatTestFixture : public ::testing::Test {
protected:
  cudensitymatHandle_t handle;
  cudaStream_t stream;

  void SetUp() override {
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
    HANDLE_CUDM_ERROR(cudensitymatDestroy(handle));
  }
};

// Test for initialize_state
TEST_F(CuDensityMatTestFixture, InitializeState) {
  std::vector<int64_t> mode_extents = {2, 2};

  auto state = cudaq::initialize_state(handle, CUDENSITYMAT_STATE_PURITY_PURE,
                                       mode_extents);
  ASSERT_NE(state, nullptr);

  cudaq::destroy_state(state);
}

// Test for scale_state
TEST_F(CuDensityMatTestFixture, ScaleState) {
  std::vector<int64_t> mode_extents = {2};

  auto state = cudaq::initialize_state(handle, CUDENSITYMAT_STATE_PURITY_PURE,
                                       mode_extents);
  ASSERT_NE(state, nullptr);

  EXPECT_NO_THROW(cudaq::scale_state(handle, state, 2.0, stream));

  cudaq::destroy_state(state);
}

// Test for compute_lindblad_op
TEST_F(CuDensityMatTestFixture, ComputeLindbladOp) {
  std::vector<int64_t> mode_extents = {2, 2};

  cudaq::matrix_2 c_op1({1.0, 0.0, 0.0, 0.0}, {2, 2});
  cudaq::matrix_2 c_op2({0.0, 0.0, 0.0, 1.0}, {2, 2});
  std::vector<cudaq::matrix_2> c_ops = {c_op1, c_op2};

  auto lindblad_op =
      cudaq::compute_lindblad_operator(handle, c_ops, mode_extents);
  ASSERT_NE(lindblad_op, nullptr);

  cudensitymatDestroyOperator(lindblad_op);
}

// Test for convert_to_cudensitymat_operator
TEST_F(CuDensityMatTestFixture, ConvertToCuDensityMatOperator) {
  std::vector<int64_t> mode_extents = {2, 2};

  auto op_sum = initialize_operator_sum();

  auto result =
      cudaq::convert_to_cudensitymat_operator(handle, {}, op_sum, mode_extents);
  ASSERT_NE(result, nullptr);

  cudensitymatDestroyOperator(result);
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
