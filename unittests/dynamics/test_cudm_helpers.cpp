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
template <typename HandlerTy = std::complex<double>>
cudaq::operator_sum<HandlerTy> initialize_operator_sum() {
  std::vector<int> degrees = {0, 1};

  // Matrix operators
  cudaq::matrix_operator pauli_x("pauli_x", {0});
  cudaq::matrix_operator pauli_z("pauli_z", {1});
  cudaq::product_operator<cudaq::matrix_operator> identity =
      cudaq::matrix_operator::identity(0);

  std::vector<cudaq::matrix_operator> identity_vec = identity.get_terms();
  auto identity_op = cudaq::product_operator<cudaq::matrix_operator>(
      cudaq::scalar_operator(1.0), identity_vec);

  std::vector<cudaq::matrix_operator> pauli_x_vec = {pauli_x};
  auto prod_op_1 = cudaq::product_operator<cudaq::matrix_operator>(
      cudaq::scalar_operator(1.0), pauli_x_vec);

  std::vector<cudaq::matrix_operator> pauli_z_vec = {pauli_z};
  prod_op_1 = prod_op_1 * cudaq::product_operator<cudaq::matrix_operator>(
                              cudaq::scalar_operator(1.0), pauli_z_vec);

  auto prod_op_2 =
      cudaq::product_operator<cudaq::matrix_operator>(
          cudaq::scalar_operator(std::complex<double>(0.5, -0.5))) *
      identity_op;

  std::vector<cudaq::product_operator<cudaq::matrix_operator>> terms = {
      prod_op_1, prod_op_2};
  cudaq::operator_sum<cudaq::product_operator> op_sum(terms);

  return op_sum;
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

  auto op_sum = initialize_operator_sum<std::complex<double>>();

  EXPECT_NO_THROW({
    auto result = cudaq::convert_to_cudensitymat_operator<std::complex<double>>(
        handle, {}, op_sum, mode_extents);
    ASSERT_NE(result, nullptr);
    cudensitymatDestroyOperator(result);
  });
}

// Test invalid handle
TEST_F(CuDensityMatTestFixture, InvalidHandle) {
  cudensitymatHandle_t invalid_handle = nullptr;

  std::vector<int64_t> mode_extents = {2, 2};
  auto op_sum = initialize_operator_sum<std::complex<double>>();

  EXPECT_THROW(cudaq::convert_to_cudensitymat_operator<std::complex<double>>(
                   invalid_handle, {}, op_sum, mode_extents),
               std::runtime_error);
}
