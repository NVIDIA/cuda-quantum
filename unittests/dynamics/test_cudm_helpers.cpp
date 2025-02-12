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
#include <cudaq/operators.h>
#include <gtest/gtest.h>

// Initialize operator_sum
cudaq::operator_sum<cudaq::matrix_operator> initialize_operator_sum() {
  return cudaq::matrix_operator::create(0) + cudaq::matrix_operator::create(1);
}

class CuDensityMatHelpersTestFixture : public ::testing::Test {
protected:
  cudensitymatHandle_t handle;
  cudaStream_t stream;
  std::unique_ptr<cudaq::cudm_helper> helper;
  std::unique_ptr<cudaq::cudm_state> state;

  void SetUp() override {
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
    stream = 0;
    helper = std::make_unique<cudaq::cudm_helper>(handle);

    std::vector<int64_t> mode_extents = {2};
    std::vector<std::complex<double>> rawData = {{1.0, 0.0}, {0.0, 0.0}};
    state = std::make_unique<cudaq::cudm_state>(handle, rawData, mode_extents);
  }

  void TearDown() override { HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); }
};

// Test for initialize_state
TEST_F(CuDensityMatHelpersTestFixture, InitializeState) {
  std::vector<int64_t> mode_extents = {2};

  std::vector<std::complex<double>> rawData = {{1.0, 0.0}, {0.0, 0.0}};

  cudaq::cudm_state state(handle, rawData, mode_extents);

  ASSERT_TRUE(state.is_initialized());
}

// Test for scale_state
// TEST_F(CuDensityMatHelpersTestFixture, ScaleState) {
//   ASSERT_TRUE(state->is_initialized());

//   EXPECT_NO_THROW(helper->scale_state(state->get_impl(), 2.0,
//   stream));
// }

// Test for compute_lindblad_op
TEST_F(CuDensityMatHelpersTestFixture, ComputeLindbladOp) {
  std::vector<int64_t> mode_extents = {2, 2};

  std::vector<std::complex<double>> c_op1_values = {
      {1.0, 0.0},
      {0.0, 0.0},
      {0.0, 0.0},
      {0.0, 0.0},
  };

  std::vector<std::complex<double>> c_op2_values = {
      {0.0, 0.0},
      {0.0, 1.0},
      {0.0, 0.0},
      {0.0, 0.0},
  };

  cudaq::matrix_2 c_op1(c_op1_values, {2, 2});
  cudaq::matrix_2 c_op2(c_op2_values, {2, 2});
  std::vector<cudaq::matrix_2> c_ops = {c_op1, c_op2};

  EXPECT_NO_THROW({
    auto lindblad_op = helper->compute_lindblad_operator(c_ops, mode_extents);

    ASSERT_NE(lindblad_op, nullptr)
        << "Error: Lindblad operator creation failed!";

    cudensitymatDestroyOperator(lindblad_op);
  });
}

// Test for convert_to_cudensitymat_operator
TEST_F(CuDensityMatHelpersTestFixture, ConvertToCuDensityMatOperator) {
  std::vector<int64_t> mode_extents = mock_hilbert_space_dims();

  auto op_sum = initialize_operator_sum();

  EXPECT_NO_THROW({
    auto result =
        helper->convert_to_cudensitymat_operator<cudaq::matrix_operator>(
            {}, op_sum, mode_extents);

    ASSERT_NE(result, nullptr);
    cudensitymatDestroyOperator(result);
  });
}

// Test with a higher-dimensional mode extent
TEST_F(CuDensityMatHelpersTestFixture, ConvertHigherDimensionalOperator) {
  std::vector<int64_t> mode_extents = {3, 3};

  auto op_sum = initialize_operator_sum();

  EXPECT_NO_THROW({
    auto result =
        helper->convert_to_cudensitymat_operator<cudaq::matrix_operator>(
            {}, op_sum, mode_extents);
    ASSERT_NE(result, nullptr);
    cudensitymatDestroyOperator(result);
  });
}

// Test with a coefficient callback function
TEST_F(CuDensityMatHelpersTestFixture, ConvertOperatorWithCallback) {
  std::vector<int64_t> mode_extents = {2, 2};

  auto callback_function = [](std::map<std::string, std::complex<double>>) {
    return std::complex<double>(1.5, 0.0);
  };

  cudaq::scalar_operator scalar_callback(callback_function);

  auto op_sum = scalar_callback * cudaq::matrix_operator::create(0);

  EXPECT_NO_THROW({
    auto result =
        helper->convert_to_cudensitymat_operator<cudaq::matrix_operator>(
            {}, op_sum, mode_extents);
    ASSERT_NE(result, nullptr);
    cudensitymatDestroyOperator(result);
  });
}

// Test with tensor callback function
TEST_F(CuDensityMatHelpersTestFixture, ConvertOperatorWithTensorCallback) {
  std::vector<int64_t> mode_extents = {2, 2};

  const std::string op_id = "custom_op";
  auto func = [](std::vector<int> dimensions,
                 std::map<std::string, std::complex<double>> _none) {
    if (dimensions.size() != 1)
      throw std::invalid_argument("Must have a singe dimension");
    if (dimensions[0] != 2)
      throw std::invalid_argument("Must have dimension 2");
    auto mat = cudaq::matrix_2(2, 2);
    mat[{1, 0}] = 1.0;
    mat[{0, 1}] = 1.0;
    return mat;
  };
  cudaq::matrix_operator::define(op_id, {-1}, func);
  cudaq::matrix_operator matrix_op(op_id, {0});

  auto wrapped_tensor_callback =
      cudaq::cudm_helper::_wrap_tensor_callback(matrix_op);

  ASSERT_NE(wrapped_tensor_callback.callback, nullptr);

  // auto op_sum = matrix_op + matrix_op;

  // EXPECT_NO_THROW({
  //   auto result =
  //       helper->convert_to_cudensitymat_operator<cudaq::matrix_operator>(
  //           {}, op_sum, mode_extents);
  //   ASSERT_NE(result, nullptr);
  //   cudensitymatDestroyOperator(result);
  // });
}

// Test for appending a scalar to a term
TEST_F(CuDensityMatHelpersTestFixture, AppendScalarToTerm) {
  cudensitymatOperatorTerm_t term;
  std::vector<int64_t> mode_extents = {2, 2};

  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
      handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
      &term));

  cudaq::scalar_operator scalar_op(2.0);

  EXPECT_NO_THROW(helper->append_scalar_to_term(term, scalar_op));

  cudensitymatDestroyOperatorTerm(term);
}
