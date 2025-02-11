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

  void SetUp() override {
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
    stream = 0;
  }

  void TearDown() override { cudensitymatDestroy(handle); }
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
//   std::vector<int64_t> mode_extents = {2};

//   std::vector<std::complex<double>> rawData = {{1.0, 0.0}, {0.0, 0.0}};

//   cudaq::cudm_state state(handle, rawData, mode_extents);

//   ASSERT_TRUE(state.is_initialized());

//   EXPECT_NO_THROW(cudaq::scale_state(handle, state.get_impl(), 2.0,
//   stream));
// }

// Test for compute_lindblad_op
TEST_F(CuDensityMatHelpersTestFixture, ComputeLindbladOp) {
  cudaq::cudm_helper helper(handle);
  std::vector<int64_t> mode_extents = {2, 2};

  cudaq::matrix_2 c_op1({1.0, 0.0, 0.0, 0.0}, {2, 2});
  cudaq::matrix_2 c_op2({0.0, 0.0, 0.0, 1.0}, {2, 2});
  std::vector<cudaq::matrix_2> c_ops = {c_op1, c_op2};

  EXPECT_NO_THROW({
    auto lindblad_op = helper.compute_lindblad_operator(c_ops, mode_extents);
    ASSERT_NE(lindblad_op, nullptr)
        << "Error: Lindblad operator creation failed!";

    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    cudensitymatDestroyOperator(lindblad_op);
  });
}

// Test for convert_to_cudensitymat_operator
TEST_F(CuDensityMatHelpersTestFixture, ConvertToCuDensityMatOperator) {
  cudaq::cudm_helper helper(handle);
  std::vector<int64_t> mode_extents = mock_hilbert_space_dims();

  auto op_sum = initialize_operator_sum();

  EXPECT_NO_THROW({
    auto result =
        helper.convert_to_cudensitymat_operator<cudaq::matrix_operator>(
            {}, op_sum, mode_extents);

    ASSERT_NE(result, nullptr);
    cudensitymatDestroyOperator(result);
  });
}

// Test with a higher-dimensional mode extent
TEST_F(CuDensityMatHelpersTestFixture, ConvertHigherDimensionalOperator) {
  cudaq::cudm_helper helper(handle);
  std::vector<int64_t> mode_extents = {3, 3};

  auto op_sum = initialize_operator_sum();

  EXPECT_NO_THROW({
    auto result =
        helper.convert_to_cudensitymat_operator<cudaq::matrix_operator>(
            {}, op_sum, mode_extents);
    ASSERT_NE(result, nullptr);
    cudensitymatDestroyOperator(result);
  });
}

// Test with a coefficient callback function
TEST_F(CuDensityMatHelpersTestFixture, ConvertOperatorWithCallback) {
  cudaq::cudm_helper helper(handle);
  std::vector<int64_t> mode_extents = {2, 2};

  auto callback_function = [](std::map<std::string, std::complex<double>>) {
    return std::complex<double>(1.5, 0.0);
  };

  cudaq::scalar_operator scalar_callback(callback_function);

  auto op_sum = scalar_callback * cudaq::matrix_operator::create(0);

  EXPECT_NO_THROW({
    auto result =
        helper.convert_to_cudensitymat_operator<cudaq::matrix_operator>(
            {}, op_sum, mode_extents);
    ASSERT_NE(result, nullptr);
    cudensitymatDestroyOperator(result);
  });
}

// Test with tensor callback function
TEST_F(CuDensityMatHelpersTestFixture, ConvertOperatorWithTensorCallback) {
  cudaq::cudm_helper helper(handle);
  std::vector<int64_t> mode_extents = {2, 2};

  cudaq::matrix_operator matrix_op("CustomOp", {0, 1});

  auto wrapped_tensor_callback = helper._wrap_tensor_callback(matrix_op);

  ASSERT_NE(wrapped_tensor_callback.callback, nullptr);

  // auto op_sum = cudaq::operator_sum<cudaq::matrix_operator>(matrix_op) +
  // matrix_op;

  // EXPECT_NO_THROW({
  //   auto result =
  //       cudaq::convert_to_cudensitymat_operator<cudaq::matrix_operator>(
  //           handle, {}, op_sum, mode_extents);
  //   ASSERT_NE(result, nullptr);
  //   cudensitymatDestroyOperator(result);
  // });
}

// Test for appending a scalar to a term
TEST_F(CuDensityMatHelpersTestFixture, AppendScalarToTerm) {
  cudaq::cudm_helper helper(handle);
  cudensitymatOperatorTerm_t term;
  std::vector<int64_t> mode_extents = {2, 2};

  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
      handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
      &term));

  cudaq::scalar_operator scalar_op(2.0);

  EXPECT_NO_THROW(helper.append_scalar_to_term(term, scalar_op));

  cudensitymatDestroyOperatorTerm(term);
}

// Test for appending a matrix_operator
// TEST_F(CuDensityMatHelpersTestFixture, AppendElementaryOperatorToTerm) {
//   cudaq::cudm_helper helper(handle);
//   cudensitymatOperatorTerm_t term;
//   std::vector<int64_t> mode_extents = {2, 2};

//   HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
//       handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
//       &term));

//   cudaq::matrix_operator matrix_op = mock_matrix_operator("CustomOp", 0);

//   auto flat_matrix = helper.flatten_matrix(
//       matrix_op.to_matrix(helper.convert_dimensions(mode_extents), {}));
//   auto subspace_extents = helper.get_subspace_extents(mode_extents, {0, 1});

//   auto elementary_op =
//   helper.create_elementary_operator(subspace_extents, flat_matrix);
//   ASSERT_NE(elementary_op, nullptr);

//   EXPECT_NO_THROW(helper.append_elementary_operator_to_term(
//       term, elementary_op, {0, 1}));

//   cudensitymatDestroyOperatorTerm(term);
//   cudensitymatDestroyElementaryOperator(elementary_op);
// }
