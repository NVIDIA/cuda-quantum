/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "test_mocks.h"
#include <cudaq/cudm_error_handling.h>
#include <cudaq/cudm_expectation.h>
#include <cudaq/cudm_helpers.h>
#include <cudaq/cudm_state.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

using namespace cudaq;

class CuDensityExpectationTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;

  void SetUp() override {
    // Create library handle
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));
  }

  void TearDown() override {
    // Clean up
    HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_));
  }
};

TEST_F(CuDensityExpectationTest, checkCompute) {
  const std::vector<int64_t> dims = {10};
  // Check number operator on boson Fock space
  auto op = cudaq::matrix_operator::number(0);
  auto cudmOp = cudaq::convert_to_cudensitymat_operator<cudaq::matrix_operator>(
      handle_, {}, op, dims);

  cudm_expectation expectation(handle_, cudmOp);

  for (std::size_t stateIdx = 0; stateIdx < dims[0]; ++stateIdx) {
    std::vector<std::complex<double>> initialState(dims[0], 0.0);
    initialState[stateIdx] = 1.0;
    auto inputState = std::make_unique<cudm_state>(handle_, initialState, dims);
    expectation.prepare(inputState->get_impl());
    const auto expVal = expectation.compute(inputState->get_impl(), 0.0);
    EXPECT_NEAR(expVal.real(), 1.0 * stateIdx, 1e-12);
    EXPECT_NEAR(expVal.imag(), 0.0, 1e-12);
  }
}
