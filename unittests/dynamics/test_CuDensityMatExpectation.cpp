/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatState.h"
#include "common/EigenDense.h"
#include "test_Mocks.h"
#include <CuDensityMatErrorHandling.h>
#include <CuDensityMatExpectation.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <unsupported/Eigen/KroneckerProduct>

using namespace cudaq;

class CuDensityMatExpectationTest : public ::testing::Test {
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

TEST_F(CuDensityMatExpectationTest, checkCompute) {
  const std::vector<int64_t> dims = {10};
  // Check number operator on boson Fock space
  auto op = cudaq::matrix_operator::number(0);
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator({}, op, dims);

  CuDensityMatExpectation expectation(handle_, cudmOp);

  for (std::size_t stateIdx = 0; stateIdx < dims[0]; ++stateIdx) {
    std::vector<std::complex<double>> initialState(dims[0], 0.0);
    initialState[stateIdx] = 1.0;
    auto inputState =
        std::make_unique<CuDensityMatState>(handle_, initialState, dims);
    expectation.prepare(inputState->get_impl());
    const auto expVal = expectation.compute(inputState->get_impl(), 0.0);
    EXPECT_NEAR(expVal.real(), 1.0 * stateIdx, 1e-12);
    EXPECT_NEAR(expVal.imag(), 0.0, 1e-12);
  }
}

TEST_F(CuDensityMatExpectationTest, checkCompositeSystem) {
  const std::vector<int64_t> dims = {2, 10};
  // Check number operator on boson Fock space
  auto op = cudaq::matrix_operator::number(1);
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator({}, op, dims);

  CuDensityMatExpectation expectation(handle_, cudmOp);

  for (std::size_t stateIdx = 0; stateIdx < dims[1]; ++stateIdx) {
    Eigen::Vector2cd qubit_state;
    qubit_state << 1.0, 0.0;
    Eigen::VectorXcd cavity_state = Eigen::VectorXcd::Zero(dims[1]);
    cavity_state[stateIdx] = 1.0;
    Eigen::VectorXcd initial_state_vec =
        Eigen::kroneckerProduct(cavity_state, qubit_state);
    std::vector<std::complex<double>> initialState(
        initial_state_vec.data(),
        initial_state_vec.data() + initial_state_vec.size());
    auto inputState =
        std::make_unique<CuDensityMatState>(handle_, initialState, dims);
    expectation.prepare(inputState->get_impl());
    const auto expVal = expectation.compute(inputState->get_impl(), 0.0);
    std::cout << "Result: " << expVal << "\n";
    EXPECT_NEAR(expVal.real(), 1.0 * stateIdx, 1e-12);
    EXPECT_NEAR(expVal.imag(), 0.0, 1e-12);
  }
}