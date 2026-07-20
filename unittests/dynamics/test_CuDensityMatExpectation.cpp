/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatState.h"
#include "CuDensityMatUtils.h"
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
  auto op = cudaq::matrix_op::number(0);
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator({}, op, dims);

  CuDensityMatExpectation expectation(handle_, cudmOp);

  for (std::size_t stateIdx = 0; stateIdx < dims[0]; ++stateIdx) {
    std::vector<std::complex<double>> initialState(dims[0], 0.0);
    initialState[stateIdx] = 1.0;
    CuDensityMatState inputState(initialState.size(),
                                 cudaq::dynamics::createArrayGpu(initialState));
    inputState.initialize_cudm(handle_, dims, /*batchSize=*/1);
    expectation.prepare(inputState.get_impl());
    const auto expVal =
        expectation.compute(inputState.get_impl(), 0.0, /*batchSize=*/1)[0];
    EXPECT_NEAR(expVal.real(), 1.0 * stateIdx, 1e-12);
    EXPECT_NEAR(expVal.imag(), 0.0, 1e-12);
  }
}

TEST_F(CuDensityMatExpectationTest, checkCompositeSystem) {
  const std::vector<int64_t> dims = {2, 10};
  // Check number operator on boson Fock space
  auto op = cudaq::matrix_op::number(1);
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
    CuDensityMatState inputState(initialState.size(),
                                 cudaq::dynamics::createArrayGpu(initialState));
    inputState.initialize_cudm(handle_, dims, /*batchSize=*/1);
    expectation.prepare(inputState.get_impl());
    const auto expVal =
        expectation.compute(inputState.get_impl(), 0.0, /*batchSize=*/1)[0];
    std::cout << "Result: " << expVal << "\n";
    EXPECT_NEAR(expVal.real(), 1.0 * stateIdx, 1e-12);
    EXPECT_NEAR(expVal.imag(), 0.0, 1e-12);
  }
}

TEST_F(CuDensityMatExpectationTest, checkCompositeSystemDensityMatrix) {
  const std::vector<int64_t> dims = {2, 10};
  // Check number operator on boson Fock space
  auto op = cudaq::matrix_op::number(1);
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
    CuDensityMatState inputPureState(
        initialState.size(), cudaq::dynamics::createArrayGpu(initialState));
    inputPureState.initialize_cudm(handle_, dims, /*batchSize=*/1);
    auto inputState = inputPureState.to_density_matrix();
    inputState.dump(std::cout);
    expectation.prepare(inputState.get_impl());
    const auto expVal =
        expectation.compute(inputState.get_impl(), 0.0, /*batchSize=*/1)[0];
    std::cout << "Result: " << expVal << "\n";
    EXPECT_NEAR(expVal.real(), 1.0 * stateIdx, 1e-12);
    EXPECT_NEAR(expVal.imag(), 0.0, 1e-12);
  }
}

TEST_F(CuDensityMatExpectationTest, checkCompositeSystemDensityMatrixKron) {
  const std::vector<int64_t> dims = {2, 10};
  // Check number operator on boson Fock space
  auto op = cudaq::matrix_op::number(1);
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator({}, op, dims);

  CuDensityMatExpectation expectation(handle_, cudmOp);

  for (std::size_t stateIdx = 0; stateIdx < dims[1]; ++stateIdx) {
    Eigen::Matrix2cd qubit_state;
    qubit_state << 1.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXcd cavity_state = Eigen::MatrixXcd::Zero(dims[1], dims[1]);
    cavity_state(stateIdx, stateIdx) = 1.0;
    Eigen::MatrixXcd initial_state_vec =
        Eigen::kroneckerProduct(cavity_state, qubit_state);
    std::vector<std::complex<double>> initialState(
        initial_state_vec.data(),
        initial_state_vec.data() + initial_state_vec.size());
    CuDensityMatState inputState(initialState.size(),
                                 cudaq::dynamics::createArrayGpu(initialState));
    inputState.initialize_cudm(handle_, dims, /*batchSize=*/1);
    inputState.dump(std::cout);
    expectation.prepare(inputState.get_impl());
    const auto expVal =
        expectation.compute(inputState.get_impl(), 0.0, /*batchSize=*/1)[0];
    std::cout << "Result: " << expVal << "\n";
    EXPECT_NEAR(expVal.real(), 1.0 * stateIdx, 1e-12);
    EXPECT_NEAR(expVal.imag(), 0.0, 1e-12);
  }
}

TEST_F(CuDensityMatExpectationTest, checkBatchedState1) {
  const std::vector<int64_t> dims = {2};
  auto op_t = cudaq::spin::z(0);
  cudaq::sum_op<cudaq::matrix_handler> op(op_t);
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator({}, op, dims);

  CuDensityMatExpectation expectation(handle_, cudmOp);

  std::vector<std::complex<double>> initial_state_zero = {1.0, 0.0};
  std::vector<std::complex<double>> initial_state_one = {0.0, 1.0};
  CuDensityMatState inputState0(
      initial_state_zero.size(),
      cudaq::dynamics::createArrayGpu(initial_state_zero));
  CuDensityMatState inputState1(
      initial_state_one.size(),
      cudaq::dynamics::createArrayGpu(initial_state_one));
  auto batchedState = CuDensityMatState::createBatchedState(
      handle_, {&inputState0, &inputState1}, dims, false);
  EXPECT_EQ(batchedState->getBatchSize(), 2);
  expectation.prepare(batchedState->get_impl());
  const auto expVals =
      expectation.compute(batchedState->get_impl(), 0.0, /*batchSize=*/2);
  EXPECT_EQ(expVals.size(), 2);
  EXPECT_NEAR(std::abs(expVals[0] - 1.0), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(expVals[1] + 1.0), 0.0, 1e-6);
}

TEST_F(CuDensityMatExpectationTest, checkBatchedState2) {
  constexpr int N = 10;
  constexpr int numStates = 4;
  const std::vector<int64_t> dims = {N};
  auto op_t = cudaq::boson_op::number(0);
  cudaq::sum_op<cudaq::matrix_handler> op(op_t);
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator({}, op, dims);
  CuDensityMatExpectation expectation(handle_, cudmOp);

  std::vector<CuDensityMatState> initialStates;
  for (int i = 0; i < numStates; ++i) {
    std::vector<std::complex<double>> psi0_(N, 0.0);
    psi0_[N - i - 1] = 1.0;
    initialStates.emplace_back(
        CuDensityMatState(N, cudaq::dynamics::createArrayGpu(psi0_)));
  }
  std::vector<CuDensityMatState *> initialStatePtrs;
  for (auto &state : initialStates)
    initialStatePtrs.emplace_back(&state);
  auto batchedState = CuDensityMatState::createBatchedState(
      handle_, initialStatePtrs, dims, true);
  EXPECT_EQ(batchedState->getBatchSize(), numStates);

  for (int test = 0; test < 10; ++test) {
    auto clone = CuDensityMatState::clone(*batchedState);
    expectation.prepare(clone->get_impl());
    const auto expVals = expectation.compute(clone->get_impl(), 0.0,
                                             /*batchSize=*/numStates);
    EXPECT_EQ(expVals.size(), numStates);
    int numPhotons = N - 1;
    for (const auto &expVal : expVals) {
      std::cout << "Exp val = " << expVal << "\n";
      EXPECT_NEAR(std::abs(expVal - 1.0 * numPhotons), 0.0, 1e-6);
      numPhotons--;
    }
  }
}
