/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "test_Mocks.h"
#include <CuDensityMatErrorHandling.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
using namespace cudaq;

class CuDensityMatTimeStepperTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;
  cudensitymatOperator_t liouvillian_;
  std::unique_ptr<CuDensityMatTimeStepper> time_stepper_;
  cudaq::state state_ =
      cudaq::state::from_data(std::vector<std::complex<double>>{0.0});

  void SetUp() override {
    // Create library handle
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));

    // Create a mock Liouvillian
    liouvillian_ = mock_liouvillian(handle_);

    // Initialize the time stepper
    time_stepper_ =
        std::make_unique<CuDensityMatTimeStepper>(handle_, liouvillian_);

    state_ = cudaq::state::from_data(mock_initial_state_data());
    auto *simState = cudaq::state_helper::getSimulationState(&state_);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    EXPECT_TRUE(castSimState != nullptr);
    castSimState->initialize_cudm(handle_, mock_hilbert_space_dims(),
                                  /*batchSize=*/1);
    ASSERT_TRUE(castSimState->is_initialized());
  }

  void TearDown() override {
    // Clean up
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(liouvillian_));
    HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_));
  }
};

// Test initialization of CuDensityMatTimeStepper
TEST_F(CuDensityMatTimeStepperTest, Initialization) {
  ASSERT_NE(time_stepper_, nullptr);
  auto *simState = cudaq::state_helper::getSimulationState(&state_);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  EXPECT_TRUE(castSimState != nullptr);
  ASSERT_TRUE(castSimState->is_initialized());
  ASSERT_FALSE(castSimState->is_density_matrix());
}

// Test a single compute step
TEST_F(CuDensityMatTimeStepperTest, ComputeStep) {
  EXPECT_NO_THROW(time_stepper_->compute(state_, 0.0, {}));
}

// Compute step when handle is uninitialized
TEST_F(CuDensityMatTimeStepperTest, ComputeStepUninitializedHandle) {
  CuDensityMatTimeStepper invalidStepper(nullptr, liouvillian_);
  EXPECT_THROW(invalidStepper.compute(state_, 0.0, {}), std::runtime_error);
}

// Compute step when liouvillian is missing
TEST_F(CuDensityMatTimeStepperTest, ComputeStepNoLiouvillian) {
  CuDensityMatTimeStepper invalidStepper(handle_, nullptr);
  EXPECT_THROW(invalidStepper.compute(state_, 0.0, {}), std::runtime_error);
}

// Compute step with large time values
TEST_F(CuDensityMatTimeStepperTest, ComputeStepLargeTimeValues) {
  EXPECT_NO_THROW(time_stepper_->compute(state_, 1e6, {}));
}

TEST_F(CuDensityMatTimeStepperTest, ComputeStepCheckOutput) {
  const std::vector<std::complex<double>> initialState = {
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  const std::vector<int64_t> dims = {4};
  auto inputState = cudaq::state::from_data(initialState);
  auto *simState = cudaq::state_helper::getSimulationState(&inputState);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  EXPECT_TRUE(castSimState != nullptr);
  castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);

  cudaq::boson_op_term op_1 = cudaq::boson_op::create(0);
  cudaq::sum_op<cudaq::matrix_handler> op(op_1);
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator(
                        {}, op, dims); // Initialize the time stepper
  auto time_stepper =
      std::make_unique<CuDensityMatTimeStepper>(handle_, cudmOp);
  auto outputState = time_stepper->compute(inputState, 0.0, {});

  std::vector<std::complex<double>> outputStateVec(4);
  outputState.to_host(outputStateVec.data(), outputStateVec.size());
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
  auto input_state = cudaq::state::from_data(initial_state);
  auto *simState = cudaq::state_helper::getSimulationState(&input_state);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  EXPECT_TRUE(castSimState != nullptr);
  castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);
  cudaq::product_op<cudaq::matrix_handler> c_op_0 =
      cudaq::boson_op::annihilate(0);
  cudaq::sum_op<cudaq::matrix_handler> c_op(c_op_0);
  cudaq::sum_op<cudaq::matrix_handler> zero_op = 0.0 * c_op;
  auto cudm_lindblad_op =
      cudaq::dynamics::Context::getCurrentContext()
          ->getOpConverter()
          .constructLiouvillian({zero_op}, {{c_op}}, dims, {}, true);

  auto time_stepper =
      std::make_unique<CuDensityMatTimeStepper>(handle_, cudm_lindblad_op);
  auto output_state = time_stepper->compute(input_state, 0.0, {});

  std::vector<std::complex<double>> output_state_vec(100);
  output_state.to_host(output_state_vec.data(), output_state_vec.size());
  EXPECT_NEAR(
      std::abs(output_state_vec[4 * 10 + 4] - std::complex<double>(5.0, 0.0)),
      0.0, 1e-12);
  EXPECT_NEAR(
      std::abs(output_state_vec[5 * 10 + 5] - std::complex<double>(-5.0, 0.0)),
      0.0, 1e-12);

  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(cudm_lindblad_op));
}

TEST_F(CuDensityMatTimeStepperTest, CheckScalarCallback) {
  const std::vector<std::complex<double>> initialState = {
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  const std::vector<int64_t> dims = {4};
  auto inputState = cudaq::state::from_data(initialState);
  auto *simState = cudaq::state_helper::getSimulationState(&inputState);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  EXPECT_TRUE(castSimState != nullptr);
  castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);
  const std::string paramName = "alpha";
  const std::complex<double> paramValue{2.0, 3.0};
  std::unordered_map<std::string, std::complex<double>> params{
      {paramName, paramValue}};

  auto function =
      [paramName](const std::unordered_map<std::string, std::complex<double>>
                      &parameters) {
        auto entry = parameters.find(paramName);
        if (entry == parameters.end())
          throw std::runtime_error(
              "Cannot find value of expected parameter named " + paramName);
        return entry->second;
      };

  cudaq::product_op<cudaq::matrix_handler> op_t =
      cudaq::scalar_operator(function) * cudaq::boson_op::create(0);
  cudaq::sum_op<cudaq::matrix_handler> op(op_t);
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator(params, op, dims);
  // Initialize the time stepper
  auto time_stepper =
      std::make_unique<CuDensityMatTimeStepper>(handle_, cudmOp);
  auto outputState = time_stepper->compute(inputState, 1.0, params);
  outputState.dump(std::cout);
  std::vector<std::complex<double>> outputStateVec(4);
  outputState.to_host(outputStateVec.data(), outputStateVec.size());
  // Create operator move the state up 1 step.
  const std::vector<std::complex<double>> expectedOutputState = {
      {0.0, 0.0}, paramValue, {0.0, 0.0}, {0.0, 0.0}};

  for (std::size_t i = 0; i < expectedOutputState.size(); ++i) {
    EXPECT_TRUE(std::abs(expectedOutputState[i] - outputStateVec[i]) < 1e-12);
  }
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(cudmOp));
}

TEST_F(CuDensityMatTimeStepperTest, CheckTensorCallback) {
  const std::vector<std::complex<double>> initialState = {{1.0, 0.0},
                                                          {1.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  auto inputState = cudaq::state::from_data(initialState);
  auto *simState = cudaq::state_helper::getSimulationState(&inputState);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  EXPECT_TRUE(castSimState != nullptr);
  castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);

  const std::string paramName = "beta";
  const std::complex<double> paramValue{2.0, 3.0};
  std::unordered_map<std::string, std::complex<double>> params{
      {paramName, paramValue}};

  auto tensorFunction =
      [paramName](const std::vector<int64_t> &dimensions,
                  const std::unordered_map<std::string, std::complex<double>>
                      &parameters) -> complex_matrix {
    if (dimensions.empty())
      throw std::runtime_error("Empty dimensions vector received!");

    auto entry = parameters.find(paramName);
    if (entry == parameters.end())
      throw std::runtime_error(
          "Cannot find value of expected parameter named " + paramName);

    std::complex<double> value = entry->second;
    complex_matrix mat(2, 2);
    mat[{0, 0}] = value;
    mat[{1, 1}] = std::conj(value);
    mat[{0, 1}] = {0.0, 0.0};
    mat[{1, 0}] = {0.0, 0.0};
    return mat;
  };

  matrix_handler::define("CustomTensorOp", {2}, tensorFunction);
  auto op = cudaq::matrix_handler::instantiate("CustomTensorOp", {0});
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator(params, op, dims);
  // Initialize the time stepper
  auto time_stepper =
      std::make_unique<CuDensityMatTimeStepper>(handle_, cudmOp);
  auto outputState = time_stepper->compute(inputState, 1.0, params);
  outputState.dump(std::cout);
  std::vector<std::complex<double>> outputStateVec(2);
  outputState.to_host(outputStateVec.data(), outputStateVec.size());
  // Create operator move the state up 1 step.
  const std::vector<std::complex<double>> expectedOutputState = {
      paramValue, std::conj(paramValue)};

  for (std::size_t i = 0; i < expectedOutputState.size(); ++i) {
    EXPECT_TRUE(std::abs(expectedOutputState[i] - outputStateVec[i]) < 1e-12);
  }
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(cudmOp));
}

static CuDensityMatState *asCudmState(cudaq::state &cudaqState) {
  auto *simState = cudaq::state_helper::getSimulationState(&cudaqState);
  auto *cudmState = dynamic_cast<CuDensityMatState *>(simState);
  if (!cudmState)
    throw std::runtime_error("Invalid state.");
  return cudmState;
}

TEST_F(CuDensityMatTimeStepperTest, ComputeOperatorOrder) {
  const std::vector<std::complex<double>> initialState = {
      {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
  const std::vector<int64_t> dims = {4};
  auto inputState = cudaq::state::from_data(initialState);
  auto *simState = cudaq::state_helper::getSimulationState(&inputState);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  EXPECT_TRUE(castSimState != nullptr);
  castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);

  cudaq::product_op<cudaq::matrix_handler> op_t =
      cudaq::boson_op::create(0) *
      cudaq::boson_op::annihilate(0); // a_dagger * a
  cudaq::sum_op<cudaq::matrix_handler> op(op_t);
  const auto opMat = op.to_matrix({{0, 4}});

  std::cout << "Op matrix:\n" << opMat.to_string() << "\n";
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator(
                        {}, op, dims); // Initialize the time stepper
  auto time_stepper =
      std::make_unique<CuDensityMatTimeStepper>(handle_, cudmOp);
  auto outputState = time_stepper->compute(inputState, 0.0, {});
  std::vector<std::complex<double>> expectedOutputStateVec(4);
  // Diagonal elements
  for (std::size_t i = 0; i < expectedOutputStateVec.size(); ++i)
    expectedOutputStateVec[i] = opMat[{i, i}];

  std::vector<std::complex<double>> outputStateVec(4);
  outputState.to_host(outputStateVec.data(), outputStateVec.size());
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(cudmOp));
  for (std::size_t i = 0; i < expectedOutputStateVec.size(); ++i) {
    std::cout << "Result = " << outputStateVec[i]
              << "; vs. expected = " << expectedOutputStateVec[i] << "\n";
    EXPECT_TRUE(std::abs(expectedOutputStateVec[i] - outputStateVec[i]) <
                1e-12);
  }
}

TEST_F(CuDensityMatTimeStepperTest, ComputeOperatorOrderDensityMatrix) {
  constexpr int N = 4;
  const std::vector<std::complex<double>> initialState(N * N, 1.0);
  const std::vector<int64_t> dims = {N};
  auto inputState = cudaq::state::from_data(initialState);
  auto *simState = cudaq::state_helper::getSimulationState(&inputState);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  EXPECT_TRUE(castSimState != nullptr);
  castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);

  cudaq::product_op<cudaq::matrix_handler> op_t =
      cudaq::boson_op::create(0) *
      cudaq::boson_op::annihilate(0); // a_dagger * a
  cudaq::sum_op<cudaq::matrix_handler> op(op_t);
  const auto opMat = op.to_matrix({{0, N}});
  cudaq::complex_matrix rho = cudaq::complex_matrix::identity(N);
  for (std::size_t col = 0; col < N; ++col)
    for (std::size_t row = 0; row < N; ++row)
      rho[{row, col}] = 1.0;
  const auto expectedResult =
      std::complex<double>(0.0, -1.0) * (opMat * rho - rho * opMat);
  std::cout << "Expected result:\n" << expectedResult.to_string() << "\n";
  auto cudmOp = cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .constructLiouvillian({op}, {}, dims, {}, true);
  auto time_stepper =
      std::make_unique<CuDensityMatTimeStepper>(handle_, cudmOp);
  auto outputState = time_stepper->compute(inputState, 0.0, {});
  std::vector<std::complex<double>> outputStateVec(initialState.size());
  outputState.to_host(outputStateVec.data(), outputStateVec.size());
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(cudmOp));
  for (std::size_t i = 0; i < outputStateVec.size(); ++i) {
    const auto col = i / N;
    const auto row = i % N;
    std::cout << "Result = " << outputStateVec[i]
              << "; vs. expected = " << expectedResult[{row, col}] << "\n";
    EXPECT_TRUE(std::abs(outputStateVec[i] - expectedResult[{row, col}]) <
                1e-12);
  }
}

TEST_F(CuDensityMatTimeStepperTest, BatchedOperatorSimple) {
  const std::vector<int64_t> dims = {2};
  cudaq::product_op<cudaq::matrix_handler> ham_1 =
      (2.0 * M_PI * 0.1 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham1(ham_1);
  auto ham1Mat = ham_1.to_matrix({{0, 2}});
  cudaq::product_op<cudaq::matrix_handler> ham_2 =
      (2.0 * M_PI * 0.2 * cudaq::spin_op::x(0));
  cudaq::sum_op<cudaq::matrix_handler> ham2(ham_2);
  auto ham2Mat = ham_2.to_matrix({{0, 2}});
  auto batchedLiouvillian =
      cudaq::dynamics::Context::getCurrentContext()
          ->getOpConverter()
          .constructLiouvillian({ham1, ham2}, {}, dims, {}, false);

  auto time_stepper =
      std::make_unique<CuDensityMatTimeStepper>(handle_, batchedLiouvillian);

  auto initialState1 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  auto initialState2 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  std::vector<CuDensityMatState *> states{
      asCudmState(const_cast<state &>(initialState1)),
      asCudmState(const_cast<state &>(initialState2))};

  auto batchedState =
      CuDensityMatState::createBatchedState(handle_, states, dims, false);

  auto outputState =
      time_stepper->compute(cudaq::state(batchedState.release()), 0.0, {});
  outputState.dump();
  auto outputStates =
      CuDensityMatState::splitBatchedState(*asCudmState(outputState));
  EXPECT_EQ(outputStates.size(), 2);

  std::vector<std::complex<double>> outputStateVec1(2);
  std::vector<std::complex<double>> outputStateVec2(2);
  outputStates[0]->toHost(outputStateVec1.data(), outputStateVec1.size());
  outputStates[1]->toHost(outputStateVec2.data(), outputStateVec2.size());
  ham1Mat.dump();
  ham2Mat.dump();
  EXPECT_NEAR(std::abs(std::complex<double>(0.0, -1.0) * ham1Mat[{0, 0}] -
                       outputStateVec1[0]),
              0.0, 1e-6);
  EXPECT_NEAR(std::abs(std::complex<double>(0.0, -1.0) * ham1Mat[{1, 0}] -
                       outputStateVec1[1]),
              0.0, 1e-6);
  EXPECT_NEAR(std::abs(std::complex<double>(0.0, -1.0) * ham2Mat[{0, 0}] -
                       outputStateVec2[0]),
              0.0, 1e-6);
  EXPECT_NEAR(std::abs(std::complex<double>(0.0, -1.0) * ham2Mat[{1, 0}] -
                       outputStateVec2[1]),
              0.0, 1e-6);
}
