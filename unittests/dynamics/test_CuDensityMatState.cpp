/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatState.h"
#include "CuDensityMatUtils.h"
#include "common/EigenDense.h"
#include <CuDensityMatErrorHandling.h>
#include <complex>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace cudaq;

class CuDensityMatStateTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle;

  void SetUp() override {
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));

    // Set up test data for a single 2-qubit system
    hilbertSpaceDims = {2, 2};

    // State vector (pure state) for |00>
    stateVectorData = {
        std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0)};

    // Density matrix for |00><00|
    densityMatrixData = {
        std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0)};
  }

  void TearDown() override { cudensitymatDestroy(handle); }

  std::vector<int64_t> hilbertSpaceDims;
  std::vector<std::complex<double>> stateVectorData;
  std::vector<std::complex<double>> densityMatrixData;
};

TEST_F(CuDensityMatStateTest, InitializeWithStateVector) {
  CuDensityMatState state(stateVectorData.size(),
                          cudaq::dynamics::createArrayGpu(stateVectorData));
  state.initialize_cudm(handle, hilbertSpaceDims, /*batchSize=*/1);
  EXPECT_TRUE(state.is_initialized());
  EXPECT_FALSE(state.is_density_matrix());
  EXPECT_NO_THROW(state.dump(std::cout));
}

TEST_F(CuDensityMatStateTest, InitializeWithDensityMatrix) {
  CuDensityMatState state(densityMatrixData.size(),
                          cudaq::dynamics::createArrayGpu(densityMatrixData));
  state.initialize_cudm(handle, hilbertSpaceDims, /*batchSize=*/1);
  EXPECT_TRUE(state.is_initialized());
  EXPECT_TRUE(state.is_density_matrix());
  EXPECT_NO_THROW(state.dump(std::cout));
}

TEST_F(CuDensityMatStateTest, InvalidInitialization) {
  // Data size mismatch for hilbertSpaceDims (2x2 system expects size 4 or 16)
  std::vector<std::complex<double>> invalidData = {{1.0, 0.0}, {0.0, 0.0}};
  CuDensityMatState state(invalidData.size(),
                          cudaq::dynamics::createArrayGpu(invalidData));
  EXPECT_THROW(state.initialize_cudm(handle, hilbertSpaceDims, /*batchSize=*/1),
               std::invalid_argument);
}

TEST_F(CuDensityMatStateTest, ToDensityMatrixConversion) {
  CuDensityMatState state(stateVectorData.size(),
                          cudaq::dynamics::createArrayGpu(stateVectorData));
  state.initialize_cudm(handle, hilbertSpaceDims, /*batchSize=*/1);
  EXPECT_FALSE(state.is_density_matrix());

  CuDensityMatState densityMatrixState = state.to_density_matrix();
  EXPECT_TRUE(densityMatrixState.is_density_matrix());
  EXPECT_TRUE(densityMatrixState.is_initialized());
  EXPECT_NO_THROW(densityMatrixState.dump(std::cout));
}

TEST_F(CuDensityMatStateTest, ToDensityMatrixConversionCorrectnessCheck) {
  // Check a range of dimensions
  for (int64_t N = 2; N < 10; N += 2) {
    Eigen::VectorXcd randomVec = Eigen::VectorXcd::Random(N);
    std::vector<std::complex<double>> initialState(
        randomVec.data(), randomVec.data() + randomVec.size());
    CuDensityMatState state(N, cudaq::dynamics::createArrayGpu(initialState));
    state.initialize_cudm(handle, {N}, /*batchSize=*/1);
    EXPECT_FALSE(state.is_density_matrix());
    Eigen::MatrixXcd expectedDensityMatrix = randomVec * randomVec.adjoint();
    std::cout << "Expected:\n" << expectedDensityMatrix << "\n";
    CuDensityMatState densityMatrixState = state.to_density_matrix();
    EXPECT_TRUE(densityMatrixState.is_density_matrix());
    EXPECT_TRUE(densityMatrixState.is_initialized());
    Eigen::MatrixXcd resultVec = Eigen::MatrixXcd::Zero(N, N);
    densityMatrixState.toHost(resultVec.data(), resultVec.size());
    std::cout << "Result:\n" << resultVec << "\n";
    EXPECT_TRUE(expectedDensityMatrix.isApprox(resultVec));
  }
}

TEST_F(CuDensityMatStateTest, AlreadyDensityMatrixConversion) {
  CuDensityMatState state(densityMatrixData.size(),
                          cudaq::dynamics::createArrayGpu(densityMatrixData));
  state.initialize_cudm(handle, hilbertSpaceDims, /*batchSize=*/1);
  EXPECT_TRUE(state.is_density_matrix());
  EXPECT_THROW(state.to_density_matrix(), std::runtime_error);
}

TEST_F(CuDensityMatStateTest, DestructorCleansUp) {
  EXPECT_NO_THROW({
    CuDensityMatState state(stateVectorData.size(),
                            cudaq::dynamics::createArrayGpu(stateVectorData));
  });
}

TEST_F(CuDensityMatStateTest, InitializeWithEmptyRawData) {
  std::vector<std::complex<double>> emptyData;

  EXPECT_THROW(
      CuDensityMatState state(emptyData.size(),
                              cudaq::dynamics::createArrayGpu(emptyData)),
      std::invalid_argument);
}

TEST_F(CuDensityMatStateTest, ConversionForSingleQubitSystem) {
  hilbertSpaceDims = {2};
  stateVectorData = {{1.0, 0.0}, {0.0, 0.0}};
  CuDensityMatState state(stateVectorData.size(),
                          cudaq::dynamics::createArrayGpu(stateVectorData));
  state.initialize_cudm(handle, hilbertSpaceDims, /*batchSize=*/1);

  EXPECT_FALSE(state.is_density_matrix());

  CuDensityMatState densityMatrixState = state.to_density_matrix();
  EXPECT_TRUE(densityMatrixState.is_density_matrix());
  EXPECT_TRUE(densityMatrixState.is_initialized());
  EXPECT_NO_THROW(densityMatrixState.dump(std::cout));
}

TEST_F(CuDensityMatStateTest, InvalidHilbertSpaceDims) {
  // 3x3 space is not supported by the provided rawData size
  hilbertSpaceDims = {3, 3};
  CuDensityMatState state(stateVectorData.size(),
                          cudaq::dynamics::createArrayGpu(stateVectorData));
  EXPECT_THROW(state.initialize_cudm(handle, hilbertSpaceDims, /*batchSize=*/1),
               std::invalid_argument);
}

TEST_F(CuDensityMatStateTest, DumpWorksForInitializedState) {
  CuDensityMatState state(stateVectorData.size(),
                          cudaq::dynamics::createArrayGpu(stateVectorData));
  state.initialize_cudm(handle, hilbertSpaceDims, /*batchSize=*/1);
  EXPECT_NO_THROW(state.dump(std::cout));
}

TEST_F(CuDensityMatStateTest, InitialStateEnum) {
  const std::unordered_map<std::size_t, std::int64_t> dims{{0, 2}, {1, 3}};
  for (auto stateType :
       {cudaq::InitialState::ZERO, cudaq::InitialState::UNIFORM}) {
    for (auto isDm : {false, true}) {
      auto state =
          CuDensityMatState::createInitialState(handle, stateType, dims, isDm);
      state->dump(std::cout);
      EXPECT_TRUE(state->is_initialized());
      EXPECT_EQ(state->is_density_matrix(), isDm);
      Eigen::MatrixXcd hostBuffer = Eigen::MatrixXcd::Zero(6, isDm ? 6 : 1);
      state->toHost(hostBuffer.data(), hostBuffer.size());
      const auto checkNorm = [&]() {
        if (isDm) {
          return std::abs(hostBuffer.trace() - 1.0) < 1e-6;
        } else {
          return std::abs(hostBuffer.squaredNorm() - 1.0) < 1e-6;
        }
      };
      EXPECT_TRUE(checkNorm());
      auto hostBufferView = hostBuffer.reshaped();
      const auto checkVal = [&]() {
        if (stateType == cudaq::InitialState::ZERO) {
          const std::complex<double> firstVal = *hostBufferView.begin();
          // First element is 1.0, the rest are zero
          return std::abs(firstVal - 1.0) < 1e-12 &&
                 std::all_of(hostBufferView.begin() + 1, hostBufferView.end(),
                             [](std::complex<double> val) {
                               return std::abs(val) < 1e-12;
                             });
        } else {
          // All elements are equal.
          // The norm condition should guarantee that it's the expected value.
          const std::complex<double> firstVal = *hostBufferView.begin();
          return std::all_of(hostBufferView.begin(), hostBufferView.end(),
                             [&](std::complex<double> val) {
                               return std::abs(val - firstVal) < 1e-12;
                             });
        }
      };
      EXPECT_TRUE(checkVal());
    }
  }
}
