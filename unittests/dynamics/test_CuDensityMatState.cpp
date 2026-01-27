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

// Regression test for density matrix indexing bug.
// The bug was that operator() used the total dimension (dim*dim) instead of
// single-side dimension (dim) for bounds checking and linear index calculation.
TEST_F(CuDensityMatStateTest, DensityMatrixIndexing) {
  // Create a 2-qubit density matrix (4x4 = 16 elements)
  // For |00><00| state
  CuDensityMatState state(densityMatrixData.size(),
                          cudaq::dynamics::createArrayGpu(densityMatrixData));
  state.initialize_cudm(handle, hilbertSpaceDims, /*batchSize=*/1);
  EXPECT_TRUE(state.is_density_matrix());

  // Valid indices for 4x4 density matrix are 0, 1, 2, 3
  // The bug would compute linear index as i * 16 + j instead of i * 4 + j

  // Test (0,0) - this worked even with the bug because 0*16+0 == 0*4+0
  auto val00 = state(0, {0, 0});
  EXPECT_NEAR(val00.real(), 1.0, 1e-12);
  EXPECT_NEAR(val00.imag(), 0.0, 1e-12);

  // Test (1,1) - this would fail with the bug: 1*16+1=17 > 16 elements
  auto val11 = state(0, {1, 1});
  EXPECT_NEAR(val11.real(), 0.0, 1e-12);
  EXPECT_NEAR(val11.imag(), 0.0, 1e-12);

  // Test (2,2) - this would fail with the bug: 2*16+2=34 > 16 elements
  auto val22 = state(0, {2, 2});
  EXPECT_NEAR(val22.real(), 0.0, 1e-12);
  EXPECT_NEAR(val22.imag(), 0.0, 1e-12);

  // Test (3,3) - this would fail with the bug: 3*16+3=51 > 16 elements
  auto val33 = state(0, {3, 3});
  EXPECT_NEAR(val33.real(), 0.0, 1e-12);
  EXPECT_NEAR(val33.imag(), 0.0, 1e-12);

  // Test off-diagonal elements
  auto val03 = state(0, {0, 3});
  EXPECT_NEAR(val03.real(), 0.0, 1e-12);
  auto val30 = state(0, {3, 0});
  EXPECT_NEAR(val30.real(), 0.0, 1e-12);

  // Test out-of-bounds access is rejected
  EXPECT_THROW(state(0, {4, 0}), std::runtime_error);
  EXPECT_THROW(state(0, {0, 4}), std::runtime_error);
  EXPECT_THROW(state(0, {4, 4}), std::runtime_error);
}

// Test indexing for single-qubit density matrix
TEST_F(CuDensityMatStateTest, SingleQubitDensityMatrixIndexing) {
  // 1-qubit system: 2x2 density matrix (4 elements)
  std::vector<std::complex<double>> singleQubitDm = {
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  CuDensityMatState state(singleQubitDm.size(),
                          cudaq::dynamics::createArrayGpu(singleQubitDm));
  state.initialize_cudm(handle, {2}, /*batchSize=*/1);
  EXPECT_TRUE(state.is_density_matrix());

  // Valid indices are 0, 1
  auto val00 = state(0, {0, 0});
  EXPECT_NEAR(val00.real(), 1.0, 1e-12);

  auto val11 = state(0, {1, 1});
  EXPECT_NEAR(val11.real(), 0.0, 1e-12);

  // Out-of-bounds
  EXPECT_THROW(state(0, {2, 0}), std::runtime_error);
  EXPECT_THROW(state(0, {0, 2}), std::runtime_error);
}
