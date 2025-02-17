/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <cudm_error_handling.h>
#include <cudm_helpers.h>
#include "CuDensityMatState.h"
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
  cudm_state state(handle, stateVectorData, hilbertSpaceDims);

  EXPECT_TRUE(state.is_initialized());
  EXPECT_FALSE(state.is_density_matrix());
  EXPECT_NO_THROW(state.dump(std::cout));
}

TEST_F(CuDensityMatStateTest, InitializeWithDensityMatrix) {
  cudm_state state(handle, densityMatrixData, hilbertSpaceDims);

  EXPECT_TRUE(state.is_initialized());
  EXPECT_TRUE(state.is_density_matrix());
  EXPECT_NO_THROW(state.dump(std::cout));
}

TEST_F(CuDensityMatStateTest, InvalidInitialization) {
  // Data size mismatch for hilbertSpaceDims (2x2 system expects size 4 or 16)
  std::vector<std::complex<double>> invalidData = {{1.0, 0.0}, {0.0, 0.0}};

  EXPECT_THROW(cudm_state state(handle, invalidData, hilbertSpaceDims),
               std::invalid_argument);
}

TEST_F(CuDensityMatStateTest, ToDensityMatrixConversion) {
  cudm_state state(handle, stateVectorData, hilbertSpaceDims);
  EXPECT_FALSE(state.is_density_matrix());

  cudm_state densityMatrixState = state.to_density_matrix();
  EXPECT_TRUE(densityMatrixState.is_density_matrix());
  EXPECT_TRUE(densityMatrixState.is_initialized());
  EXPECT_NO_THROW(densityMatrixState.dump(std::cout));
}

TEST_F(CuDensityMatStateTest, AlreadyDensityMatrixConversion) {
  cudm_state state(handle, densityMatrixData, hilbertSpaceDims);

  EXPECT_TRUE(state.is_density_matrix());
  EXPECT_THROW(state.to_density_matrix(), std::runtime_error);
}

TEST_F(CuDensityMatStateTest, DestructorCleansUp) {
  EXPECT_NO_THROW(
      { cudm_state state(handle, stateVectorData, hilbertSpaceDims); });
}

TEST_F(CuDensityMatStateTest, InitializeWithEmptyRawData) {
  std::vector<std::complex<double>> emptyData;

  EXPECT_THROW(cudm_state state(handle, emptyData, hilbertSpaceDims),
               std::invalid_argument);
}

TEST_F(CuDensityMatStateTest, ConversionForSingleQubitSystem) {
  hilbertSpaceDims = {2};
  stateVectorData = {{1.0, 0.0}, {0.0, 0.0}};
  cudm_state state(handle, stateVectorData, hilbertSpaceDims);

  EXPECT_FALSE(state.is_density_matrix());

  cudm_state densityMatrixState = state.to_density_matrix();
  EXPECT_TRUE(densityMatrixState.is_density_matrix());
  EXPECT_TRUE(densityMatrixState.is_initialized());
  EXPECT_NO_THROW(densityMatrixState.dump(std::cout));
}

TEST_F(CuDensityMatStateTest, InvalidHilbertSpaceDims) {
  // 3x3 space is not supported by the provided rawData size
  hilbertSpaceDims = {3, 3};
  EXPECT_THROW(cudm_state state(handle, stateVectorData, hilbertSpaceDims),
               std::invalid_argument);
}

TEST_F(CuDensityMatStateTest, ValidDensityMatrixState) {
  cudm_state state(handle, densityMatrixData, hilbertSpaceDims);
  EXPECT_TRUE(state.is_density_matrix());
  EXPECT_TRUE(state.is_initialized());
}

TEST_F(CuDensityMatStateTest, DumpWorksForInitializedState) {
  cudm_state state(handle, stateVectorData, hilbertSpaceDims);
  EXPECT_NO_THROW(state.dump(std::cout));
}
