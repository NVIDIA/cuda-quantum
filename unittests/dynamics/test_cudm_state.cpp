/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <cudaq/cudm_error_handling.h>
#include <cudaq/cudm_helpers.h>
#include <cudaq/cudm_state.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace cudaq;

class CuDensityMatStateTest : public ::testing::Test {
protected:
  void SetUp() override {
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

  void TearDown() override {}

  std::vector<int64_t> hilbertSpaceDims;
  std::vector<std::complex<double>> stateVectorData;
  std::vector<std::complex<double>> densityMatrixData;
};

TEST_F(CuDensityMatStateTest, InitializeWithStateVector) {
  cudm_mat_state state(stateVectorData);
  EXPECT_FALSE(state.is_initialized());

  EXPECT_NO_THROW(state.init_state(hilbertSpaceDims));
  EXPECT_TRUE(state.is_initialized());
  EXPECT_FALSE(state.is_density_matrix());

  EXPECT_NO_THROW(state.dump());
}

TEST_F(CuDensityMatStateTest, InitializeWithDensityMatrix) {
  cudm_mat_state state(densityMatrixData);
  EXPECT_FALSE(state.is_initialized());

  EXPECT_NO_THROW(state.init_state(hilbertSpaceDims));
  EXPECT_TRUE(state.is_initialized());
  EXPECT_TRUE(state.is_density_matrix());

  EXPECT_NO_THROW(state.dump());
}

TEST_F(CuDensityMatStateTest, InvalidInitialization) {
  // Data size mismatch for hilbertSpaceDims (2x2 system expects size 4 or 16)
  std::vector<std::complex<double>> invalidData = {
      std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0)};

  cudm_mat_state state(invalidData);
  EXPECT_THROW(state.init_state(hilbertSpaceDims), std::invalid_argument);
}

TEST_F(CuDensityMatStateTest, ToDensityMatrixConversion) {
  cudm_mat_state state(stateVectorData);
  state.init_state(hilbertSpaceDims);

  EXPECT_FALSE(state.is_density_matrix());

  cudm_mat_state densityMatrixState = state.to_density_matrix();

  EXPECT_TRUE(densityMatrixState.is_density_matrix());
  EXPECT_TRUE(densityMatrixState.is_initialized());

  EXPECT_NO_THROW(densityMatrixState.dump());
}

TEST_F(CuDensityMatStateTest, AlreadyDensityMatrixConversion) {
  cudm_mat_state state(densityMatrixData);
  state.init_state(hilbertSpaceDims);

  EXPECT_TRUE(state.is_density_matrix());
  EXPECT_THROW(state.to_density_matrix(), std::runtime_error);
}

TEST_F(CuDensityMatStateTest, DumpUninitializedState) {
  cudm_mat_state state(stateVectorData);
  EXPECT_THROW(state.dump(), std::runtime_error);
}

TEST_F(CuDensityMatStateTest, AttachStorageErrorHandling) {
  cudm_mat_state state(stateVectorData);

  EXPECT_THROW(state.attach_storage(), std::runtime_error);
}

TEST_F(CuDensityMatStateTest, DestructorCleansUp) {
  cudm_mat_state state(stateVectorData);

  EXPECT_NO_THROW(state.init_state(hilbertSpaceDims));
}
