/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecError.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace {

TEST(CuStateVecErrorTest, AcceptsSuccessfulOperations) {
  EXPECT_NO_THROW(HANDLE_CUSTATEVEC_ERROR(CUSTATEVEC_STATUS_SUCCESS));
  EXPECT_NO_THROW(HANDLE_CUDA_ERROR(cudaSuccess));
  EXPECT_NO_THROW(HANDLE_CUBLAS_ERROR(CUBLAS_STATUS_SUCCESS));
  EXPECT_NO_THROW(HANDLE_CURAND_ERROR(CURAND_STATUS_SUCCESS));
}

TEST(CuStateVecErrorTest, ReportsCublasErrorAndLocation) {
  try {
    HANDLE_CUBLAS_ERROR(CUBLAS_STATUS_INVALID_VALUE);
    FAIL() << "Expected an exception for an invalid cuBLAS argument.";
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("[cublas]"), std::string::npos);
    EXPECT_NE(message.find("7"), std::string::npos);
    EXPECT_NE(message.find("TestBody"), std::string::npos);
    EXPECT_NE(message.find("line"), std::string::npos);
  }
}

TEST(CuStateVecErrorTest, ReportsCurandErrorAndLocation) {
  try {
    HANDLE_CURAND_ERROR(CURAND_STATUS_OUT_OF_RANGE);
    FAIL() << "Expected an exception for an invalid cuRAND argument.";
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("[curand]"), std::string::npos);
    EXPECT_NE(message.find("104"), std::string::npos);
    EXPECT_NE(message.find("TestBody"), std::string::npos);
    EXPECT_NE(message.find("line"), std::string::npos);
  }
}

TEST(CuStateVecErrorTest, ReportsLibraryErrorAndLocation) {
  try {
    HANDLE_CUSTATEVEC_ERROR(CUSTATEVEC_STATUS_INVALID_VALUE);
    FAIL() << "Expected an exception for an invalid cuStateVec argument.";
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("[custatevec]"), std::string::npos);
    EXPECT_NE(
        message.find(custatevecGetErrorString(CUSTATEVEC_STATUS_INVALID_VALUE)),
        std::string::npos);
    EXPECT_NE(message.find("TestBody"), std::string::npos);
    EXPECT_NE(message.find("line"), std::string::npos);
  }
}

} // namespace
