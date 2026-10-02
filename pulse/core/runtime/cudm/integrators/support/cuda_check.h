/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

/// \file cuda_check.h
/// \brief Lightweight CUDA / cuBLAS / cuSOLVER error-checking helpers for the
/// dynamics support code (matrix exponential, propagator / Hamiltonian caches).
/// All of the runtime-API error checks live here in one place.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <string>

namespace cudaq::detail {

/// \brief Check CUDA error and throw on failure.
inline void checkCudaError(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                             std::to_string(line) + ": " +
                             cudaGetErrorString(error));
  }
}

/// \brief Check cuBLAS error and throw on failure.
inline void checkCublasError(cublasStatus_t status, const char *file,
                             int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS error at ") + file + ":" +
                             std::to_string(line) + " - code " +
                             std::to_string(status));
  }
}

/// \brief Check cuSOLVER error and throw on failure.
inline void checkCusolverError(cusolverStatus_t status, const char *file,
                               int line) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuSOLVER error at ") + file + ":" +
                             std::to_string(line) + " - code " +
                             std::to_string(status));
  }
}

} // namespace cudaq::detail

#define CUDA_CHECK(call)                                                       \
  ::cudaq::detail::checkCudaError((call), __FILE__, __LINE__)

#define CUBLAS_CHECK(call)                                                     \
  ::cudaq::detail::checkCublasError((call), __FILE__, __LINE__)

#define CUSOLVER_CHECK(call)                                                   \
  ::cudaq::detail::checkCusolverError((call), __FILE__, __LINE__)
