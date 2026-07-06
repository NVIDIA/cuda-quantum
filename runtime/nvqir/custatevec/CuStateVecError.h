/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/runtime/logger/cudaq_fmt.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <custatevec.h>

#include <stdexcept>

#define HANDLE_CUSTATEVEC_ERROR(x)                                             \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUSTATEVEC_STATUS_SUCCESS) {                                    \
      throw std::runtime_error(cudaq_fmt::format(                              \
          "[custatevec] %{} in {} (line {})",                                  \
          custatevecGetErrorString(/*status=*/err), __FUNCTION__, __LINE__));  \
    }                                                                          \
  }

#define HANDLE_CUBLAS_ERROR(x)                                                 \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      throw std::runtime_error(                                                \
          cudaq_fmt::format("[cublas] error {} in {} (line {})",               \
                            static_cast<int>(err), __FUNCTION__, __LINE__));   \
    }                                                                          \
  }

#define HANDLE_CURAND_ERROR(x)                                                 \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CURAND_STATUS_SUCCESS) {                                        \
      throw std::runtime_error(                                                \
          cudaq_fmt::format("[curand] error {} in {} (line {})",               \
                            static_cast<int>(err), __FUNCTION__, __LINE__));   \
    }                                                                          \
  }

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(cudaq_fmt::format("[cuda] %{} in {} (line {})", \
                                                 cudaGetErrorString(err),      \
                                                 __FUNCTION__, __LINE__));     \
    }                                                                          \
  }

namespace cudaq::cusv {

/// RAII owner of a cuBLAS handle used for state vector operations.
class CublasHandle {
public:
  CublasHandle() { HANDLE_CUBLAS_ERROR(cublasCreate(/*handle=*/&m_handle)); }
  ~CublasHandle() {
    if (m_handle)
      cublasDestroy(/*handle=*/m_handle);
  }
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;
  cublasHandle_t get() const { return m_handle; }

private:
  cublasHandle_t m_handle = nullptr;
};

} // namespace cudaq::cusv
