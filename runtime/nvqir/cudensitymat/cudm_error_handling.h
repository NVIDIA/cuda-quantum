/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <cublas_v2.h>
#include <cudensitymat.h>
#include <fmt/core.h>
#include <stdexcept>

#define HANDLE_CUDM_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUDENSITYMAT_STATUS_SUCCESS) {                                  \
      throw std::runtime_error(fmt::format("[cudaq] %{} in {} (line {})", err, \
                                           __FUNCTION__, __LINE__));           \
    }                                                                          \
  }

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(fmt::format("[cuda] %{} in {} (line {})", err,  \
                                           __FUNCTION__, __LINE__));           \
    }                                                                          \
  }

#define HANDLE_CUBLAS_ERROR(err)                                               \
  do {                                                                         \
    cublasStatus_t err_ = (err);                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::printf("[cublas] error %d at %s:%d\n", err_, __FILE__, __LINE__);   \
      throw std::runtime_error("cublas error");                                \
    }                                                                          \
  } while (0)
