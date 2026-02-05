/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/cudaq_fmt.h"
#include "cuda_runtime.h"
#include "custabilizer.h"
#include <stdexcept>

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(cudaq_fmt::format("[cuda] %{} in {} (line {})", \
                                                 cudaq_fmt::underlying(err),   \
                                                 __FUNCTION__, __LINE__));     \
    }                                                                          \
  }

#define HANDLE_CUST_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUSTABILIZER_STATUS_SUCCESS) {                                  \
      throw std::runtime_error(cudaq_fmt::format(                              \
          "[custabilizer] {} in {} (line {})",                                 \
          custabilizerGetErrorString(err), __FUNCTION__, __LINE__));           \
    }                                                                          \
  }
