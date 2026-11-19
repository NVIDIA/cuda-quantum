/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file error_handling.h
/// @brief Error handling shared by the validation receiver applications.
///
/// Evaluate each call once and throw on failure with the call, function and
/// line number. Exceptions allow the apps to clean up partial setup.
#pragma once

#include "cudaq/realtime/daemon/dispatcher/cudaq_status.h"
#include <cuda_runtime_api.h>
#include <sstream>
#include <stdexcept>

namespace cudaq::realtime::validation {
// validation_app reports unsupported execution separately (exit 4). Other
// callers can handle it as an ordinary runtime_error, as roce_app does.
struct UnsupportedConfiguration : std::runtime_error {
  using std::runtime_error::runtime_error;
};
} // namespace cudaq::realtime::validation

// Keep these as single statements so they work in unbraced if/else branches.
// Stringize the call rather than requiring a separately maintained API name.
#define HANDLE_REALTIME_ERROR(call)                                            \
  do {                                                                         \
    const auto cudaq_realtime_status = (call);                                 \
    if (cudaq_realtime_status != CUDAQ_OK) {                                   \
      std::ostringstream error_message;                                        \
      error_message << "[realtime] error " << cudaq_realtime_status            \
                    << " from " << #call << " in " << __func__ << " (line "    \
                    << __LINE__ << ")";                                        \
      if (cudaq_realtime_status == CUDAQ_ERR_UNSUPPORTED)                      \
        throw cudaq::realtime::validation::UnsupportedConfiguration(           \
            error_message.str());                                              \
      throw std::runtime_error(error_message.str());                           \
    }                                                                          \
  } while (false)

#define HANDLE_CUDA_ERROR(call)                                                \
  do {                                                                         \
    const auto cudaq_cuda_status = (call);                                     \
    if (cudaq_cuda_status != cudaSuccess) {                                    \
      std::ostringstream error_message;                                        \
      error_message << "[cuda] " << cudaGetErrorString(cudaq_cuda_status)      \
                    << " from " << #call << " in " << __func__ << " (line "    \
                    << __LINE__ << ")";                                        \
      throw std::runtime_error(error_message.str());                           \
    }                                                                          \
  } while (false)
