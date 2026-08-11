/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

/// \file cuda_check.h
/// \brief Lightweight CUDA error checking macros for the dynamics support
/// helpers (matrix exponential, propagator / Hamiltonian caches).

#include <cuda_runtime.h>
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

} // namespace cudaq::detail

#define CUDA_CHECK(call)                                                       \
  ::cudaq::detail::checkCudaError((call), __FILE__, __LINE__)
