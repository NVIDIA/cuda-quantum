/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cutensornet.h"
#include <complex>
#include <random>
#include <vector>

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error %s in line %d\n", cudaGetErrorString(err), __LINE__); \
      fflush(stdout);                                                          \
      std::abort();                                                            \
    }                                                                          \
  }

#define HANDLE_CUTN_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUTENSORNET_STATUS_SUCCESS) {                                   \
      printf("cuTensorNet error %s in line %d\n",                              \
             cutensornetGetErrorString(err), __LINE__);                        \
      fflush(stdout);                                                          \
      std::abort();                                                            \
    }                                                                          \
  }

/// @brief Allocate and initialize device memory according to the input host
/// data.
template <typename T>
inline void *
allocateGateMatrix(const std::vector<std::complex<T>> &gateMatHost) {
  // Copy quantum gates to Device memory
  void *d_gate{nullptr};
  const auto sizeBytes = gateMatHost.size() * sizeof(std::complex<T>);
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gate, sizeBytes));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gate, gateMatHost.data(), sizeBytes,
                               cudaMemcpyHostToDevice));
  return d_gate;
}

/// @brief Generate an array of random values in the range (0.0, max)
std::vector<double> randomValues(uint64_t num_samples, double max_value,
                                 std::mt19937 &randomEngine);

/// @brief Struct to allocate and clean up device memory scratch space.
struct ScratchDeviceMem {
  // Device pointer to scratch buffer
  void *d_scratch = nullptr;
  // Actual size in bytes
  std::size_t scratchSize = 0;
  // Ratio to current free memory size to allocate the scratch size.
  // Note: the actual allocation size may slightly be different due to alignment
  // consideration.
  // The default ratio if not otherwise specified.
  static inline constexpr double defaultFreeMemRatio = 0.5;
  double freeMemRatio = defaultFreeMemRatio;

  ScratchDeviceMem();
  // Compute the scratch size to allocate.
  void computeScratchSize();

  // Allocate scratch device memory based on available memory
  void allocate();

  ~ScratchDeviceMem();
};

/// Initialize `cutensornet` MPI Comm
/// If MPI is not available, fallback to an empty implementation.
void initCuTensornetComm(cutensornetHandle_t cutnHandle);

/// Reset `cutensornet` MPI Comm, e.g., in preparation for shutdown.
/// Note: this will make sure no further MPI activities from `cutensornet` can
/// occur once MPI has been finalized by CUDA-Q.
void resetCuTensornetComm(cutensornetHandle_t cutnHandle);
