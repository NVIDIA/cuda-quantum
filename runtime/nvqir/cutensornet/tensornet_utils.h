/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cutensornet.h"
#include <algorithm>
#include <complex>
#include <random>

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
inline void *
allocateGateMatrix(const std::vector<std::complex<double>> &gateMatHost) {
  // Copy quantum gates to Device memory
  void *d_gate{nullptr};
  const auto sizeBytes = gateMatHost.size() * sizeof(std::complex<double>);
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gate, sizeBytes));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gate, gateMatHost.data(), sizeBytes,
                               cudaMemcpyHostToDevice));
  return d_gate;
}

/// @brief Generate an array of random values in the range (0.0, max)
inline std::vector<double>
randomValues(uint64_t num_samples, double max_value,
             std::size_t seed = std::random_device()()) {
  std::mt19937 randomEngine(seed);
  std::vector<double> rs;
  rs.reserve(num_samples);
  std::uniform_real_distribution<double> distr(0.0, max_value);
  for (uint64_t i = 0; i < num_samples; ++i) {
    rs.emplace_back(distr(randomEngine));
  }
  std::sort(rs.begin(), rs.end());
  return rs;
}

/// @brief Struct to allocate and clean up device memory scratch space.
struct ScratchDeviceMem {
  void *d_scratch = nullptr;
  std::size_t scratchSize = 0;
  // Compute the scratch size to allocate.
  void computeScratchSize() {
    // Query the free memory on Device
    std::size_t freeSize{0}, totalSize{0};
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
    scratchSize = (freeSize - (freeSize % 4096)) /
                  2; // use half of available memory with alignment
  }

  ScratchDeviceMem() {
    computeScratchSize();
    // Try allocate device memory
    auto errCode = cudaMalloc(&d_scratch, scratchSize);
    if (errCode == cudaErrorMemoryAllocation) {
      // This indicates race condition whereby other GPU code is allocating
      // memory while we are calling cudaMemGetInfo.
      // Attempt to redo the allocation with an updated cudaMemGetInfo data.
      computeScratchSize();
      HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
    } else {
      HANDLE_CUDA_ERROR(errCode);
    }
  }
  ~ScratchDeviceMem() { HANDLE_CUDA_ERROR(cudaFree(d_scratch)); }
};

/// Initialize `cutensornet` MPI Comm
/// If MPI is not available, fallback to an empty implementation.
void initCuTensornetComm(cutensornetHandle_t cutnHandle);

/// Reset `cutensornet` MPI Comm, e.g., in preparation for shutdown.
/// Note: this will make sure no further MPI activities from `cutensornet` can
/// occur once MPI has been finalized by CUDAQ.
void resetCuTensornetComm(cutensornetHandle_t cutnHandle);