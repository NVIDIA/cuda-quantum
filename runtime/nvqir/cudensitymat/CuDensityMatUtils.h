/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "CuDensityMatErrorHandling.h"
#include <complex>
#include <concepts>
#include <vector>

namespace cudaq {
namespace dynamics {
// GPU memory management
template <std::floating_point T>
void *createArrayGpu(const std::vector<std::complex<T>> &cpuArray) {
  void *gpuArray{nullptr};
  const std::size_t arraySizeBytes = cpuArray.size() * sizeof(std::complex<T>);
  if (arraySizeBytes > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&gpuArray, arraySizeBytes));
    HANDLE_CUDA_ERROR(cudaMemcpy(gpuArray,
                                 static_cast<const void *>(cpuArray.data()),
                                 arraySizeBytes, cudaMemcpyHostToDevice));
  }
  return gpuArray;
}
inline void destroyArrayGpu(void *gpuArray) {
  if (gpuArray)
    HANDLE_CUDA_ERROR(cudaFree(gpuArray));
}
} // namespace dynamics
} // namespace cudaq
