/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "CuDensityMatErrorHandling.h"
#include "common/Logger.h"
#include <chrono>
#include <complex>
#include <concepts>
#include <iostream>
#include <vector>

#ifndef NTIMING
#define LOG_API_TIME() ScopedTraceWithContext(__FUNCTION__);
#else
#define LOG_API_TIME()
#endif

namespace cudaq::dynamics {
// Track performance metric for a repetitive task in terms of the number of
// executions and the total elapsed time on this task.
struct PerfMetric {
  std::size_t numCalls = 0;
  double totalTimeMs = 0.0;
  void add(double durationMs);
};

// Scope timer that will update the performance metric indexed by a unique name.
struct PerfMetricScopeTimer {
  PerfMetricScopeTimer(const std::string &name);
  ~PerfMetricScopeTimer();

private:
  std::string m_name;
  std::chrono::time_point<std::chrono::system_clock> m_startTime;
};

// Dump and reset the performance metric
void dumpPerfTrace(std::ostream &os = std::cout);

// Wrapper for CUDA memory allocator.
// This allows us to switch between stream-based/blocking allocation scheme and
// to track performance metric for allocation.
struct DeviceAllocator {
  static inline bool useStreamAllocator = false;

  static void *allocate(std::size_t arraySizeBytes) {
    PerfMetricScopeTimer metricTimer("DeviceAllocator::allocate");
    void *gpuArray{nullptr};
    if (useStreamAllocator) {
      HANDLE_CUDA_ERROR(
          cudaMallocAsync(&gpuArray, arraySizeBytes, /*stream=*/0));
      HANDLE_CUDA_ERROR(cudaStreamSynchronize(0));
    } else {
      HANDLE_CUDA_ERROR(cudaMalloc(&gpuArray, arraySizeBytes));
    }
    return gpuArray;
  }

  static void free(void *gpuArray) {
    if (useStreamAllocator) {
      cudaFreeAsync(gpuArray, 0);
      cudaStreamSynchronize(0);
    } else {
      cudaFree(gpuArray);
    }
  }
};

// Adapted from cuquantum team
// GPU memory management
template <std::floating_point T>
void *createArrayGpu(const std::vector<std::complex<T>> &cpuArray) {
  void *gpuArray{nullptr};
  const std::size_t arraySizeBytes = cpuArray.size() * sizeof(std::complex<T>);
  if (arraySizeBytes > 0) {
    gpuArray = cudaq::dynamics::DeviceAllocator::allocate(arraySizeBytes);
    HANDLE_CUDA_ERROR(cudaMemcpy(gpuArray,
                                 static_cast<const void *>(cpuArray.data()),
                                 arraySizeBytes, cudaMemcpyHostToDevice));
  }
  return gpuArray;
}

// Adapted from cuquantum team
inline void destroyArrayGpu(void *gpuArray) {
  if (gpuArray) {
    cudaq::dynamics::DeviceAllocator::free(gpuArray);
  }
}
} // namespace cudaq::dynamics
