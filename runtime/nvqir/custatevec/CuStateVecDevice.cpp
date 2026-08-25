/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecDevice.h"

#include "CuStateVecError.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cstdio>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cudaq::cusv {

std::size_t DeviceMemoryCapacity::usableBytes() const {
  std::size_t result = cudaFreeBytes;
  // cudaMemGetInfo excludes free allocations retained by the default memory
  // pool even though cudaMalloc can reuse them.
  if (poolReservedBytes > poolUsedBytes) {
    const std::size_t reusable = poolReservedBytes - poolUsedBytes;
    result = reusable > std::numeric_limits<std::size_t>::max() - result
                 ? std::numeric_limits<std::size_t>::max()
                 : result + reusable;
  }
  // On the unified-memory GB10 platform, cudaMemGetInfo can omit reclaimable
  // host cache and swap. MemAvailable is a better upper bound there.
  if (computeCapabilityMajor == 12 && computeCapabilityMinor == 1 &&
      systemAvailableBytes)
    result = std::max(result, *systemAvailableBytes);
  return result;
}

std::optional<std::size_t> systemMemAvailableBytes() {
  std::ifstream memoryInfo("/proc/meminfo");
  std::string line;
  while (std::getline(memoryInfo, line)) {
    std::istringstream fields(line);
    std::string key;
    std::size_t kibibytes = 0;
    fields >> key >> kibibytes;
    if (key == "MemAvailable:")
      return kibibytes * std::size_t{1024};
  }
  return std::nullopt;
}

DeviceMemoryCapacity queryDeviceMemoryCapacity() {
  DeviceMemoryCapacity result;
  std::size_t ignoredTotal = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&result.cudaFreeBytes, &ignoredTotal));

  int32_t device = 0;
  HANDLE_CUDA_ERROR(cudaGetDevice(&device));
  HANDLE_CUDA_ERROR(cudaDeviceGetAttribute(&result.computeCapabilityMajor,
                                           cudaDevAttrComputeCapabilityMajor,
                                           device));
  HANDLE_CUDA_ERROR(cudaDeviceGetAttribute(&result.computeCapabilityMinor,
                                           cudaDevAttrComputeCapabilityMinor,
                                           device));

  int32_t memoryPoolsSupported = 0;
  HANDLE_CUDA_ERROR(cudaDeviceGetAttribute(
      &memoryPoolsSupported, cudaDevAttrMemoryPoolsSupported, device));
  if (memoryPoolsSupported) {
    cudaMemPool_t pool = nullptr;
    HANDLE_CUDA_ERROR(cudaDeviceGetDefaultMemPool(&pool, device));
    std::uint64_t reserved = 0;
    std::uint64_t used = 0;
    HANDLE_CUDA_ERROR(cudaMemPoolGetAttribute(
        pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
    HANDLE_CUDA_ERROR(
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
    result.poolReservedBytes = static_cast<std::size_t>(reserved);
    result.poolUsedBytes = static_cast<std::size_t>(used);
  }

  // GB10 (compute capability 12.1) shares coherent memory between the Grace CPU
  // and the GPU, so cudaMemGetInfo underreports what is usable. Fall back to
  // the system MemAvailable figure instead.
  if (result.computeCapabilityMajor == 12 &&
      result.computeCapabilityMinor == 1) {
    result.systemAvailableBytes = systemMemAvailableBytes();
    if (!result.systemAvailableBytes)
      std::fprintf(stderr,
                   "Warning: Could not read MemAvailable for GB10; available "
                   "GPU memory may be underestimated.\n");
  }
  return result;
}

int32_t migrationWireCapacity(std::size_t hostBytes, std::size_t deviceBytes) {
  if (deviceBytes == 0)
    throw std::invalid_argument("Device memory capacity cannot be zero.");
  const std::size_t hostChunks = hostBytes / deviceBytes;
  if (hostChunks <= 1)
    return 0;
  return std::min<int32_t>(
      3, static_cast<int32_t>(std::bit_width(hostChunks) - 1));
}

std::size_t trajectoryBatchSize(std::size_t availableBytes,
                                std::size_t stateBytes, std::size_t workItems,
                                std::optional<int32_t> configuredBatchSize) {
  if (stateBytes == 0 || workItems == 0)
    throw std::invalid_argument(
        "Trajectory batch sizing requires non-zero state and work sizes.");
  std::size_t result =
      configuredBatchSize ? static_cast<std::size_t>(*configuredBatchSize)
                          : std::max<std::size_t>(
                                1, std::bit_floor(availableBytes / stateBytes));
  if (workItems < result)
    result = std::bit_ceil(workItems);
  return result;
}

} // namespace cudaq::cusv
