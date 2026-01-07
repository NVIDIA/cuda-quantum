/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/GPUInfo.h"
#ifdef CUDAQ_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif
namespace cudaq {
#ifdef CUDAQ_ENABLE_CUDA
// Early return nullopt if any CUDA API failed.
#define SAFE_HANDLE_CUDA_ERROR(x)                                              \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error %s in line %d\n", cudaGetErrorString(err), __LINE__); \
      return {};                                                               \
    }                                                                          \
  }
std::optional<CudaDeviceProperties> getCudaProperties() {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
    cudaDeviceProp deviceProp;
    int driverVersion = 0, runtimeVersion = 0;
    SAFE_HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    SAFE_HANDLE_CUDA_ERROR(cudaDriverGetVersion(&driverVersion));
    SAFE_HANDLE_CUDA_ERROR(cudaRuntimeGetVersion(&runtimeVersion));
    CudaDeviceProperties info;
    info.deviceName = deviceProp.name;
    int memoryClockRate;
    SAFE_HANDLE_CUDA_ERROR(cudaDeviceGetAttribute(
        &memoryClockRate, cudaDevAttrMemoryClockRate, 0));
    info.memoryClockRateMhz = memoryClockRate * 1e-3;
    int clockRate;
    SAFE_HANDLE_CUDA_ERROR(
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0));
    info.clockRateMhz = clockRate * 1e-3;
    info.totalGlobalMemMbytes = deviceProp.totalGlobalMem / 1048576;
    info.driverVersion = driverVersion;
    info.runtimeVersion = runtimeVersion;
    return info;
  }

  return {};
}
#else
// No CUDA information, returns null.
std::optional<CudaDeviceProperties> getCudaProperties() { return {}; }
#endif
} // namespace cudaq
