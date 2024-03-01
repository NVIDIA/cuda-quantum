/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
    info.memoryClockRateMhz = deviceProp.memoryClockRate * 1e-3;
    info.clockRateMhz = deviceProp.clockRate * 1e-3;
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
