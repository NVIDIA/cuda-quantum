/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "MQPUUtils.h"
#include <stdexcept>
#ifdef CUDAQ_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif

int cudaq::getCudaDeviceCount() {
#ifdef CUDAQ_ENABLE_CUDA
  int nDevices{0};
  const auto status = cudaGetDeviceCount(&nDevices);
  return status != cudaSuccess ? 0 : nDevices;
#else
  return 0;
#endif
}

void cudaq::setCudaDevice(int deviceId) {
#ifdef CUDAQ_ENABLE_CUDA
  const auto status = cudaSetDevice(deviceId);
  if (status != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device");
  }
#else
  // No-op if CUDA is not enabled.
#endif
}
