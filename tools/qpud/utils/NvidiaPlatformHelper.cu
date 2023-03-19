/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include <sys/wait.h>
#include <unistd.h>

#include <map>
#include <string>
#include <utility>

#include "NvidiaPlatformHelper.h"
#include "cuda_runtime_api.h"
namespace cudaq {
void NvidiaPlatformHelper::createLogicalToPhysicalDeviceMap() {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  int counter = 0;
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    if (std::string(prop.name).find("Display") == std::string::npos) {
      logical_to_physical_device_id.insert({counter, i});
      counter++;
    }
  }
}
std::size_t NvidiaPlatformHelper::setQPU(const std::size_t deviceID) {
  int currentQPU = logical_to_physical_device_id[deviceID];
  cudaSetDevice(currentQPU);
  return currentQPU;
}
int NvidiaPlatformHelper::getNumQPUs() {
  return logical_to_physical_device_id.size();
}
} // namespace cudaq
