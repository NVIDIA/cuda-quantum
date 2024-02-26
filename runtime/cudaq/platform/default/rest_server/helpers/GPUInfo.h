/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "nlohmann/json.hpp"
namespace cudaq {
// A subset of *cudaDeviceProp* to be populated by the server.
struct CudaDeviceProperties {
  std::string deviceName;
  double memoryClockRateMhz;
  double clockRateMhz;
  std::size_t totalGlobalMemMbytes;
  int driverVersion;
  int runtimeVersion;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(CudaDeviceProperties, deviceName,
                                 memoryClockRateMhz, clockRateMhz,
                                 totalGlobalMemMbytes, driverVersion,
                                 runtimeVersion);
};
std::optional<CudaDeviceProperties> getCudaProperties();
} // namespace cudaq
