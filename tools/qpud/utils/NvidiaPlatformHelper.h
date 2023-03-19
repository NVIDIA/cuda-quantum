/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once
#include <map>

namespace cudaq {

/// This handles how the DGX QunatumPlatform interfaces with the GPU(s).
class NvidiaPlatformHelper {
public:
  NvidiaPlatformHelper() = default;
  ~NvidiaPlatformHelper() = default;

  /// Initialize the device map.
  void createLogicalToPhysicalDeviceMap();

  /// Set the GPU corresponding to the QPU deviceID.
  std::size_t setQPU(const std::size_t deviceID);

  /// Get the number of "QPUs" (GPUs)
  int getNumQPUs();

private:
  /// This maps the IDs of the QPUs to the GPU devices.
  std::map<std::size_t, std::size_t> logical_to_physical_device_id;
};
} // namespace cudaq
