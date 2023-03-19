/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "NvidiaPlatformHelper.h"

void cudaq::NvidiaPlatformHelper::createLogicalToPhysicalDeviceMap() {}

std::size_t cudaq::NvidiaPlatformHelper::setQPU(const std::size_t deviceID) {
  return 0;
}

int cudaq::NvidiaPlatformHelper::getNumQPUs() { return 0; }
