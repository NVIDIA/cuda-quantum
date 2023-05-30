/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/builder/kernel_builder.h"
#include "cudaq/platform/quantum_platform.h"

namespace cudaq {
quantum_platform *getQuantumPlatformInternal();

/// @brief Return the quantum platform provided by the linked platform library
/// @return
inline quantum_platform &get_platform() {
  return *getQuantumPlatformInternal();
}

// Declare this function, implemented elsewhere
std::string getQIR(const std::string &);

} // namespace cudaq
