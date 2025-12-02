/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "lqpu.h"

namespace cudaq::nvqlink {

/// @brief Initialize NVQLink runtime
/// @param cfg LQPU configuration
void initialize(const LQPUConfig& cfg);

/// @brief Shutdown NVQLink runtime
void shutdown();

} // namespace cudaq::nvqlink
