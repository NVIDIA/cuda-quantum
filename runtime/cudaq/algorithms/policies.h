/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/observe/policy.h"
#include "cudaq/algorithms/sample/policy.h"

namespace cudaq {

/// @brief Fallback policy tag used when no specific policy matches.
struct other_policies {};

} // namespace cudaq
