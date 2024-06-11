/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

namespace cudaq::__internal__ {

/// @brief Provide an API call that enables
/// target modification at runtime (primarily used by
/// runtime execution environments, like Python).
void enableTargetModification();

/// @brief Provide an API call that disables
/// target modification at runtime (primarily used by
/// runtime execution environments, like Python).
void disableTargetModification();

/// @brief Provide an API call that returns
/// true if the target is modifiable at runtime.
bool canModifyTarget();
} // namespace cudaq::__internal__