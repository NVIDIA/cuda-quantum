/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ExplicitPipelineProfile.h"

namespace cudaq::config {

/// All target IR profiles.
using Profile = std::variant<ExplicitPipelineProfile>;

} // namespace cudaq::config
