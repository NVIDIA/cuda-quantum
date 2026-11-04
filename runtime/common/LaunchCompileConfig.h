/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompileOptions.h"
#include "cudaq/Target/CompileTarget.h"

namespace cudaq {

/// The target and options to compile a kernel with for one launch.
struct LaunchCompileConfig {
  CompileTarget target;
  CompileOptions options;
};

} // namespace cudaq
