/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Builder/Factory.h"

namespace cudaq::runtime {

/// Prefix for all kernel entry functions.
static constexpr const char cudaqGenPrefixName[] = "__nvqpp__mlirgen__";

/// Convenience constant for the length of the kernel entry prefix.
static constexpr unsigned cudaqGenPrefixLength = sizeof(cudaqGenPrefixName) - 1;

/// Name of the callback into ihe runtime.
/// A kernel entry procedure can either be replaced with a new function at
/// compile time (see `cudaqGenPrefixName`) or it can be rewritten to call back
/// to the runtime library (and be handled at runtime).
static constexpr const char launchKernelFuncName[] = "altLaunchKernel";

} // namespace cudaq::runtime
