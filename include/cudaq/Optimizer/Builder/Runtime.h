/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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

/// Name of the callback into the runtime.
/// A kernel entry procedure can either be replaced with a new function at
/// compile time (see `cudaqGenPrefixName`) or it can be rewritten to call back
/// to the runtime library (and be handled at runtime).
static constexpr const char launchKernelFuncName[] = "altLaunchKernel";
static constexpr const char launchKernelStreamlinedFuncName[] =
    "streamlinedLaunchKernel";
static constexpr const char launchKernelHybridFuncName[] = "hybridLaunchKernel";

static constexpr const char mangledNameMap[] = "quake.mangled_name_map";

static constexpr const char deviceCodeHolderAdd[] =
    "__cudaq_deviceCodeHolderAdd";

static constexpr const char registerLinkableKernel[] =
    "__cudaq_registerLinkableKernel";
static constexpr const char getLinkableKernelKey[] =
    "__cudaq_getLinkableKernelKey";
static constexpr const char getLinkableKernelName[] =
    "__cudaq_getLinkableKernelName";
static constexpr const char getLinkableKernelDeviceSide[] =
    "__cudaq_getLinkableKernelDeviceFunction";

static constexpr const char CudaqRegisterLambdaName[] =
    "cudaqRegisterLambdaName";
static constexpr const char CudaqRegisterArgsCreator[] =
    "cudaqRegisterArgsCreator";
static constexpr const char CudaqRegisterKernelName[] =
    "cudaqRegisterKernelName";

/// Prefix for an analog kernel entry functions.
static constexpr const char cudaqAHSPrefixName[] =
    "__analog_hamiltonian_kernel__";

} // namespace cudaq::runtime
