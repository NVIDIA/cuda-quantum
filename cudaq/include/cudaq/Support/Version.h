/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq {

/// Get the CUDA-Q version.
const char *getVersion();

/// Get the CUDA-Q full repository revision info.
const char *getFullRepositoryVersion();

/// A generic bug report message.
constexpr const char *bugReportMsg =
    "PLEASE submit a bug report to https://github.com/NVIDIA/cuda-quantum and "
    "include the crash backtrace.\n";

} // namespace cudaq
