/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/integrator.h"

namespace {
[[deprecated("This header is deprecated - please use "
             "cudaq/algorithms/integrator.h instead.")]] constexpr static int
    dynamics_integrator_header_is_deprecated = 0;
constexpr static int please_use_integrator_header =
    dynamics_integrator_header_is_deprecated;
} // namespace
