/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompileTarget.h"
#include "cudaq/platform/RuntimeEndpoint.h"
#include "cudaq/platform/quantum_platform.h"

namespace cudaq::detail {

/// Test-only access to the platform's protected QPU installation API.
class PlatformTestAccess {
public:
  /// Replace every QPU of @p platform with a single (target, endpoint) pair.
  static void setTarget(quantum_platform &platform, const CompileTarget &target,
                        const RuntimeEndpoint &endpoint) {
    platform.clearQPUs();
    platform.addQPU(target, endpoint);
  }

  static void addQPU(quantum_platform &platform, const CompileTarget &target,
                     const RuntimeEndpoint &endpoint) {
    platform.addQPU(target, endpoint);
  }
};

} // namespace cudaq::detail
