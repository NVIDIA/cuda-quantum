/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// This header exposes a minimal, free-function interface into the
// quantum_platform without pulling in `cudaq/platform.h` (or any of the
// `quantum_platform` definition's transitive dependencies).
//
// The goal is to avoid header file circular dependencies between the platform
// and its QPUs.
//
// This is possible for the implementations that call `get_platform()`
// internally but whose inputs and ouputs does not depend on the platform
// definition itself.

#include <functional>

namespace cudaq {
class ExecutionContext;

namespace platform {

/// @brief Execute the given function within the given execution context,
/// delegating to the current quantum_platform. This free function avoids
/// a header dependency on platform.h from QPU implementation headers.
void with_execution_context(ExecutionContext &ctx, std::function<void()> f);

} // namespace platform
} // namespace cudaq
