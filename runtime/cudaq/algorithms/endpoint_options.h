/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <memory>

namespace cudaq {

/// Type-erased, endpoint-specific options attached to a launch policy.
/// Core policies remain independent of the language used by an endpoint.
struct endpoint_options {
  virtual ~endpoint_options() = default;
};

} // namespace cudaq
