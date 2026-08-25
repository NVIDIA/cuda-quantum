/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/RuntimeEndpoint.h"
#include <nanobind/nanobind.h>

namespace cudaq {

/// Create python bindings for C++ code in this compilation unit.
void bindRuntimeEndpoint(nanobind::module_ &mod);

} // namespace cudaq
