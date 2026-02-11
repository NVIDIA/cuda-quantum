/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <string>
#include <vector>

namespace cudaq {

using LayoutInfoType = std::pair<std::size_t, std::vector<std::size_t>>;

LayoutInfoType getLayoutInfo(const std::string &name,
                             void *opt_module = nullptr);
} // namespace cudaq
