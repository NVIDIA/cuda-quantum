/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq {
static constexpr int TIMING_OBSERVE = 1;
static constexpr int TIMING_ALLOCATE = 2;
static constexpr int TIMING_LAUNCH = 3;
static constexpr int TIMING_SAMPLE = 4;
static constexpr int TIMING_GATE_COUNT = 5;
static constexpr int TIMING_MAX_VALUE = 5;
} // namespace cudaq
