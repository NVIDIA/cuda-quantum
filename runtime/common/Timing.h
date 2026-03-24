/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// The intent of this file is to have no other header dependencies so that it
// can always be included everywhere without inheriting any additional
// dependencies.

namespace cudaq {
static constexpr int TIMING_OBSERVE = 1;
static constexpr int TIMING_ALLOCATE = 2;
static constexpr int TIMING_LAUNCH = 3;
static constexpr int TIMING_SAMPLE = 4;
static constexpr int TIMING_GATE_COUNT = 5;
static constexpr int TIMING_JIT = 6;
static constexpr int TIMING_JIT_PASSES = 7;
static constexpr int TIMING_RUN = 8;
static constexpr int TIMING_TENSORNET = 9;
static constexpr int TIMING_MAX_VALUE = 9;
bool isTimingTagEnabled(int tag);
} // namespace cudaq
