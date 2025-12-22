/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/draw.h"
#include <cstdint>
#include <vector>

namespace cudaq {

namespace __internal__ {

std::vector<std::vector<std::int64_t>>
traceToDetectorMzIndices(const cudaq::Trace &trace);

} // namespace __internal__

template <typename QuantumKernel, typename... Args>
std::vector<std::vector<std::int64_t>> detectors(QuantumKernel &&kernel,
                                                 Args &&...args) {
  return __internal__::traceToDetectorMzIndices(
      contrib::traceFromKernel(kernel, std::forward<Args>(args)...));
}

} // namespace cudaq
