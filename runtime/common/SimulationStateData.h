/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"
#include "cudaq/Optimizer/Transforms/ArgumentDataStore.h"

#include <utility>
#include <vector>

namespace cudaq::runtime {

/// Collect simulation state data from all `cudaq::state *` arguments.
cudaq::opt::ArgumentDataStore readSimulationStateData(
    std::pair<std::size_t, std::vector<std::size_t>> &argumentLayout,
    const void *args);

} // namespace cudaq::runtime
