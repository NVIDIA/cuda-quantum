/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/NoiseModel.h"
#include <cstddef>
#include <optional>

namespace cudaq {

/// @brief Observe options to provide as an argument to the `observe()`,
/// `async_observe()` functions.
/// @param shots number of shots to run for the given kernel, or -1 if not
/// applicable.
/// @param noise noise model to use for the sample operation
/// @param num_trajectories is the optional number of trajectories to be used
/// when computing the expectation values in the presence of noise. This
/// parameter is only applied to simulation backends that support noisy
/// simulation of trajectories.
struct observe_options {
  int shots = -1;
  cudaq::noise_model noise;
  std::optional<std::size_t> num_trajectories;
};
} // namespace cudaq
