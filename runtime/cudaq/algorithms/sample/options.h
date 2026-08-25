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

constexpr int DEFAULT_NUM_SHOTS = 1000;
namespace cudaq {

/// @brief Sample options to provide to the sample() / async_sample() functions
///
/// @param shots number of shots to run for the given kernel
/// @param noise noise model to use for the sample operation
/// @param explicit_measurements whether or not to form the global register
/// based on user-supplied measurement order.
struct sample_options {
  std::size_t shots = DEFAULT_NUM_SHOTS;
  cudaq::noise_model noise;
  bool explicit_measurements = false;
};

} // namespace cudaq
