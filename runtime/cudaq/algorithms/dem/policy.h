/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/dem/options.h"
#include "cudaq/algorithms/dem/result.h"
#include <string>

namespace cudaq {

class noise_model;

/// @brief Tag and options for Detector Error Model (DEM) generation.
struct dem_policy {
  static constexpr char name[] = "dem";
  using result_type = dem_result;
  dem_options options;
  std::string kernelName;
  const noise_model *noiseModel = nullptr;
};

} // namespace cudaq
