/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"
#include "cudaq/platform/quantum_platform.h"
#include <functional>
#include <vector>

namespace cudaq::orca {

struct TBIParameters {
  std::vector<std::size_t> input_state;
  std::vector<std::size_t> loop_lengths;

  std::vector<double> bs_angles;
  std::vector<double> ps_angles;

  int n_samples;
};

/// @brief Implementation of the sample method of the cudaq::orca namespace
cudaq::sample_result sample(std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            std::vector<double> &bs_angles,
                            std::vector<double> &ps_angles,
                            int n_samples = 10000);
cudaq::sample_result sample(std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            std::vector<double> &bs_angles,
                            int n_samples = 10000);
}; // namespace cudaq::orca