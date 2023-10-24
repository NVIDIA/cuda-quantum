/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
  std::vector<double> bs_angles;
  std::vector<double> ps_angles;

  std::vector<std::size_t> input_state;
  std::vector<std::size_t> loop_lengths;

  int n_samples;
};

/// @brief Implementation of the sample method of the cudaq::orca namespace
cudaq::sample_result sample(std::vector<double> &bs_angles,
                            std::vector<double> &ps_angles,
                            std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            int n_samples = 1000000) {
  TBIParameters parameters{bs_angles, ps_angles, input_state, loop_lengths,
                           n_samples};
  cudaq::ExecutionContext context("sample", n_samples);
  auto &platform = get_platform();
  platform.set_exec_ctx(&context, 0);
  cudaq::altLaunchKernel("orca_launch", nullptr, &parameters,
                         sizeof(TBIParameters), 0);

  return context.result;
}

}; // namespace cudaq::orca