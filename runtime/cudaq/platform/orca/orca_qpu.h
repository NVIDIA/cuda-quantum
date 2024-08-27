/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "OrcaFuture.h"
// #include "OrcaSample.h"
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

/// @brief Return type for asynchronous sampling.
using async_sample_result = orca_async_result<sample_result>;

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

// async_sample_result sample_async(std::vector<std::size_t> &input_state,
//                                  std::vector<std::size_t> &loop_lengths,
//                                  std::vector<double> &bs_angles,
//                                  int n_samples = 10000);

}; // namespace cudaq::orca