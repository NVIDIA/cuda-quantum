/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ExecutionContext.h"
#include "common/Future.h"
#include "common/SampleResult.h"
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
using async_sample_result = cudaq::async_result<cudaq::sample_result>;

/// @brief Implementation of the sample method of the cudaq::orca namespace
cudaq::sample_result sample(std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            std::vector<double> &bs_angles,
                            std::vector<double> &ps_angles,
                            int n_samples = 10000, std::size_t qpu_id = 0);

cudaq::sample_result sample(std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            std::vector<double> &bs_angles,
                            int n_samples = 10000, std::size_t qpu_id = 0);

async_sample_result sample_async(std::vector<std::size_t> &input_state,
                                 std::vector<std::size_t> &loop_lengths,
                                 std::vector<double> &bs_angles,
                                 std::vector<double> &ps_angles,
                                 int n_samples = 10000, std::size_t qpu_id = 0);

async_sample_result sample_async(std::vector<std::size_t> &input_state,
                                 std::vector<std::size_t> &loop_lengths,
                                 std::vector<double> &bs_angles,
                                 int n_samples = 10000, std::size_t qpu_id = 0);

}; // namespace cudaq::orca
