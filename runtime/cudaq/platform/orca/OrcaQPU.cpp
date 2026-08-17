
/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// #include "common/ExecutionContext.h"
// #include "common/Future.h"
#include "orca_qpu.h"
#include "cudaq/algorithms/launch.h"
#include "cudaq/platform.h"

namespace cudaq::orca {

cudaq::sample_result runSampling(TBIParameters &parameters,
                                 std::size_t qpu_id = 0) {
  sample_policy policy;
  ExecutionContext ctx(sample_policy::name, parameters.n_samples, qpu_id);
  auto &platform = cudaq::get_platform();
  return detail::launch(policy, qpu_id, ctx, platform, [&]() {
    [[maybe_unused]] auto dynamicResult =
        cudaq::altLaunchKernel(sample_policy::kernelName, nullptr, &parameters,
                               sizeof(TBIParameters), 0);
  });
}

async_sample_result runAsyncSampling(TBIParameters &parameters,
                                     std::size_t qpu_id = 0) {
  // Indicate that this is an async exec
  async_sample_policy policy{sample_policy{}};
  async_sample_result futureResult;
  ExecutionContext ctx(sample_policy::name, parameters.n_samples, qpu_id);
  auto &platform = cudaq::get_platform();
  // If we have a non-null future, set it
  futureResult = detail::launch(policy, qpu_id, ctx, platform, [&]() {
    [[maybe_unused]] auto dynamicResult =
        cudaq::altLaunchKernel(sample_policy::kernelName, nullptr, &parameters,
                               sizeof(TBIParameters), 0);
  });
  return futureResult;
}

cudaq::sample_result sample(std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            std::vector<double> &bs_angles,
                            std::vector<double> &ps_angles, int n_samples,
                            std::size_t qpu_id) {
  TBIParameters parameters{input_state, loop_lengths, bs_angles, ps_angles,
                           n_samples};
  return runSampling(parameters, qpu_id);
}

cudaq::sample_result sample(std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            std::vector<double> &bs_angles, int n_samples,
                            std::size_t qpu_id) {
  std::vector<double> ps_angles = {};
  TBIParameters parameters{input_state, loop_lengths, bs_angles, ps_angles,
                           n_samples};
  return runSampling(parameters, qpu_id);
}

async_sample_result sample_async(std::vector<std::size_t> &input_state,
                                 std::vector<std::size_t> &loop_lengths,
                                 std::vector<double> &bs_angles,
                                 std::vector<double> &ps_angles, int n_samples,
                                 std::size_t qpu_id) {
  TBIParameters parameters{input_state, loop_lengths, bs_angles, ps_angles,
                           n_samples};
  return runAsyncSampling(parameters, qpu_id);
}

async_sample_result sample_async(std::vector<std::size_t> &input_state,
                                 std::vector<std::size_t> &loop_lengths,
                                 std::vector<double> &bs_angles, int n_samples,
                                 std::size_t qpu_id) {
  std::vector<double> ps_angles = {};
  TBIParameters parameters{input_state, loop_lengths, bs_angles, ps_angles,
                           n_samples};
  return runAsyncSampling(parameters, qpu_id);
}

} // namespace cudaq::orca
