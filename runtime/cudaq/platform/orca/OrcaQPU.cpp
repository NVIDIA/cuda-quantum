
/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// #include "common/ExecutionContext.h"
// #include "common/Future.h"
#include "cudaq/platform.h"
#include "orca_qpu.h"

namespace cudaq::orca {

cudaq::sample_result runSampling(TBIParameters &parameters,
                                 std::size_t qpu_id = 0) {
  std::size_t shots = parameters.n_samples;
  auto ctx = std::make_unique<cudaq::ExecutionContext>("sample", shots, qpu_id);

  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(ctx.get());

  [[maybe_unused]] auto dynamicResult = cudaq::altLaunchKernel(
      "orca_launch", nullptr, &parameters, sizeof(TBIParameters), 0);

  platform.reset_exec_ctx();
  return ctx->result;
}

async_sample_result runAsyncSampling(TBIParameters &parameters,
                                     std::size_t qpu_id = 0) {
  std::size_t shots = parameters.n_samples;
  auto ctx = std::make_unique<cudaq::ExecutionContext>("sample", shots, qpu_id);

  // Indicate that this is an async exec
  cudaq::details::future futureResult;
  ctx->asyncExec = true;

  auto &platform = get_platform();
  platform.set_exec_ctx(ctx.get());

  [[maybe_unused]] auto dynamicResult = cudaq::altLaunchKernel(
      "orca_launch", nullptr, &parameters, sizeof(TBIParameters), 0);

  // If we have a non-null future, set it
  futureResult = ctx->futureResult;

  platform.reset_exec_ctx();
  return async_sample_result(std::move(futureResult));
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
