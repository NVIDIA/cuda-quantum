
/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "OrcaFuture.h"
#include "common/ExecutionContext.h"
#include "orca_qpu.h"

namespace cudaq::orca {
cudaq::sample_result sample(std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            std::vector<double> &bs_angles,
                            std::vector<double> &ps_angles, int n_samples) {
  TBIParameters parameters{input_state, loop_lengths, bs_angles, ps_angles,
                           n_samples};
  int qpu_id = 0;
  auto ctx = std::make_unique<cudaq::ExecutionContext>("sample", n_samples);
  auto &platform = get_platform();
  platform.set_exec_ctx(ctx.get(), qpu_id);
  platform.set_current_qpu(qpu_id);

  cudaq::altLaunchKernel("orca_launch", nullptr, &parameters,
                         sizeof(TBIParameters), 0);

  platform.reset_exec_ctx(qpu_id);
  return ctx->result;
}

cudaq::sample_result sample(std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            std::vector<double> &bs_angles, int n_samples) {
  std::vector<double> ps_angles = {};
  TBIParameters parameters{input_state, loop_lengths, bs_angles, ps_angles,
                           n_samples};
  int qpu_id = 0;
  auto ctx = std::make_unique<cudaq::ExecutionContext>("sample", n_samples);
  auto &platform = get_platform();
  platform.set_exec_ctx(ctx.get(), qpu_id);
  platform.set_current_qpu(qpu_id);

  cudaq::altLaunchKernel("orca_launch", nullptr, &parameters,
                         sizeof(TBIParameters), 0);

  platform.reset_exec_ctx(qpu_id);
  return ctx->result;
}

// async_sample_result sample_async(std::vector<std::size_t> &input_state,
//                                  std::vector<std::size_t> &loop_lengths,
//                                  std::vector<double> &bs_angles,
//                                  int n_samples) {
//   std::vector<double> ps_angles = {};
//   TBIParameters parameters{input_state, loop_lengths, bs_angles, ps_angles,
//                            n_samples};
//   details::Orcafuture *futureResult = nullptr;

//   int qpu_id = 0;
//   auto ctx = std::make_unique<OrcaExecutionContext>("sample", n_samples);
//   // Indicate that this is an async exec
//   ctx->asyncExec = futureResult != nullptr;

//   auto &platform = get_platform();
//   platform.set_exec_ctx(ctx.get(), qpu_id);
//   platform.set_current_qpu(qpu_id);

//   cudaq::altLaunchKernel("orca_launch", nullptr, &parameters,
//                          sizeof(TBIParameters), 0);

//   // // If we have a non-null future, set it and return
//   // if (futureResult) {
//   //   *futureResult = ctx->futureResult;
//   //   return std::nullopt;
//   // }

//   // platform.reset_exec_ctx(qpu_id);
//   // return async_sample_result(std::move(futureResult));
//   return ctx->result;
// }

} // namespace cudaq::orca
