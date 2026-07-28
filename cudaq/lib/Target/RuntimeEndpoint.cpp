/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/RuntimeEndpoint.h"
#include "cudaq/platform/qpu.h"

template <typename Policy>
static Policy::result_type
forwardLaunchKernelToQpu(std::any &impl, const Policy &policy,
                         const cudaq::CompiledModule &module,
                         cudaq::KernelArgs args) {
  auto &qpu = *std::any_cast<cudaq::QPU *>(impl);
  return qpu.launchKernel(policy, module, args);
}

cudaq::RuntimeEndpoint cudaq::RuntimeEndpoint::wrapQPU(cudaq::QPU &qpu) {
  return RuntimeEndpoint{
      .sample = forwardLaunchKernelToQpu<sample_policy>,
      .async_sample = forwardLaunchKernelToQpu<async_sample_policy>,
      .observe = forwardLaunchKernelToQpu<observe_policy>,
      .async_observe = forwardLaunchKernelToQpu<async_observe_policy>,
      .run = forwardLaunchKernelToQpu<run_policy>,
      .async_run = forwardLaunchKernelToQpu<async_run_policy>,
      .msm_size = forwardLaunchKernelToQpu<msm_size_policy>,
      .msm = forwardLaunchKernelToQpu<msm_policy>,
      .dem = forwardLaunchKernelToQpu<dem_policy>,
      .ptsbe_sample = forwardLaunchKernelToQpu<ptsbe::sample_policy>,

      // QPU state
      .impl = &qpu,
  };
}
