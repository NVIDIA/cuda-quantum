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
  RuntimeEndpoint ep;
  ep.dispatch = detail::DispatchTable<all_policies>::create(
      []<typename P>() { return &forwardLaunchKernelToQpu<P>; });
  ep.impl = &qpu;
  return ep;
}
