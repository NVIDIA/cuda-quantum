/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qis/remote_state.h"

namespace cudaq {

void RemoteSimulationState::execute() const {
  if (!state) {
    auto &platform = cudaq::get_platform();
    // Create an execution context, indicate this is for
    // extracting the state representation
    ExecutionContext context("extract-state");
    // Perform the usual pattern set the context,
    // execute and then reset
    platform.set_exec_ctx(&context);
    platform.launchKernel(kernelName, nullptr,
                          static_cast<void *>(argsBuffer.data()),
                          argsBuffer.size(), 0);
    platform.reset_exec_ctx();
    state = std::move(context.simulationState);
  }
}

std::tuple<std::string, void *, std::size_t>
RemoteSimulationState::getKernelInfo() const {
  return std::make_tuple(kernelName, static_cast<void *>(argsBuffer.data()),
                         argsBuffer.size());
}

std::complex<double>
RemoteSimulationState::overlap(const cudaq::SimulationState &other) {
  const auto &otherState = dynamic_cast<const RemoteSimulationState &>(other);
  auto &platform = cudaq::get_platform();
  ExecutionContext context("state-overlap");
  context.overlapComputeStates =
      std::make_pair(static_cast<const cudaq::SimulationState *>(this),
                     static_cast<const cudaq::SimulationState *>(&otherState));
  platform.set_exec_ctx(&context);
  platform.launchKernel(kernelName, nullptr, nullptr, 0, 0);
  platform.reset_exec_ctx();
  assert(context.overlapResult.has_value());
  return context.overlapResult.value();
}
} // namespace cudaq