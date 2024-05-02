/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "remote_state.h"

namespace cudaq {

void RemoteSimulationState::execute() const {
  if (!executed) {
    auto &platform = cudaq::get_platform();
    // Create an execution context, indicate this is for
    // extracting the state representation
    ExecutionContext context("extract-state");
    // Perform the usual pattern set the context,
    // execute and then reset
    platform.set_exec_ctx(&context);
    platform.launchKernel(kernelName, nullptr, nullptr, 0, 0);
    platform.reset_exec_ctx();
    state.resize(1ULL << context.simulationState->getNumQubits());
    context.simulationState->toHost(state.data(), state.size());
  }
  executed = true;
}

} // namespace cudaq