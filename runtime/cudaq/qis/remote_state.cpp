/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "remote_state.h"
#include "common/Logger.h"
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
RemoteSimulationState::getAmplitude(const std::vector<int> &basisState) {
  if (basisState.size() <= maxQubitCountForFullStateTransfer()) {
    execute();
    return state->getAmplitude(basisState);
  }
  auto &platform = cudaq::get_platform();
  // Create an execution context, indicate this is for
  // extracting the state representation
  ExecutionContext context("extract-state");
  context.amplitudeMaps[basisState] = {};
  // Perform the usual pattern set the context,
  // execute and then reset
  platform.set_exec_ctx(&context);
  platform.launchKernel(kernelName, nullptr,
                        static_cast<void *>(argsBuffer.data()),
                        argsBuffer.size(), 0);
  platform.reset_exec_ctx();
  return context.amplitudeMaps[basisState];
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

std::size_t RemoteSimulationState::maxQubitCountForFullStateTransfer() {
  // Default number of qubits for full state vector transfer.
  constexpr std::size_t NUM_QUBITS_STATE_TRANSFER = 30;
  if (auto envVal = std::getenv("CUDAQ_REMOTE_STATE_MAX_QUBIT_COUNT")) {

    const int val = std::stoi(envVal);
    cudaq::info("[RemoteSimulationState] Setting remote state data transfer "
                "qubit count threshold to {}.",
                val);
    return val;
  }
  return NUM_QUBITS_STATE_TRANSFER;
}
} // namespace cudaq