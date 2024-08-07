/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
    // Redirect remote platform log (if any)
    // Note: due to the lazy-evaluation mechanism, the execution on the remote
    // platform may occur during accessor API calls (e.g., amplitude, overlap).
    // We want to defer any platform logging so that it would not interrupt
    // potential logging of the result of the API call.
    std::ostringstream remoteLogCout;
    platform.setLogStream(remoteLogCout);
    platform.launchKernel(kernelName, nullptr,
                          static_cast<void *>(argsBuffer.data()),
                          argsBuffer.size(), 0);
    platform.reset_exec_ctx();
    platform.resetLogStream();
    // Cache the info log if any.
    platformExecutionLog = remoteLogCout.str();
    state = std::move(context.simulationState);
  }
}

RemoteSimulationState::~RemoteSimulationState() {
  if (!platformExecutionLog.empty()) {
    // Flush any info log from the remote execution
    printf("%s\n", platformExecutionLog.c_str());
    platformExecutionLog.clear();
  }
}

std::size_t RemoteSimulationState::getNumQubits() const {
  execute();
  return state->getNumQubits();
}

cudaq::SimulationState::Tensor
RemoteSimulationState::getTensor(std::size_t tensorIdx) const {
  execute();
  return state->getTensor(tensorIdx);
}

/// @brief Return all tensors that represent this state
std::vector<cudaq::SimulationState::Tensor>
RemoteSimulationState::getTensors() const {
  return {getTensor()};
}

/// @brief Return the number of tensors that represent this state.
std::size_t RemoteSimulationState::getNumTensors() const { return 1; }

std::complex<double>
RemoteSimulationState::operator()(std::size_t tensorIdx,
                                  const std::vector<std::size_t> &indices) {
  execute();
  return state->operator()(tensorIdx, indices);
}

std::unique_ptr<SimulationState>
RemoteSimulationState::createFromSizeAndPtr(std::size_t size, void *ptr,
                                            std::size_t) {
  throw std::runtime_error("RemoteSimulationState cannot be created from data");
}

void RemoteSimulationState::dump(std::ostream &os) const {
  execute();
  state->dump(os);
}

cudaq::SimulationState::precision RemoteSimulationState::getPrecision() const {
  execute();
  return state->getPrecision();
}

void RemoteSimulationState::destroyState() { state.reset(); }

bool RemoteSimulationState::isDeviceData() const {
  execute();
  return state->isDeviceData();
}

void RemoteSimulationState::toHost(std::complex<double> *clientAllocatedData,
                                   std::size_t numElements) const {
  execute();
  // For remote state, performs data conversion matching the expected data type.
  if (state->getPrecision() == cudaq::SimulationState::precision::fp64) {
    state->toHost(clientAllocatedData, numElements);
  } else {
    cudaq::info("[RemoteSimulationState] Perform data conversion in toHost");
    std::vector<std::complex<float>> temp(numElements);
    state->toHost(temp.data(), numElements);
    std::copy(temp.begin(), temp.end(), clientAllocatedData);
  }
}

void RemoteSimulationState::toHost(std::complex<float> *clientAllocatedData,
                                   std::size_t numElements) const {
  execute();
  // For remote state, performs data conversion matching the expected data type.
  if (state->getPrecision() == cudaq::SimulationState::precision::fp32) {
    state->toHost(clientAllocatedData, numElements);
  } else {
    cudaq::info("[RemoteSimulationState] Perform data conversion in toHost");
    std::vector<std::complex<double>> temp(numElements);
    state->toHost(temp.data(), numElements);
    std::copy(temp.begin(), temp.end(), clientAllocatedData);
  }
}

std::tuple<std::string, void *, std::size_t>
RemoteSimulationState::getKernelInfo() const {
  return std::make_tuple(kernelName, static_cast<void *>(argsBuffer.data()),
                         argsBuffer.size());
}

std::vector<std::complex<double>> RemoteSimulationState::getAmplitudes(
    const std::vector<std::vector<int>> &basisStates) {
  if (basisStates.empty())
    return {};

  if (basisStates[0].size() <= maxQubitCountForFullStateTransfer()) {
    execute();
    return state->getAmplitudes(basisStates);
  }
  auto &platform = cudaq::get_platform();
  // Create an execution context, indicate this is for
  // extracting the state representation
  ExecutionContext context("extract-state");
  std::map<std::vector<int>, std::complex<double>> amplitudeMaps;
  for (const auto &basisState : basisStates)
    amplitudeMaps[basisState] = {};
  context.amplitudeMaps = std::move(amplitudeMaps);
  // Perform the usual pattern set the context,
  // execute and then reset
  platform.set_exec_ctx(&context);
  platform.launchKernel(kernelName, nullptr,
                        static_cast<void *>(argsBuffer.data()),
                        argsBuffer.size(), 0);
  platform.reset_exec_ctx();
  std::vector<std::complex<double>> amplitudes;
  amplitudes.reserve(basisStates.size());
  for (const auto &basisState : basisStates)
    amplitudes.emplace_back(context.amplitudeMaps.value()[basisState]);
  return amplitudes;
}

std::complex<double>
RemoteSimulationState::getAmplitude(const std::vector<int> &basisState) {
  return getAmplitudes({basisState}).front();
}

std::complex<double>
RemoteSimulationState::overlap(const cudaq::SimulationState &other) {
  if (!dynamic_cast<const RemoteSimulationState *>(&other)) {
    // The other state is not a remote state.
    // This could be the case whereby we're unable to lazily evaluate that
    // state, e.g., due to quake code retrieval by name.
    execute();
    return state->overlap(other);
  }

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
  constexpr std::size_t NUM_QUBITS_STATE_TRANSFER = 25; // (~100MB of data)
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
