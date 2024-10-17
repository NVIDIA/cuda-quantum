/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "quantum_state.h"
#include "common/Logger.h"

namespace cudaq {

QuantumState::~QuantumState() {
  if (!platformExecutionLog.empty()) {
    // Flush any info log from the remote execution
    printf("%s\n", platformExecutionLog.c_str());
    platformExecutionLog.clear();
  }

  for (std::size_t counter = 0; auto &ptr : args)
    deleters[counter++](ptr);

  args.clear();
  deleters.clear();
}

std::size_t QuantumState::getNumQubits() const {
  throw std::runtime_error(
      "getNumQubits is not implemented for quantum hardware");
}

cudaq::SimulationState::Tensor
QuantumState::getTensor(std::size_t tensorIdx) const {
  throw std::runtime_error("getTensor is not implemented for quantum hardware");
}

/// @brief Return all tensors that represent this state
std::vector<cudaq::SimulationState::Tensor> QuantumState::getTensors() const {
  throw std::runtime_error(
      "getTensors is not implemented for quantum hardware");
  return {getTensor()};
}

/// @brief Return the number of tensors that represent this state.
std::size_t QuantumState::getNumTensors() const {
  throw std::runtime_error(
      "getNumTensors is not implemented for quantum hardware");
}

std::complex<double>
QuantumState::operator()(std::size_t tensorIdx,
                         const std::vector<std::size_t> &indices) {
  throw std::runtime_error(
      "operator() is not implemented for quantum hardware");
}

std::unique_ptr<SimulationState>
QuantumState::createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) {
  throw std::runtime_error(
      "createFromSizeAndPtr is not implemented for quantum hardware");
}

void QuantumState::dump(std::ostream &os) const {
  throw std::runtime_error("dump is not implemented for quantum hardware");
}

cudaq::SimulationState::precision QuantumState::getPrecision() const {
  throw std::runtime_error(
      "getPrecision is not implemented for quantum hardware");
}

void QuantumState::destroyState() {
  // There is no state data so nothing to destroy.
}

bool QuantumState::isDeviceData() const {
  throw std::runtime_error(
      "isDeviceData is not implemented for quantum hardware");
}

void QuantumState::toHost(std::complex<double> *clientAllocatedData,
                          std::size_t numElements) const {
  throw std::runtime_error("toHost is not implemented for quantum hardware");
}

void QuantumState::toHost(std::complex<float> *clientAllocatedData,
                          std::size_t numElements) const {
  throw std::runtime_error("toHost is not implemented for quantum hardware");
}

std::optional<std::pair<std::string, std::vector<void *>>>
QuantumState::getKernelInfo() const {
  return std::make_pair(kernelName, args);
}

std::vector<std::complex<double>>
QuantumState::getAmplitudes(const std::vector<std::vector<int>> &basisStates) {
  throw std::runtime_error(
      "getAmplitudes is not implemented for quantum hardware");
}

std::complex<double>
QuantumState::getAmplitude(const std::vector<int> &basisState) {
  throw std::runtime_error(
      "getAmplitudes is not implemented for quantum hardware");
}

std::complex<double>
QuantumState::overlap(const cudaq::SimulationState &other) {
  throw std::runtime_error("overlap is not implemented for quantum hardware");
}
} // namespace cudaq
