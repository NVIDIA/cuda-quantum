/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu_state.h"

namespace cudaq {

QPUState::~QPUState() {
  if (!deleters.empty())
    for (std::size_t counter = 0; auto &ptr : args)
      deleters[counter++](ptr);

  args.clear();
  deleters.clear();
}

std::size_t QPUState::getNumQubits() const {
  throw std::runtime_error(
      "getNumQubits is not implemented for quantum hardware");
}

cudaq::SimulationState::Tensor
QPUState::getTensor(std::size_t tensorIdx) const {
  throw std::runtime_error("getTensor is not implemented for quantum hardware");
}

/// @brief Return all tensors that represent this state
std::vector<cudaq::SimulationState::Tensor> QPUState::getTensors() const {
  throw std::runtime_error(
      "getTensors is not implemented for quantum hardware");
  return {getTensor()};
}

/// @brief Return the number of tensors that represent this state.
std::size_t QPUState::getNumTensors() const {
  throw std::runtime_error(
      "getNumTensors is not implemented for quantum hardware");
}

std::complex<double>
QPUState::operator()(std::size_t tensorIdx,
                     const std::vector<std::size_t> &indices) {
  throw std::runtime_error(
      "operator() is not implemented for quantum hardware");
}

std::unique_ptr<SimulationState>
QPUState::createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) {
  throw std::runtime_error(
      "createFromSizeAndPtr is not implemented for quantum hardware");
}

void QPUState::dump(std::ostream &os) const {
  throw std::runtime_error("dump is not implemented for quantum hardware");
}

cudaq::SimulationState::precision QPUState::getPrecision() const {
  throw std::runtime_error(
      "getPrecision is not implemented for quantum hardware");
}

void QPUState::destroyState() {
  // There is no state data so nothing to destroy.
}

bool QPUState::isDeviceData() const {
  throw std::runtime_error(
      "isDeviceData is not implemented for quantum hardware");
}

void QPUState::toHost(std::complex<double> *clientAllocatedData,
                      std::size_t numElements) const {
  throw std::runtime_error("toHost is not implemented for quantum hardware");
}

void QPUState::toHost(std::complex<float> *clientAllocatedData,
                      std::size_t numElements) const {
  throw std::runtime_error("toHost is not implemented for quantum hardware");
}

std::optional<std::pair<std::string, std::vector<void *>>>
QPUState::getKernelInfo() const {
  return std::make_pair(kernelName, args);
}

std::vector<std::complex<double>>
QPUState::getAmplitudes(const std::vector<std::vector<int>> &basisStates) {
  throw std::runtime_error(
      "getAmplitudes is not implemented for quantum hardware");
}

std::complex<double>
QPUState::getAmplitude(const std::vector<int> &basisState) {
  throw std::runtime_error(
      "getAmplitudes is not implemented for quantum hardware");
}

std::complex<double> QPUState::overlap(const cudaq::SimulationState &other) {
  throw std::runtime_error("overlap is not implemented for quantum hardware");
}
} // namespace cudaq
