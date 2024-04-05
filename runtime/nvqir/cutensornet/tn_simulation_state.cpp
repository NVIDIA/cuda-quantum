/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "tn_simulation_state.h"
#include <cuComplex.h>

namespace nvqir {
int deviceFromPointer(void *ptr) {
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.device;
}
std::size_t TensorNetSimulationState::getNumQubits() const {
  return m_state->getNumQubits();
}

TensorNetSimulationState::TensorNetSimulationState(
    std::unique_ptr<TensorNetState> inState)
    : m_state(std::move(inState)) {}

TensorNetSimulationState::~TensorNetSimulationState() {}

std::complex<double>
TensorNetSimulationState::overlap(const cudaq::SimulationState &other) {
  // TODO:
  return 0.0;
}

std::complex<double>
TensorNetSimulationState::getAmplitude(const std::vector<int> &basisState) {
  std::vector<int32_t> projectedModes(m_state->getNumQubits());
  std::iota(projectedModes.begin(), projectedModes.end(), 0);
  std::vector<int64_t> projectedModeValues;
  projectedModeValues.assign(basisState.begin(), basisState.end());
  auto subStateVec =
      m_state->getStateVector(projectedModes, projectedModeValues);
  assert(subStateVec.size() == 1);
  return subStateVec[0];
}

cudaq::SimulationState::Tensor
TensorNetSimulationState::getTensor(std::size_t tensorIdx) const {
  // TODO:
  return cudaq::SimulationState::Tensor();
}

std::vector<cudaq::SimulationState::Tensor>
TensorNetSimulationState::getTensors() const {
  // TODO
  return {};
}

std::size_t TensorNetSimulationState::getNumTensors() const {
  // TODO:
  return m_state->getNumQubits();
}

void TensorNetSimulationState::destroyState() {
  cudaq::info("mps-state destroying state vector handle.");
  m_state.reset();
}

} // namespace nvqir