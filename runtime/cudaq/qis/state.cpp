/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "state.h"
#include "common/EigenDense.h"
#include "cudaq/simulators.h"
#include <iostream>

namespace {
/// Create a shared pointer that calls destroyState at destruction.
std::shared_ptr<cudaq::SimulationState>
makeSharedSimulationState(cudaq::SimulationState *ptrToOwn) {
  return std::shared_ptr<cudaq::SimulationState>(
      ptrToOwn, [](cudaq::SimulationState *ptr) {
        ptr->destroyState();
        delete ptr;
      });
}
} // namespace

namespace cudaq {

state &state::initialize(const state_data &data) {
  auto *simulator = cudaq::get_simulator();
  if (!simulator)
    throw std::runtime_error(
        "[state::from_data] Could not find valid simulator backend.");
  std::shared_ptr<SimulationState> newPtr =
      makeSharedSimulationState(simulator->createStateFromData(data).release());
  std::swap(internal, newPtr);
  return *this;
}

state::state(SimulationState *ptrToOwn)
    : internal(makeSharedSimulationState(ptrToOwn)) {}

SimulationState::precision state::get_precision() const {
  return internal->getPrecision();
}

bool state::is_on_gpu() const { return internal->isDeviceData(); }

SimulationState::Tensor state::get_tensor(std::size_t tensorIdx) const {
  return internal->getTensor(tensorIdx);
}
std::vector<SimulationState::Tensor> state::get_tensors() const {
  return internal->getTensors();
}

std::size_t state::get_num_tensors() const { return internal->getNumTensors(); }

std::size_t state::get_num_qubits() const { return internal->getNumQubits(); }

void state::dump() const { dump(std::cout); }
void state::dump(std::ostream &os) const { internal->dump(os); }

std::complex<double>
state::operator()(const std::initializer_list<std::size_t> &indices,
                  std::size_t tensorIdx) const {
  std::vector<std::size_t> idxVec(indices.begin(), indices.end());
  return (*internal)(tensorIdx, idxVec);
}

std::complex<double> state::operator[](std::size_t idx) const {
  std::size_t numQubits = internal->getNumQubits();
  std::size_t numElements = internal->getNumElements();

  if (!internal->isArrayLike()) {
    // Use amplitude accessor if linear indexing is not supported, e.g., tensor
    // network state.
    std::vector<int> basisState(numQubits, 0);
    // Are we dealing with qudits or qubits?
    /// NOTE: Following check makes assumption that only qubit simulation uses
    /// GPU(s)
    if (!internal->isDeviceData() && std::log2(numElements) / numQubits > 1) {
      for (std::size_t i = 0; i < numQubits; ++i) {
        basisState[i] = 1; // TODO: This is a placeholder. We need to figure out
                           // how to handle qudits.
      }
    } else {
      for (std::size_t i = 0; i < numQubits; ++i) {
        if (idx & (1ULL << i))
          basisState[(numQubits - 1) - i] = 1;
      }
    }
    return internal->getAmplitude(basisState);
  }

  std::size_t newIdx = 0;
  if (std::log2(numElements) / numQubits > 1) {
    newIdx = idx;
  } else {
    for (std::size_t i = 0; i < numQubits; ++i)
      if (idx & (1ULL << i))
        newIdx |= (1ULL << ((numQubits - 1) - i));
  }
  return operator()({newIdx}, 0);
}

std::complex<double> state::operator()(std::size_t idx, std::size_t jdx) const {
  return operator()({idx, jdx}, 0);
}

std::complex<double> state::overlap(const state &other) {
  return internal->overlap(*other.internal.get());
}

std::complex<double> state::amplitude(const std::vector<int> &basisState) {
  return internal->getAmplitude(basisState);
}

std::vector<std::complex<double>>
state::amplitudes(const std::vector<std::vector<int>> &basisStates) {
  return internal->getAmplitudes(basisStates);
}

state &state::operator=(state &&other) {
  // Copy and swap idiom
  std::swap(internal, other.internal);
  return *this;
}

extern "C" {
std::int64_t __nvqpp_cudaq_state_numberOfQubits(state *obj) {
  return obj->get_num_qubits();
}

// Code gen helpers to convert spans (device side data) to state objects.
state *__nvqpp_cudaq_state_createFromData_complex_f64(std::complex<double> *d,
                                                      std::size_t size) {
  if (cudaq::get_simulator()->isSinglePrecision())
    return new state(std::vector<std::complex<float>>{d, d + size});

  return new state(std::vector<std::complex<double>>{d, d + size});
}

state *__nvqpp_cudaq_state_createFromData_f64(double *d, std::size_t size) {

  if (cudaq::get_simulator()->isSinglePrecision())
    return new state(std::vector<std::complex<float>>{d, d + size});

  return new state(std::vector<std::complex<double>>{d, d + size});
}

state *__nvqpp_cudaq_state_createFromData_complex_f32(std::complex<float> *d,
                                                      std::size_t size) {
  if (cudaq::get_simulator()->isSinglePrecision())
    return new state(std::vector<std::complex<float>>{d, d + size});

  return new state(std::vector<std::complex<double>>{d, d + size});
}

state *__nvqpp_cudaq_state_createFromData_f32(float *d, std::size_t size) {

  if (cudaq::get_simulator()->isSinglePrecision())
    return new state(std::vector<std::complex<float>>{d, d + size});

  return new state(std::vector<std::complex<double>>{d, d + size});
}

void __nvqpp_cudaq_state_delete(state *obj) { delete obj; }
}
} // namespace cudaq
