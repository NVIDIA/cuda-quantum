/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "state.h"
#include "common/EigenDense.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "cudaq/simulators.h"
#include <iostream>

namespace cudaq {

std::mutex deleteStateMutex;

state state::from_data(const state_data &data) {
  auto *simulator = cudaq::get_simulator();
  if (!simulator)
    throw std::runtime_error(
        "[state::from_data] Could not find valid simulator backend.");

  return state(simulator->createStateFromData(data).release());
}

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
                  std::size_t tensorIdx) {
  std::vector<std::size_t> idxVec(indices.begin(), indices.end());
  return (*internal)(tensorIdx, idxVec);
}

std::complex<double> state::operator[](std::size_t idx) {
  std::size_t numQubits = internal->getNumQubits();
  if (!internal->isArrayLike()) {
    // Use amplitude accessor if linear indexing is not supported, e.g., tensor
    // network state.
    std::vector<int> basisState(numQubits, 0);
    for (std::size_t i = 0; i < numQubits; ++i) {
      if (idx & (1ULL << i))
        basisState[(numQubits - 1) - i] = 1;
    }
    return internal->getAmplitude(basisState);
  }

  std::size_t newIdx = 0;
  for (std::size_t i = 0; i < numQubits; ++i)
    if (idx & (1ULL << i))
      newIdx |= (1ULL << ((numQubits - 1) - i));
  return operator()({newIdx}, 0);
}

std::complex<double> state::operator()(std::size_t idx, std::size_t jdx) {
  return operator()({idx, jdx}, 0);
}

std::complex<double> state::overlap(const state &other) {
  return internal->overlap(*other.internal.get());
}

std::complex<double> state::amplitude(const std::vector<int> &basisState) {
  return internal->getAmplitude(basisState);
}

state &state::operator=(state other) {
  // Copy and swap idiom
  std::swap(internal, other.internal);
  return *this;
}

state::~state() {
  // Make sure destroying the state is thread safe.
  std::lock_guard<std::mutex> lock(deleteStateMutex);

  // Current use count is 1, so the
  // shared_ptr is about to go out of scope,
  // there are no users. Delete the state data.
  if (internal.use_count() == 1)
    internal->destroyState();
}

extern "C" {
std::int64_t __nvqpp_cudaq_state_numberOfQubits(state *obj) {
  // throw std::runtime_error(
  //     "not yet implemented: getting number of qubits from state");
  return obj->get_num_qubits();
}

void *__nvqpp_cudaq_state_vectorData(state *obj) {
  // throw std::runtime_error(
  //     "not yet implemented: getting vector data from state");

  auto num = obj->get_num_qubits();
  std::cout << "Num of qubits from state: " << num << std::endl;
  std::cout << "State is on gpu: " << obj->is_on_gpu() << std::endl;

  void *dataPtr = nullptr;
  auto stateVector = obj->get_tensor();
  auto precision = obj->get_precision();
  if (obj->is_on_gpu()) {
    auto numElements = stateVector.get_num_elements();
    if (precision == SimulationState::precision::fp32) {
      auto *hostData = new std::complex<float>[numElements];
      obj->to_host(hostData, numElements);
      dataPtr = reinterpret_cast<void *>(hostData);
    } else {
      auto *hostData = new std::complex<double>[numElements];
      obj->to_host(hostData, numElements);
      dataPtr = reinterpret_cast<void *>(hostData);
    }
  } else {
    dataPtr = stateVector.data;
  }
  // {
  //   auto data = reinterpret_cast<std::complex<double>*>(dataPtr);
  //   auto vec = std::vector<std::complex<double>>(data, data + num);
  //   std::cout << "Data from state: ";
  //   for (auto& e :vec) {
  //     std::cout << e << ", ";
  //   }
  //   std::cout << std::endl;
  // }
  return dataPtr;
}
}

} // namespace cudaq
