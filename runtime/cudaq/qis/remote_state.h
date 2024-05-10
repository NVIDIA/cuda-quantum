/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/FmtCore.h"
#include "common/SimulationState.h"
#include "cudaq.h"
#include "cudaq/utils/cudaq_utils.h"
#include <algorithm>
#include <iostream>
#include <numeric>
namespace cudaq {
/// Implementation of `SimulationState` for remote simulator backends.
// The state is represented by a quantum kernel.
// For accessor APIs, we may resolve the state to a state vector by executing
// the kernel on the remote simulator. For overlap API b/w 2 remote states, we
// can send both kernels to the remote backend for execution and compute the
// overlap.
class RemoteSimulationState : public cudaq::SimulationState {
  std::string kernelName;
  mutable std::unique_ptr<cudaq::SimulationState> state;
  mutable std::vector<char> argsBuffer;

public:
  template <typename QuantumKernel, typename... Args>
  RemoteSimulationState(QuantumKernel &&kernel, Args &&...args) {
    kernelName = cudaq::getKernelName(kernel);
    argsBuffer = cudaq::serializeArgs(std::forward<Args>(args)...);
  }

  /// @brief Triggers remote execution to resolve the state data.
  void execute() const;

  std::tuple<std::string, void *, std::size_t> getKernelInfo() const;

  std::size_t getNumQubits() const override {
    execute();
    return std::log2(state->getNumQubits());
  }

  std::complex<double> overlap(const cudaq::SimulationState &other) override;

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    // FIXME: handle amplitude access for extremely-large state vector (mgpu,
    // mps, etc.)
    // i.e., needs to forward getAmplitude as a REST API call.
    execute();
    return state->getAmplitude(basisState);
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    execute();
    return state->getTensor(tensorIdx);
  }

  /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    execute();
    return state->operator()(tensorIdx, indices);
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    throw std::runtime_error("Unsupported");
  }

  void dump(std::ostream &os) const override {
    execute();
    state->dump(os);
  }

  precision getPrecision() const override {
    execute();
    return state->getPrecision(); 
  }

  void destroyState() override { state.reset(); }
};
} // namespace cudaq