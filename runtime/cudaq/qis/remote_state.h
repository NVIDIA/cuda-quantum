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
  mutable std::vector<cudaq::complex> state;
  mutable bool executed = false;

public:
  template <typename QuantumKernel, typename... Args>
  RemoteSimulationState(QuantumKernel &&kernel, Args &&...args) {
    kernelName = cudaq::getKernelName(kernel);
  }
  void execute() const;
  std::size_t getNumQubits() const override {
    execute();
    return std::log2(state.size());
  }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    const auto &otherState = dynamic_cast<const RemoteSimulationState &>(other);
    // Now submit an overlap computation
    if (!executed && otherState.executed) {
      // submit
      // TODO
    }
    // We've resolved the state (due to other API calls)
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    // FIXME: handle amplitude access for extremely-large state vector (mgpu,
    // mps, etc.)
    execute();
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(
          fmt::format("[RemoteSimulationState] getAmplitude with an invalid "
                      "number of bits in the "
                      "basis state: expected {}, provided {}.",
                      getNumQubits(), basisState.size()));
    if (std::any_of(basisState.begin(), basisState.end(),
                    [](int x) { return x != 0 && x != 1; }))
      throw std::runtime_error("[RemoteSimulationState] getAmplitude with an "
                               "invalid basis state: only "
                               "qubit state (0 or 1) is supported.");

    // Convert the basis state to an index value
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [](std::size_t acc, int bit) { return (acc << 1) + bit; });
    return state[idx];
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error(
          "[RemoteSimulationState] invalid tensor requested.");
    return Tensor{
        reinterpret_cast<void *>(const_cast<cudaq::complex *>(state.data())),
        std::vector<std::size_t>{static_cast<std::size_t>(state.size())},
        getPrecision()};
  }

  /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0)
      throw std::runtime_error(
          "[RemoteSimulationState] invalid tensor requested.");
    if (indices.size() != 1)
      throw std::runtime_error(
          "[RemoteSimulationState] invalid element extraction.");

    return state[indices[0]];
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    throw std::runtime_error("TODO");
  }

  void dump(std::ostream &os) const override {
    execute();
    os << "SV: [";
    for (std::size_t i = 0; const auto &el : state) {
      os << el;
      if (i < (state.size() - 1))
        os << ", ";
      ++i;
    }
    os << "]\n";
  }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override { state.clear(); }
};
} // namespace cudaq