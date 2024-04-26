/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <unordered_map>

#include "cutensornet.h"
#include "tensornet_state.h"
#include "tensornet_utils.h"
#include "timing_utils.h"

#include "common/SimulationState.h"

namespace nvqir {

class TensorNetSimulationState : public cudaq::SimulationState,
                                 public cudaq::TensorNetworkState {

public:
  TensorNetSimulationState(std::unique_ptr<TensorNetState> inState,
                           cutensornetHandle_t cutnHandle);

  TensorNetSimulationState(const TensorNetSimulationState &) = delete;
  TensorNetSimulationState &
  operator=(const TensorNetSimulationState &) = delete;
  TensorNetSimulationState(TensorNetSimulationState &&) noexcept = default;
  TensorNetSimulationState &
  operator=(TensorNetSimulationState &&) noexcept = default;

  virtual ~TensorNetSimulationState();

  std::complex<double> overlap(const cudaq::SimulationState &other) override;

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;
  std::size_t getNumQubits() const override;
  void dump(std::ostream &) const override;
  cudaq::SimulationState::precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  bool isDeviceData() const override { return true; }

  bool isArrayLike() const override { return false; }

  Tensor getTensor(std::size_t tensorIdx = 0) const override;
  virtual std::unique_ptr<cudaq::SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr,
                       std::size_t dataType) override {
    std::vector<std::complex<double>> vec(
        reinterpret_cast<std::complex<double> *>(ptr),
        reinterpret_cast<std::complex<double> *>(ptr) + size);
    auto tensorNetState =
        TensorNetState::createFromStateVector(vec, m_cutnHandle);

    return std::make_unique<TensorNetSimulationState>(std::move(tensorNetState),
                                                      m_cutnHandle);
  }
  /// Get component tensors
  // Note: for the full tensor network state, we return the gate tensors in the
  // full tensor network. The root tensor of this network is not computed and
  // may require lots of memory. Thus, we don't use it as the 'tensor' object.
  std::vector<Tensor> getTensors() const override;

  std::size_t getNumTensors() const override;

  void destroyState() override;
  // Note: this API is intended for a simulate-observe-reinit use case on single
  // state. For example, run a circuit, get the state to perform some
  // computation (e.g., overlap, expectation), then reinit the state to continue
  // the simulation. The resulting nvqir::TensorNetState should be able to be
  // fed to the appropriate tensor network based simulator to continue the
  // simulation.
  std::unique_ptr<nvqir::TensorNetState> reconstructBackendState() override;

  std::unique_ptr<cudaq::SimulationState> toSimulationState() override;

protected:
  std::unique_ptr<TensorNetState> m_state;
  cutensornetHandle_t m_cutnHandle;
  ScratchDeviceMem m_scratchPad;
};
} // namespace nvqir