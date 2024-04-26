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

class MPSSimulationState : public cudaq::SimulationState,
                           public cudaq::TensorNetworkState {

public:
  MPSSimulationState(std::unique_ptr<TensorNetState> inState,
                     const std::vector<MPSTensor> &mpsTensors,
                     const std::vector<std::size_t> &auxTensorIds,
                     cutensornetHandle_t cutnHandle);

  MPSSimulationState(const MPSSimulationState &) = delete;
  MPSSimulationState &operator=(const MPSSimulationState &) = delete;
  MPSSimulationState(MPSSimulationState &&) noexcept = default;
  MPSSimulationState &operator=(MPSSimulationState &&) noexcept = default;

  virtual ~MPSSimulationState();

  std::complex<double> overlap(const cudaq::SimulationState &other) override;

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;
  std::size_t getNumQubits() const override;
  void dump(std::ostream &) const override;
  cudaq::SimulationState::precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override;

  std::vector<Tensor> getTensors() const override;

  std::size_t getNumTensors() const override;

  void destroyState() override;

  bool isDeviceData() const override { return true; }

  bool isArrayLike() const override { return false; }

  virtual std::unique_ptr<cudaq::SimulationState>
  createFromSizeAndPtr(std::size_t, void *, std::size_t dataType) override;

  // Note: this API is intended for a simulate-observe-reinit use case on single
  // state. For example, run a circuit, get the state to perform some
  // computation (e.g., overlap, expectation), then reinit the state to continue
  // the simulation. The resulting nvqir::TensorNetState should be able to be
  // fed to the appropriate tensor network based simulator to continue the
  // simulation.
  // In this case (MPS), the initial state is set to the MPS tensor train, which
  // has been factorized when the previous get_state was called (to get a handle
  // to this MPSSimulationState).
  std::unique_ptr<nvqir::TensorNetState> reconstructBackendState() override;
  std::unique_ptr<cudaq::SimulationState> toSimulationState() override;

protected:
  void deallocate();
  std::complex<double>
  computeOverlap(const std::vector<MPSTensor> &m_mpsTensors,
                 const std::vector<MPSTensor> &mpsOtherTensors);

  // The state that this owned.
  cutensornetHandle_t m_cutnHandle;
  std::unique_ptr<TensorNetState> state;
  std::vector<MPSTensor> m_mpsTensors;
  ScratchDeviceMem m_scratchPad;
  std::vector<std::size_t> m_auxTensorIds;
};

} // namespace nvqir