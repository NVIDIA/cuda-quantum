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

class TensorNetSimulationState : public cudaq::SimulationState {

public:
  TensorNetSimulationState(std::unique_ptr<TensorNetState> inState);

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
  void dump(std::ostream &) const override {}
  cudaq::SimulationState::precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override;

  std::vector<Tensor> getTensors() const override;

  std::size_t getNumTensors() const override;

  void destroyState() override;

protected:
  std::unique_ptr<TensorNetState> m_state;
  ScratchDeviceMem m_scratchPad;
};

} // namespace nvqir