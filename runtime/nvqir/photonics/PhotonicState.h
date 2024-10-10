/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "PhotonicGates.h"

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/MeasureCounts.h"
#include "cudaq/host_config.h"

#include <iostream>
#include <set>
#include <span>

namespace cudaq {

/// @brief PhotonicState provides an implementation of `SimulationState` that
/// encapsulates the state data for the Photonic Circuit Simulators.
struct PhotonicState : public SimulationState {
protected:
  virtual std::unique_ptr<PhotonicState>
  createPSFromSizeAndPtr(std::size_t, void *, std::size_t dataType) = 0;

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    throw std::runtime_error(
        "createFromSizeAndPtr not available for this photonic simulator "
        "backend.");
  }

public:
  virtual std::unique_ptr<PhotonicState>
  createPSFromData(const state_data &data) {
    if (std::holds_alternative<TensorStateData>(data)) {
      if (isArrayLike())
        throw std::runtime_error(
            "Cannot initialize state vector/density matrix state by matrix "
            "product state tensors. Please use tensor network simulator "
            "backends.");
      auto &dataCasted = std::get<TensorStateData>(data);
      return createPSFromSizeAndPtr(
          dataCasted.size(),
          const_cast<TensorStateData::value_type *>(dataCasted.data()),
          data.index());
    }
    // Flat array state data
    // Check the precision first. Get the size and data pointer from the input
    // data.
    if (getPrecision() == precision::fp32) {
      auto [size, ptr] = getSizeAndPtr<float>(data);
      return createPSFromSizeAndPtr(size, ptr, data.index());
    }

    auto [size, ptr] = getSizeAndPtr(data);
    return createPSFromSizeAndPtr(size, ptr, data.index());
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    throw std::runtime_error(
        "getTensor not available for this photonic simulator backend.");
  }

  std::vector<Tensor> getTensors() const override {
    throw std::runtime_error(
        "getTensors not available for this photonic simulator backend.");
  }

  std::size_t getNumTensors() const override {
    throw std::runtime_error(
        "getNumTensors not available for this photonic simulator backend.");
  }

  std::size_t getNumQubits() const override {
    throw std::runtime_error(
        "getNumQubits not available for this photonic simulator backend.");
  }

  virtual std::size_t getNumQudits() const = 0;

  virtual std::complex<double> overlap(const SimulationState &other) override {
    throw std::runtime_error(
        "overlap not available for this photonic simulator backend.");
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    throw std::runtime_error(
        "getAmplitude not available for this photonic simulator backend.");
  }

  void dump(std::ostream &os) const override {
    throw std::runtime_error(
        "dump not available for this photonic simulator backend.");
  }

  precision getPrecision() const override {
    throw std::runtime_error(
        "getPrecision not available for this photonic simulator backend.");
  }

  void destroyState() override {
    throw std::runtime_error(
        "destroyState not available for this photonic simulator backend.");
  }
}; // PhotonicState

} // namespace cudaq
