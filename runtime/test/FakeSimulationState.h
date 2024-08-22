/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qis/state.h"
#include <cassert>
#include <memory>

/// @brief Fake simulation state to use in tests.
class FakeSimulationState : public cudaq::SimulationState {
private:
  std::size_t size = 0;
  void *data = 0;

public:
  virtual std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *data,
                       std::size_t dataType) override {
    std::runtime_error("Not implemented");
    return std::make_unique<FakeSimulationState>(size, data);
  }

  FakeSimulationState() = default;
  FakeSimulationState(std::size_t size, void *data) : size(size), data(data) {}

  virtual std::unique_ptr<cudaq::SimulationState>
  createFromData(const cudaq::state_data &data) override {
    std::runtime_error("Not implemented");
    return std::make_unique<FakeSimulationState>(0, nullptr);
  }

  virtual Tensor getTensor(std::size_t tensorIdx = 0) const override {
    std::runtime_error("Not implemented");
    return Tensor();
  }

  virtual std::vector<Tensor> getTensors() const override {
    std::runtime_error("Not implemented");
    return std::vector<Tensor>();
  }

  virtual std::size_t getNumTensors() const override { return 1; }

  virtual std::size_t getNumQubits() const override {
    return std::countr_zero(size);
  }

  virtual std::complex<double> overlap(const SimulationState &other) override {
    std::runtime_error("Not implemented");
    return 0;
  }

  virtual std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    std::runtime_error("Not implemented");
    return 0;
  }

  virtual std::vector<std::complex<double>>
  getAmplitudes(const std::vector<std::vector<int>> &basisStates) override {
    std::runtime_error("Not implemented");
    return {0};
  }

  virtual void dump(std::ostream &os) const override {
    std::runtime_error("Not implemented");
  }

  virtual precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  virtual void destroyState() override {
    std::runtime_error("Not implemented");
  }

  virtual std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    assert(tensorIdx == 0);
    assert(indices.size() == 1);
    return *(static_cast<std::complex<double> *>(data) + indices[0]);
  }

  virtual std::size_t getNumElements() const override { return size; }

  virtual bool isDeviceData() const override { return false; }

  virtual bool isArrayLike() const override { return true; }

  virtual void toHost(std::complex<double> *clientAllocatedData,
                      std::size_t numElements) const override {
    throw std::runtime_error(
        "SimulationState::toHost complex128 not implemented.");
  }

  virtual void toHost(std::complex<float> *clientAllocatedData,
                      std::size_t numElements) const override {
    throw std::runtime_error(
        "SimulationState::toHost complex64 not implemented.");
  }

  virtual ~FakeSimulationState() {}
};
