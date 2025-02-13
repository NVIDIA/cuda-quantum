/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qis/state.h"
#include <cassert>
#include <memory>

/// @cond DO_NOT_DOCUMENT
/// @brief Fake simulation state to use in tests.
class FakeDeviceState : public cudaq::SimulationState {
private:
  std::string kernelName;
  std::vector<void *> args;
  std::size_t size = 0;
  void *data = 0;

public:
  virtual std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *data,
                       std::size_t dataType) override {
    throw std::runtime_error("Not implemented");
  }

  FakeDeviceState() = default;
  FakeDeviceState(std::size_t size, void *data) : size(size), data(data) {}
  FakeDeviceState(const std::string &kernelName, const std::vector<void *> args)
      : kernelName(kernelName), args(args) {}
  FakeDeviceState(const FakeDeviceState &other)
      : kernelName(other.kernelName), args(other.args) {}

  virtual std::unique_ptr<cudaq::SimulationState>
  createFromData(const cudaq::state_data &data) override {
    throw std::runtime_error("Not implemented");
  }

  virtual bool hasData() const override { return data != nullptr; }

  virtual std::optional<std::pair<std::string, std::vector<void *>>>
  getKernelInfo() const override {
    return std::make_pair(kernelName, args);
  }

  virtual Tensor getTensor(std::size_t tensorIdx = 0) const override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::vector<Tensor> getTensors() const override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::size_t getNumTensors() const override {
    if (hasData())
      return 1;
    throw std::runtime_error("Not implemented");
  }

  virtual std::size_t getNumQubits() const override {
    if (hasData())
      return std::countr_zero(size);
    throw std::runtime_error("Not implemented");
  }

  virtual std::complex<double> overlap(const SimulationState &other) override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::vector<std::complex<double>>
  getAmplitudes(const std::vector<std::vector<int>> &basisStates) override {
    throw std::runtime_error("Not implemented");
  }

  virtual void dump(std::ostream &os) const override {
    throw std::runtime_error("Not implemented");
  }

  virtual precision getPrecision() const override {
    if (hasData())
      return cudaq::SimulationState::precision::fp64;
    throw std::runtime_error("Not implemented");
  }

  virtual void destroyState() override {}

  virtual std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (hasData()) {
      assert(tensorIdx == 0);
      assert(indices.size() == 1);
      return *(static_cast<std::complex<double> *>(data) + indices[0]);
    }
    throw std::runtime_error("Not implemented");
  }

  virtual std::size_t getNumElements() const override {
    if (hasData())
      return size;
    throw std::runtime_error("Not implemented");
  }

  virtual bool isDeviceData() const override { return false; }

  virtual bool isArrayLike() const override { return true; }

  virtual void toHost(std::complex<double> *clientAllocatedData,
                      std::size_t numElements) const override {
    throw std::runtime_error("Not implemented");
  }

  virtual void toHost(std::complex<float> *clientAllocatedData,
                      std::size_t numElements) const override {
    throw std::runtime_error("Not implemented");
  }

  virtual ~FakeDeviceState() override {}
};
/// @endcond
