/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "common/SimulationState.h"
#include "nvqir/CircuitSimulator.h"

/// @brief Macro for mock function bodies that log and throw "Not implemented".
/// Usage: bool myFunc(int x) override { MOCK_NOT_IMPLEMENTED("Class::myFunc");
/// }
#define MOCK_NOT_IMPLEMENTED(funcName)                                         \
  CUDAQ_INFO(funcName);                                                        \
  throw std::runtime_error("Not implemented")

/// @brief A minimal mock SimulationState that only tracks the number of qubits.
class MockSimulationState : public cudaq::SimulationState {
  std::size_t numQubits;

public:
  MockSimulationState(std::size_t nQubits) : numQubits(nQubits) {
    CUDAQ_INFO("MockSimulationState::MockSimulationState");
  }

  std::size_t getNumQubits() const override {
    CUDAQ_INFO("MockSimulationState::getNumQubits");
    return numQubits;
  }

  // Required pure virtual methods
  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr,
                       std::size_t dataType) override {
    MOCK_NOT_IMPLEMENTED("MockSimulationState::createFromSizeAndPtr");
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    MOCK_NOT_IMPLEMENTED("MockSimulationState::getTensor");
  }

  std::vector<Tensor> getTensors() const override {
    MOCK_NOT_IMPLEMENTED("MockSimulationState::getTensors");
  }

  std::size_t getNumTensors() const override {
    MOCK_NOT_IMPLEMENTED("MockSimulationState::getNumTensors");
  }

  std::complex<double> overlap(const SimulationState &other) override {
    MOCK_NOT_IMPLEMENTED("MockSimulationState::overlap");
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    MOCK_NOT_IMPLEMENTED("MockSimulationState::getAmplitude");
  }

  void dump(std::ostream &os) const override {
    MOCK_NOT_IMPLEMENTED("MockSimulationState::dump");
  }

  precision getPrecision() const override {
    MOCK_NOT_IMPLEMENTED("MockSimulationState::getPrecision");
  }

  void destroyState() override {
    CUDAQ_INFO("MockSimulationState::destroyState");
  }
};

/// @brief A minimal mock CircuitSimulator that tracks qubit allocation.
/// This is used to test the base class behavior without needing a full
/// simulator implementation.
class MockCircuitSimulator : public nvqir::CircuitSimulatorBase<double> {
private:
  std::size_t mockStateNumQubits = 0;

protected:
  void addQubitToState() override {
    CUDAQ_INFO("MockCircuitSimulator::addQubitToState");
    mockStateNumQubits++;
  }

  void addQubitsToState(std::size_t count, const void *state) override {
    CUDAQ_INFO("MockCircuitSimulator::addQubitsToState(count, state)");
    mockStateNumQubits += count;
  }

  void addQubitsToState(const cudaq::SimulationState &state) override {
    CUDAQ_INFO("MockCircuitSimulator::addQubitsToState(SimulationState)");
    // This appends the state's qubits to our current state
    mockStateNumQubits += state.getNumQubits();
  }

  void deallocateStateImpl() override {
    CUDAQ_INFO("MockCircuitSimulator::deallocateStateImpl");
    mockStateNumQubits = 0;
  }

  bool measureQubit(const std::size_t qubitIdx) override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::measureQubit");
  }

  void setToZeroState() override {
    CUDAQ_INFO("MockCircuitSimulator::setToZeroState");
    // Reset state to |0...0> but keep the same number of qubits
    // (this is what happens in batch mode between iterations)
  }

  void applyGate(const GateApplicationTask &task) override {
    CUDAQ_INFO("MockCircuitSimulator::applyGate");
  }

public:
  MockCircuitSimulator() = default;

  std::string name() const override {
    CUDAQ_INFO("MockCircuitSimulator::name");
    return "mock";
  }

  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubitIdxs,
                                const int shots) override {
    CUDAQ_INFO("MockCircuitSimulator::sample");
    return {};
  }

  std::size_t getMockStateNumQubits() const {
    CUDAQ_INFO("MockCircuitSimulator::getMockStateNumQubits");
    return mockStateNumQubits;
  }

  /// @brief Get the nQubitsAllocated from the base class for verification.
  std::size_t getNumQubitsAllocated() const {
    CUDAQ_INFO("MockCircuitSimulator::getNumQubitsAllocated");
    return nQubitsAllocated;
  }

  // Required pure virtual methods
  nvqir::CircuitSimulator *clone() override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::clone");
  }

  bool isSinglePrecision() const override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::isSinglePrecision");
  }

  std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &data) override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::createStateFromData");
  }

  void setNoiseModel(cudaq::noise_model &noise) override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::setNoiseModel");
  }

  cudaq::observe_result observe(const cudaq::spin_op &term) override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::observe");
  }

  bool mz(const std::size_t qubitIdx) override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::mz(qubitIdx)");
  }

  bool mz(const std::size_t qubitIdx,
          const std::string &registerName) override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::mz(qubitIdx, registerName)");
  }

  void measureSpinOp(const cudaq::spin_op &op) override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::measureSpinOp");
  }

  void resetQubit(const std::size_t qubitIdx) override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::resetQubit");
  }

  void applyCustomOperation(const std::vector<std::complex<double>> &matrix,
                            const std::vector<std::size_t> &controls,
                            const std::vector<std::size_t> &targets,
                            const std::string_view customUnitaryName) override {
    MOCK_NOT_IMPLEMENTED("MockCircuitSimulator::applyCustomOperation");
  }
};
