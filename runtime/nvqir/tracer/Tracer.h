/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"

using namespace cudaq;

namespace nvqir {

class Tracer : public nvqir::CircuitSimulatorBase<double> {
protected:
  size_t num_qubits;
  std::map<std::string, size_t> gate_counts;

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override {
    sampleQubits.emplace_back(num_qubits++);
    executionContext->resourceCounts.addQubit();
  }

  void applyGate(const GateApplicationTask &task) override {
    cudaq::info("Applying {} with {} controls", task.operationName, task.controls.size());
    auto gate = resource_counts::GateData{ task.operationName, task.controls.size() };
    executionContext->resourceCounts.append(gate);
  }

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t index) override {
    assert(executionContext->choice);
    auto measure = executionContext->choice();
    cudaq::info("Measure of {} returned {}", index, measure);
    return measure;
  }

public:
  Tracer() {
    // Populate the correct name so it is printed correctly during
    // deconstructor.
    summaryData.name = name();
  }
  virtual ~Tracer() = default;

  bool canHandleObserve() override {
    return false;
  }

  /// @brief Reset the qubit
  /// @param index 0-based index of qubit to reset
  void resetQubit(const std::size_t index) override {}

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots) override {
    throw std::runtime_error("Can't sample from resource counter simulator!");
  }

  std::string name() const override { return "tracer"; }

  CircuitSimulator *clone() override {
    // TODO: probably fine
    return this;
  };

  void deallocateStateImpl() override {}

  // TODO
  void setToZeroState() override {}
};

} // namespace nvqir
