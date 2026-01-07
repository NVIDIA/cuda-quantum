/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Resources.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"

namespace nvqir {

class ResourceCounter : public nvqir::CircuitSimulatorBase<double> {
protected:
  cudaq::Resources resourceCounts;
  std::function<bool()> choice;

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override { resourceCounts.addQubit(); }

  void applyGate(const GateApplicationTask &task) override {
    CUDAQ_INFO("Applying {} with {} controls", task.operationName,
               task.controls.size());
    resourceCounts.appendInstruction(task.operationName, task.controls.size());
  }

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t index) override {
    assert(choice);
    auto measure = choice();
    CUDAQ_INFO("Measure of {} returned {}", index, measure);
    return measure;
  }

public:
  ResourceCounter() {
    // Populate the correct name so it is printed correctly during
    // deconstructor.
    summaryData.name = name();
  }
  virtual ~ResourceCounter() = default;

  bool canHandleObserve() override { return false; }

  std::size_t calculateStateDim(const std::size_t numQubits) override {
    return numQubits;
  }

  /// @brief Reset the qubit
  /// @param index 0-based index of qubit to reset
  void resetQubit(const std::size_t index) override {
    resourceCounts.appendInstruction("reset", 0);
  }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots) override {
    throw std::runtime_error("Can't sample from resource counter simulator!");
  }

  std::string name() const override { return "resourcecounter"; }

  CircuitSimulator *clone() override { return this; };

  void deallocateStateImpl() override {}

  void setToZeroState() override { resourceCounts.clear(); }

  void setExecutionContext(cudaq::ExecutionContext *context) override {
    if (context->name != "resource-count")
      throw std::runtime_error(
          "Illegal use of resource counter simulator! (Did you attempt to run "
          "a kernel inside of a choice function?)");
    this->CircuitSimulatorBase::setExecutionContext(context);
  }

  cudaq::Resources *getResourceCounts() { return &this->resourceCounts; }

  void setChoiceFunction(std::function<bool()> choice) {
    assert(choice);
    this->choice = choice;
  }
};

ResourceCounter *getResourceCounterSimulator();

void setChoiceFunction(std::function<bool()> choice);

cudaq::Resources *getResourceCounts();

} // namespace nvqir
