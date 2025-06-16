/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ResourceCounts.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"

namespace nvqir {

class ResourceCounter : public nvqir::CircuitSimulatorBase<double> {
protected:
  cudaq::resource_counts resourceCounts;
  std::function<bool()> choice;

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override {
    // executionContext->resourceCounts.addQubit();
    resourceCounts.addQubit();
  }

  void applyGate(const GateApplicationTask &task) override {
    cudaq::info("Applying {} with {} controls", task.operationName,
                task.controls.size());
    auto gate = cudaq::resource_counts::GateData{task.operationName,
                                                 task.controls.size()};
    // executionContext->resourceCounts.append(gate);
    resourceCounts.append(gate);
  }

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t index) override {
    // assert(executionContext->choice);
    // auto measure = executionContext->choice();
    assert(choice);
    auto measure = choice();
    cudaq::info("Measure of {} returned {}", index, measure);
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

  /// @brief Reset the qubit
  /// @param index 0-based index of qubit to reset
  void resetQubit(const std::size_t index) override {}

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
    if (context->name != "resourcecount")
      throw std::runtime_error(
          "Illegal use of resource counter simulator! (Did you attempt to run "
          "a kernel inside of a choice function?)");
    this->CircuitSimulatorBase::setExecutionContext(context);
  }

  cudaq::resource_counts *getResourceCounts() { return &resourceCounts; }

  void setChoiceFunction(std::function<bool()> choice) {
    assert(choice);
    this->choice = choice;
  }
};

} // namespace nvqir
