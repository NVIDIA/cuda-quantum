/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/MeasureCounts.h"
#include "cudaq/qis/execution_manager.h"

namespace {
class RemoteRestExecutionManager : public cudaq::ExecutionManager {
public:
  /// Return the next available qudit index
  virtual std::size_t getAvailableIndex(std::size_t quditLevels) override {
    static std::size_t counter = 0;
    return counter++;
  }

  /// QuditInfo has been deallocated, return the qudit / id to the pool of
  /// qudits.
  virtual void returnQudit(const cudaq::QuditInfo &q) override {}

  /// Provide an ExecutionContext for the current cudaq kernel
  virtual void setExecutionContext(cudaq::ExecutionContext *ctx) override {}

  /// Reset the execution context
  virtual void resetExecutionContext() override {}

  /// Apply the quantum instruction with the given name, on the provided
  /// target qudits. Supports input of control qudits and rotational parameters.
  /// Can also optionally take a spin_op as input to affect a general
  /// Pauli rotation.
  virtual void apply(const std::string_view gateName,
                     const std::vector<double> &params,
                     const std::vector<cudaq::QuditInfo> &controls,
                     const std::vector<cudaq::QuditInfo> &targets,
                     bool isAdjoint = false,
                     const cudaq::spin_op op = cudaq::spin_op()) override {}

  /// Reset the qubit to the |0> state
  virtual void reset(const cudaq::QuditInfo &target) override {}

  /// Begin an region of code where all operations will be adjoint-ed
  virtual void startAdjointRegion() override {}
  /// End the adjoint region
  virtual void endAdjointRegion() override {}

  /// Start a region of code where all operations will be
  /// controlled on the given qudits.
  virtual void
  startCtrlRegion(const std::vector<std::size_t> &control_qubits) override {}
  /// End the control region
  virtual void endCtrlRegion(std::size_t n_controls) override {}

  /// Measure the qudit and return the observed state (0,1,2,3,...)
  /// e.g. for qubits, this can return 0 or 1;
  virtual int measure(const cudaq::QuditInfo &target) override { return 0; }

  /// Measure the current state in the given Pauli basis, return
  /// the expectation value <term>.
  virtual cudaq::SpinMeasureResult measure(cudaq::spin_op &op) override {
    return cudaq::SpinMeasureResult(0.0, {});
  }

  /// Synchronize - run all queue-ed instructions
  virtual void synchronize() override {}
};
} // namespace

CUDAQ_REGISTER_EXECUTION_MANAGER(RemoteRestExecutionManager)
