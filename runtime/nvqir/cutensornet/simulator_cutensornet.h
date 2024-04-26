/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CircuitSimulator.h"
#include "cutensornet.h"
#include "tensornet_state.h"

namespace nvqir {
/// @brief Base class of `cutensornet` simulator backends
class SimulatorTensorNetBase : public nvqir::CircuitSimulatorBase<double> {

public:
  SimulatorTensorNetBase();
  SimulatorTensorNetBase(const SimulatorTensorNetBase &another) = delete;
  SimulatorTensorNetBase &
  operator=(const SimulatorTensorNetBase &another) = delete;
  SimulatorTensorNetBase(SimulatorTensorNetBase &&another) noexcept = delete;
  SimulatorTensorNetBase &
  operator=(SimulatorTensorNetBase &&another) noexcept = delete;

  virtual ~SimulatorTensorNetBase();

  /// @brief Apply quantum gate
  void applyGate(const GateApplicationTask &task) override;

  // Override base calculateStateDim (we don't instantiate full state vector in
  // the tensornet backend). When the user want to retrieve the state vector, we
  // check if it is feasible to do so.
  virtual std::size_t calculateStateDim(const std::size_t numQubits) override {
    return numQubits;
  }

  /// @brief Reset the state of a given qubit to zero
  virtual void resetQubit(const std::size_t qubitIdx) override;

  /// @brief Device synchronization
  virtual void synchronize() override;

  /// @brief Perform a measurement on a given qubit
  virtual bool measureQubit(const std::size_t qubitIdx) override;

  /// @brief Sample a subset of qubits
  virtual cudaq::ExecutionResult
  sample(const std::vector<std::size_t> &measuredBits,
         const int shots) override;

  /// @brief Evaluate the expectation value of a given observable
  virtual cudaq::observe_result observe(const cudaq::spin_op &op) override;

  /// @brief Add qubits to the underlying quantum state
  virtual void addQubitsToState(std::size_t count,
                                const void *state = nullptr) override;

  /// Clone API
  virtual nvqir::CircuitSimulator *clone() override;

  virtual std::unique_ptr<cudaq::SimulationState>
  getSimulationState() override {
    throw std::runtime_error("[tensornet] getSimulationState not implemented");
    return nullptr;
  }

protected:
  // Sub-type need to implement
  virtual void prepareQubitTensorState() = 0;

  /// @brief Grow the qubit register by one qubit
  virtual void addQubitToState() override;

  /// @brief Destroy the entire qubit register
  virtual void deallocateStateImpl() override;

  /// @brief Reset all qubits to zero
  virtual void setToZeroState() override;

  /// @brief Query if direct expectation value calculation is enabled
  virtual bool canHandleObserve() override;

protected:
  cutensornetHandle_t m_cutnHandle;
  std::unique_ptr<TensorNetState> m_state;
  std::unordered_map<std::string, void *> m_gateDeviceMemCache;
};

} // end namespace nvqir
