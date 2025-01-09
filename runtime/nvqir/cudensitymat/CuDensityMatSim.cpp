/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CircuitSimulator.h"
#include "CuDensityMatState.h"
namespace {

class CuDensityMatSim : public nvqir::CircuitSimulatorBase<double> {
protected:
  using ScalarType = double;
  using DataType = std::complex<double>;
  using DataVector = std::vector<DataType>;

  using nvqir::CircuitSimulatorBase<ScalarType>::tracker;
  using nvqir::CircuitSimulatorBase<ScalarType>::nQubitsAllocated;
  using nvqir::CircuitSimulatorBase<ScalarType>::stateDimension;
  using nvqir::CircuitSimulatorBase<ScalarType>::calculateStateDim;
  using nvqir::CircuitSimulatorBase<ScalarType>::executionContext;
  using nvqir::CircuitSimulatorBase<ScalarType>::gateToString;
  using nvqir::CircuitSimulatorBase<ScalarType>::x;
  using nvqir::CircuitSimulatorBase<ScalarType>::flushGateQueue;
  using nvqir::CircuitSimulatorBase<ScalarType>::previousStateDimension;
  using nvqir::CircuitSimulatorBase<ScalarType>::shouldObserveFromSampling;
  using nvqir::CircuitSimulatorBase<ScalarType>::summaryData;

public:
  /// @brief The constructor
  CuDensityMatSim() {}

  /// The destructor
  virtual ~CuDensityMatSim() = default;

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    return std::make_unique<cudaq::CuDensityMatState>();
  }

  void addQubitToState() override {
    throw std::runtime_error(
        "[dynamics target] Quantum gate simulation is not supported.");
  }
  void deallocateStateImpl() override {
    throw std::runtime_error(
        "[dynamics target] Quantum gate simulation is not supported.");
  }
  bool measureQubit(const std::size_t qubitIdx) override {
    throw std::runtime_error("[dynamics target] Quantum gate simulation is not "
                             "supported.");
    return false;
  }
  void applyGate(const GateApplicationTask &task) override {
    throw std::runtime_error(
        "[dynamics target] Quantum gate simulation is not supported.");
  }
  void setToZeroState() override {
    throw std::runtime_error(
        "[dynamics target] Quantum gate simulation is not supported.");
  }
  void resetQubit(const std::size_t qubitIdx) override {
    throw std::runtime_error(
        "[dynamics target] Quantum gate simulation is not supported.");
  }
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubitIdxs,
                                const int shots) override {
    throw std::runtime_error("[dynamics target] Quantum gate simulation is not "
                             "supported.");
    return cudaq::ExecutionResult();
  }
  std::string name() const override { return "dynamics"; }
  NVQIR_SIMULATOR_CLONE_IMPL(CuDensityMatSim)
};
} // namespace

NVQIR_REGISTER_SIMULATOR(CuDensityMatSim, dynamics)
