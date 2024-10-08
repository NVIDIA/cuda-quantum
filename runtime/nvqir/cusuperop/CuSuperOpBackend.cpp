/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CircuitSimulator.h"
#include "CuSuperOpState.h"
namespace {

class CuSuperOpSim : public nvqir::CircuitSimulatorBase<double> {
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
  CuSuperOpSim() {}

  /// The destructor
  virtual ~CuSuperOpSim() = default;

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    return std::make_unique<cudaq::CuSuperOpState>();
  }

  void addQubitToState() override {
    throw std::runtime_error(
        "[nvidia-dynamics] Quantum gate simulation is not supported.");
  }
  void deallocateStateImpl() override {
    throw std::runtime_error(
        "[nvidia-dynamics] Quantum gate simulation is not supported.");
  }
  bool measureQubit(const std::size_t qubitIdx) override {
    throw std::runtime_error("[nvidia-dynamics] Quantum gate simulation is not "
                             "supported.");
    return false;
  }
  void applyGate(const GateApplicationTask &task) override {
    throw std::runtime_error(
        "[nvidia-dynamics] Quantum gate simulation is not supported.");
  }
  void setToZeroState() override {
    throw std::runtime_error(
        "[nvidia-dynamics] Quantum gate simulation is not supported.");
  }
  void resetQubit(const std::size_t qubitIdx) override {
    throw std::runtime_error(
        "[nvidia-dynamics] Quantum gate simulation is not supported.");
  }
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubitIdxs,
                                const int shots) override {
    throw std::runtime_error("[nvidia-dynamics] Quantum gate simulation is not "
                             "supported.");
    return cudaq::ExecutionResult();
  }
  std::string name() const override { return "nvidia-dynamics"; }
  NVQIR_SIMULATOR_CLONE_IMPL(CuSuperOpSim)
};
} // namespace

NVQIR_REGISTER_SIMULATOR(CuSuperOpSim, nvidia_dynamics)
