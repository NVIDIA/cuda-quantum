/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "simulator_cutensornet.h"
#include "tn_simulation_state.h"

// Forward declaration
extern "C" nvqir::CircuitSimulator *getCircuitSimulator_tensornet();

namespace nvqir {
class SimulatorTensorNet : public SimulatorTensorNetBase {
public:
  SimulatorTensorNet() : SimulatorTensorNetBase() {
    // tensornet backend supports distributed tensor network contraction,
    // i.e., distributing tensor network contraction across multiple
    // GPUs/processes.
    //
    // Note: this requires CUTENSORNET_COMM_LIB as described in
    // the Getting Started section of the cuTensorNet library documentation
    // (Installation and Compilation).
    if (cudaq::mpi::is_initialized()) {
      initCuTensornetComm(m_cutnHandle);
      m_cutnMpiInitialized = true;
    }
  }
  // Nothing to do for state preparation
  virtual void prepareQubitTensorState() override {}
  virtual std::string name() const override { return "tensornet"; }
  CircuitSimulator *clone() override {
    thread_local static auto simulator = std::make_unique<SimulatorTensorNet>();
    return simulator.get();
  }
  // Add a hook to reset the cutensornet MPI Comm before MPI finalization
  // to make sure we have a clean shutdown.
  virtual void tearDownBeforeMPIFinalize() override {
    if (cudaq::mpi::is_initialized()) {
      resetCuTensornetComm(m_cutnHandle);
      m_cutnMpiInitialized = false;
    }
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    LOG_API_TIME();
    return std::make_unique<TensorNetSimulationState>(std::move(m_state),
                                                      m_cutnHandle);
  }

private:
  friend nvqir::CircuitSimulator * ::getCircuitSimulator_tensornet();
  /// @brief Has cuTensorNet MPI been initialized?
  bool m_cutnMpiInitialized = false;
};
} // namespace nvqir

/// Register this Simulator class with NVQIR under name "tensornet"
extern "C" {
nvqir::CircuitSimulator *getCircuitSimulator_tensornet() {
  thread_local static auto simulator =
      std::make_unique<nvqir::SimulatorTensorNet>();
  // Handle multiple runtime __nvqir__setCircuitSimulator calls before/after MPI
  // initialization. If the static simulator instance was created before MPI
  // initialization, it needs to be reset to support MPI if needed.
  if (cudaq::mpi::is_initialized() && !simulator->m_cutnMpiInitialized) {
    // Reset the static instance to pick up MPI.
    simulator.reset(new nvqir::SimulatorTensorNet());
  }
  return simulator.get();
}
nvqir::CircuitSimulator *getCircuitSimulator() {
  return getCircuitSimulator_tensornet();
}
}
