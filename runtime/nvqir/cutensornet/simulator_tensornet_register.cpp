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
                                                      scratchPad, m_cutnHandle);
  }

  void addQubitsToState(std::size_t numQubits, const void *ptr) override {
    LOG_API_TIME();
    if (!m_state) {
      if (!ptr) {
        m_state = std::make_unique<TensorNetState>(numQubits, scratchPad,
                                                   m_cutnHandle);
      } else {
        auto *casted =
            reinterpret_cast<std::complex<double> *>(const_cast<void *>(ptr));
        std::span<std::complex<double>> stateVec(casted, 1ULL << numQubits);
        m_state = TensorNetState::createFromStateVector(stateVec, scratchPad,
                                                        m_cutnHandle);
      }
    } else {
      if (!ptr) {
        m_state->addQubits(numQubits);
      } else {
        auto *casted =
            reinterpret_cast<std::complex<double> *>(const_cast<void *>(ptr));
        std::span<std::complex<double>> stateVec(casted, 1ULL << numQubits);
        m_state->addQubits(stateVec);
      }
    }
  }

  virtual void
  addQubitsToState(const cudaq::SimulationState &in_state) override {
    LOG_API_TIME();
    const TensorNetSimulationState *const casted =
        dynamic_cast<const TensorNetSimulationState *>(&in_state);
    if (!casted)
      throw std::invalid_argument(
          "[Tensornet simulator] Incompatible state input");
    if (!m_state) {
      m_state = TensorNetState::createFromOpTensors(in_state.getNumQubits(),
                                                    casted->getAppliedTensors(),
                                                    scratchPad, m_cutnHandle);
    } else {
      // Expand an existing state:
      //  (1) Create a blank tensor network with combined number of qubits
      //  (2) Add back the gate tensors of the original tensor network (first
      // half of the register)
      //  (3) Add gate tensors of the incoming init state
      // after remapping the leg indices, i.e., shifting the leg id by the
      // original size.
      const auto currentSize = m_state->getNumQubits();
      // Add qubits in zero state
      m_state->addQubits(in_state.getNumQubits());
      auto mapQubitIdxs = [currentSize](const std::vector<int32_t> &idxs) {
        std::vector<int32_t> mapped(idxs);
        for (auto &x : mapped)
          x += currentSize;
        return mapped;
      };
      for (auto &op : casted->getAppliedTensors()) {
        if (op.isUnitary)
          m_state->applyGate(mapQubitIdxs(op.controlQubitIds),
                             mapQubitIdxs(op.targetQubitIds), op.deviceData,
                             op.isAdjoint);
        else
          m_state->applyQubitProjector(op.deviceData,
                                       mapQubitIdxs(op.targetQubitIds));
      }
    }
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
