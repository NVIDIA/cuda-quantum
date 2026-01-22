/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq.h"
#include "simulator_cutensornet.h"
#include "tn_simulation_state.h"

// Forward declaration
#ifdef TENSORNET_FP32
extern "C" nvqir::CircuitSimulator *getCircuitSimulator_tensornet_fp32();
#else
extern "C" nvqir::CircuitSimulator *getCircuitSimulator_tensornet();
#endif

namespace nvqir {
template <typename ScalarType = double>
class SimulatorTensorNet : public SimulatorTensorNetBase<ScalarType> {
  using SimulatorTensorNetBase<ScalarType>::m_cutnHandle;
  using SimulatorTensorNetBase<
      ScalarType>::m_maxControlledRankForFullTensorExpansion;
  using SimulatorTensorNetBase<ScalarType>::m_state;
  using SimulatorTensorNetBase<ScalarType>::scratchPad;
  using SimulatorTensorNetBase<ScalarType>::m_randomEngine;

public:
  SimulatorTensorNet() : SimulatorTensorNetBase<ScalarType>() {
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

    // Retrieve user-defined controlled rank setting if provided.
    if (auto *maxControlledRankEnvVar =
            std::getenv("CUDAQ_TENSORNET_CONTROLLED_RANK")) {
      auto maxControlledRank = std::atoi(maxControlledRankEnvVar);
      if (maxControlledRank <= 0)
        throw std::runtime_error(cudaq_fmt::format(
            "Invalid CUDAQ_TENSORNET_CONTROLLED_RANK environment "
            "variable setting. Expecting a "
            "positive integer value, got '{}'.",
            maxControlledRank));

      CUDAQ_INFO("Setting max controlled rank for full tensor expansion from "
                 "{} to {}.",
                 m_maxControlledRankForFullTensorExpansion, maxControlledRank);
      m_maxControlledRankForFullTensorExpansion = maxControlledRank;
    }
  }

  // Nothing to do for state preparation
  virtual void prepareQubitTensorState() override {}
#ifdef TENSORNET_FP32
  virtual std::string name() const override { return "tensornet-fp32"; }
#else
  virtual std::string name() const override { return "tensornet"; }
#endif
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
    return std::make_unique<TensorNetSimulationState<ScalarType>>(
        std::move(m_state), scratchPad, m_cutnHandle, m_randomEngine);
  }

  void addQubitsToState(std::size_t numQubits, const void *ptr) override {
    LOG_API_TIME();
    if (!m_state) {
      if (!ptr) {
        m_state = std::make_unique<TensorNetState<ScalarType>>(
            numQubits, scratchPad, m_cutnHandle, m_randomEngine);
      } else {
        auto *casted = reinterpret_cast<std::complex<ScalarType> *>(
            const_cast<void *>(ptr));
        std::span<std::complex<ScalarType>> stateVec(casted, 1ULL << numQubits);
        m_state = TensorNetState<ScalarType>::createFromStateVector(
            stateVec, scratchPad, m_cutnHandle, m_randomEngine);
      }
    } else {
      if (!ptr) {
        m_state->addQubits(numQubits);
      } else {
        auto *casted = reinterpret_cast<std::complex<ScalarType> *>(
            const_cast<void *>(ptr));
        std::span<std::complex<ScalarType>> stateVec(casted, 1ULL << numQubits);
        m_state->addQubits(stateVec);
      }
    }
  }

  virtual void
  addQubitsToState(const cudaq::SimulationState &in_state) override {
    LOG_API_TIME();
    const TensorNetSimulationState<ScalarType> *const casted =
        dynamic_cast<const TensorNetSimulationState<ScalarType> *>(&in_state);
    if (!casted)
      throw std::invalid_argument(
          "[Tensornet simulator] Incompatible state input");
    if (!m_state) {
      m_state = TensorNetState<ScalarType>::createFromOpTensors(
          in_state.getNumQubits(), casted->getAppliedTensors(), scratchPad,
          m_cutnHandle, m_randomEngine);
      // Need to extend lifetime of all the device pointers stored in the input
      // state.
      m_state->m_tempDevicePtrs = casted->m_state->m_tempDevicePtrs;
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
      // Append the temp. pointer
      m_state->m_tempDevicePtrs.insert(
          m_state->m_tempDevicePtrs.end(),
          casted->m_state->m_tempDevicePtrs.begin(),
          casted->m_state->m_tempDevicePtrs.end());
    }
  }
  bool requireCacheWorkspace() const override { return true; }
  bool canHandleGeneralNoiseChannel() const override {
    // Full tensornet simulator doesn't support general noise channels (only
    // unitary mixture channels)
    return false;
  }

private:
#ifdef TENSORNET_FP32
  friend nvqir::CircuitSimulator * ::getCircuitSimulator_tensornet_fp32();
#else
  friend nvqir::CircuitSimulator * ::getCircuitSimulator_tensornet();
#endif
  /// @brief Has cuTensorNet MPI been initialized?
  bool m_cutnMpiInitialized = false;
};
} // namespace nvqir
