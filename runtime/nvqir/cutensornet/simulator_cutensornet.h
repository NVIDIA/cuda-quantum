/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
template <typename ScalarType = double>
class SimulatorTensorNetBase : public nvqir::CircuitSimulatorBase<ScalarType> {
public:
  using DataType = std::complex<ScalarType>;
  static constexpr cudaDataType_t cudaDataType =
      std::is_same_v<ScalarType, float> ? CUDA_C_32F : CUDA_C_64F;
  using GateApplicationTask =
      typename nvqir::CircuitSimulatorBase<ScalarType>::GateApplicationTask;
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

  /// @brief Apply a noise channel
  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &controls,
                         const std::vector<std::size_t> &targets,
                         const std::vector<double> &params) override;

  bool isValidNoiseChannel(const cudaq::noise_model_type &type) const override;

  /// @brief Apply the given kraus_channel on the provided targets.
  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::size_t> &targets) override;

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

  QubitOrdering getQubitOrdering() const override { return QubitOrdering::msb; }

  /// @brief Sample a subset of qubits
  virtual cudaq::ExecutionResult
  sample(const std::vector<std::size_t> &measuredBits,
         const int shots) override;

  /// @brief Evaluate the expectation value of a given observable
  virtual cudaq::observe_result observe(const cudaq::spin_op &op) override;

  /// Clone API
  virtual nvqir::CircuitSimulator *clone() override;

  virtual std::unique_ptr<cudaq::SimulationState>
  getSimulationState() override {
    throw std::runtime_error("[tensornet] getSimulationState not implemented");
    return nullptr;
  }
  /// Swap gate implementation
  // Note: cutensornetStateApplyControlledTensorOperator can only handle
  // single-target.
  void swap(const std::vector<std::size_t> &ctrlBits, const std::size_t srcIdx,
            const std::size_t tgtIdx) override;

  void setRandomSeed(std::size_t randomSeed) override;

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

  /// @brief Return true if this simulator can use cache workspace (e.g., for
  /// intermediate tensors)
  virtual bool requireCacheWorkspace() const = 0;

  /// @brief Return true if this simulator handle general noise channel
  /// (non-unitary).
  virtual bool canHandleGeneralNoiseChannel() const = 0;

private:
  // Helper to apply a Kraus channel
  void applyKrausChannel(const std::vector<int32_t> &qubits,
                         const cudaq::kraus_channel &channel);

protected:
  cutensornetHandle_t m_cutnHandle;
  std::unique_ptr<TensorNetState<ScalarType>> m_state;
  std::unordered_map<std::string, void *> m_gateDeviceMemCache;
  ScratchDeviceMem scratchPad;
  // Random number generator for generating 32-bit numbers with a state size of
  // 19937 bits for measurements.
  std::mt19937 m_randomEngine;
  // Max number of controlled ranks (qubits) that the full matrix of the
  // controlled gate is used as tensor op.
  // Default is 1.
  // MPS only supports 1 (higher number of controlled ranks must use
  // cutensornetStateApplyControlledTensorOperator). Tensornet supports
  // arbitrary values.
  std::size_t m_maxControlledRankForFullTensorExpansion = 1;

  // Flag to enable contraction path reuse when computing the expectation value
  // (observe).
  //   Default is off (no contraction path reuse).
  //   Reusing the path, while saving the path finding time, prevents lightcone
  //   simplification, e.g., when the spin op is sparse (only acting on a few
  //   qubits).
  bool m_reuseContractionPathObserve = false;
};

} // end namespace nvqir

#include "simulator_cutensornet.inc"
