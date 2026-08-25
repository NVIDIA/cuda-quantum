/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <unordered_map>

#include "cutensornet.h"
#include "tensornet_state.h"
#include "tensornet_utils.h"
#include "timing_utils.h"

#include "common/SimulationState.h"

namespace nvqir {

template <typename ScalarType = double>
class TensorNetSimulationState : public cudaq::SimulationState {
  static constexpr cudaDataType_t cudaDataType =
      std::is_same_v<ScalarType, float> ? CUDA_C_32F : CUDA_C_64F;

public:
  TensorNetSimulationState(std::unique_ptr<TensorNetState<ScalarType>> inState,
                           ScratchDeviceMem &inScratchPad,
                           cutensornetHandle_t cutnHandle,
                           std::mt19937 &randomEngine);

  TensorNetSimulationState(const TensorNetSimulationState &) = delete;
  TensorNetSimulationState &
  operator=(const TensorNetSimulationState &) = delete;
  TensorNetSimulationState(TensorNetSimulationState &&) noexcept = default;
  TensorNetSimulationState &
  operator=(TensorNetSimulationState &&) noexcept = delete;

  virtual ~TensorNetSimulationState();

  std::complex<double> overlap(const cudaq::SimulationState &other) override;

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;
  std::size_t getNumQubits() const override;
  void dump(std::ostream &) const override;
  cudaq::SimulationState::precision getPrecision() const override {
    return std::is_same_v<ScalarType, float>
               ? cudaq::SimulationState::precision::fp32
               : cudaq::SimulationState::precision::fp64;
  }

  bool isDeviceData() const override { return true; }

  bool isArrayLike() const override { return false; }

  Tensor getTensor(std::size_t tensorIdx = 0) const override;
  std::unique_ptr<cudaq::SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr,
                       std::size_t dataType) override;
  /// Get component tensors
  // Note: for the full tensor network state, we return the gate tensors in the
  // full tensor network. The root tensor of this network is not computed and
  // may require lots of memory. Thus, we don't use it as the 'tensor' object.
  std::vector<Tensor> getTensors() const override;

  std::size_t getNumTensors() const override;

  void destroyState() override;
  void toHost(std::complex<double> *clientAllocatedData,
              std::size_t numElements) const override;
  void toHost(std::complex<float> *clientAllocatedData,
              std::size_t numElements) const override;

  template <typename T>
  void toHostImpl(std::complex<T> *clientAllocatedData,
                  std::size_t numElements) const;
  /// @brief Return a reference to all the tensors that have been applied to the
  /// state.
  const std::vector<AppliedTensorOp> &getAppliedTensors() const {
    return m_state->m_tensorOps;
  }

  template <typename ScalarTy>
  friend class SimulatorTensorNet;

protected:
  std::unique_ptr<TensorNetState<ScalarType>> m_state;
  ScratchDeviceMem &scratchPad;
  cutensornetHandle_t m_cutnHandle;
  // Max number of qubits whereby the tensor network state should be contracted
  // and cached into a state vector.
  // This speeds up sequential state amplitude accessors for small states.
  static constexpr std::size_t g_maxQubitsForStateContraction = 30;
  std::vector<std::complex<ScalarType>> m_contractedStateVec;
  std::mt19937 &m_randomEngine;
};
} // namespace nvqir

#include "tn_simulation_state.inc"
