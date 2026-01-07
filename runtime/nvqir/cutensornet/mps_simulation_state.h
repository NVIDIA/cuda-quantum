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
struct MPSSettings {
  // Default max bond dim
  int64_t maxBond = 64;
  // Default absolute cutoff
  double absCutoff = 1e-5;
  // Default relative cutoff
  double relCutoff = 1e-5;
  // Default SVD algorithm (Jacobi)
  cutensornetTensorSVDAlgo_t svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
  // Optional gauge option
  std::optional<cutensornetStateMPSGaugeOption_t> gaugeOption;
  MPSSettings();
};

template <typename ScalarType = double>
class MPSSimulationState : public cudaq::SimulationState {

public:
  MPSSimulationState(std::unique_ptr<TensorNetState<ScalarType>> inState,
                     const std::vector<MPSTensor> &mpsTensors,
                     ScratchDeviceMem &inScratchPad,
                     cutensornetHandle_t cutnHandle,
                     std::mt19937 &randomEngine);

  MPSSimulationState(const MPSSimulationState &) = delete;
  MPSSimulationState &operator=(const MPSSimulationState &) = delete;
  MPSSimulationState(MPSSimulationState &&) noexcept = default;
  MPSSimulationState &operator=(MPSSimulationState &&) noexcept = delete;

  virtual ~MPSSimulationState();

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

  Tensor getTensor(std::size_t tensorIdx = 0) const override;

  std::vector<Tensor> getTensors() const override;

  std::size_t getNumTensors() const override;

  void destroyState() override;

  bool isDeviceData() const override { return true; }

  bool isArrayLike() const override { return false; }

  std::unique_ptr<cudaq::SimulationState>
  createFromSizeAndPtr(std::size_t, void *, std::size_t dataType) override;
  void toHost(std::complex<double> *clientAllocatedData,
              std::size_t numElements) const override;
  void toHost(std::complex<float> *clientAllocatedData,
              std::size_t numElements) const override;
  template <typename T>
  void toHostImpl(std::complex<T> *clientAllocatedData,
                  std::size_t numElements) const;
  /// Encapsulate data needed to initialize an MPS state.
  struct MpsStateData {
    // Represents the tensor network state
    std::unique_ptr<TensorNetState<ScalarType>> networkState;
    // Individual MPS tensors
    std::vector<MPSTensor> tensors;
  };
  /// Util method to create an MPS state from an input state vector.
  // For example, state vector from the user's input.
  static MpsStateData createFromStateVec(cutensornetHandle_t cutnHandle,
                                         ScratchDeviceMem &inScratchPad,
                                         std::size_t size,
                                         std::complex<ScalarType> *data,
                                         int bondDim,
                                         std::mt19937 &randomEngine);

  /// Retrieve the MPS tensors
  std::vector<MPSTensor> getMpsTensors() const { return m_mpsTensors; }

protected:
  void deallocate();
  std::complex<double>
  computeOverlap(const std::vector<MPSTensor> &m_mpsTensors,
                 const std::vector<MPSTensor> &mpsOtherTensors);

  // The state that this owned.
  cutensornetHandle_t m_cutnHandle;
  std::unique_ptr<TensorNetState<ScalarType>> state;
  std::vector<MPSTensor> m_mpsTensors;
  ScratchDeviceMem &scratchPad;
  // Max number of qubits whereby the tensor network state should be contracted
  // and cached into a state vector.
  // This speeds up sequential state amplitude accessors for small states.
  static constexpr std::size_t g_maxQubitsForStateContraction = 30;
  std::vector<std::complex<ScalarType>> m_contractedStateVec;
  std::mt19937 &m_randomEngine;
};

} // namespace nvqir

#include "mps_simulation_state.inc"
