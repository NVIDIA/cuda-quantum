/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/EigenDense.h"
#include "common/SimulationState.h"
#include "common/Timing.h"
#include "cudaq/operators.h"
#include "cutensornet.h"
#include "tensornet_utils.h"
#include "timing_utils.h"
#include <optional>
#include <span>
#include <unordered_map>

namespace nvqir {
/// This is used to track whether the tensor state is default initialized vs
/// already has some gates applied to.
constexpr std::int64_t InvalidTensorIndexValue = -1;

/// @brief An MPSTensor is a representation
/// of a MPS tensor and encapsulates the
/// tensor device data and the tensor extents.
struct MPSTensor {
  void *deviceData = nullptr;
  std::vector<int64_t> extents;
};

// Struct captures noise channel data.
struct NoiseChannelData {
  // Device memory tensors represent general Kraus ops or unitary matrices.
  std::vector<void *> tensorData;
  // If tensorData represents unitary matrices, a list of probabilities (same
  // length) can be supplied in this field. If empty, the tensors are treated as
  // general Kraus ops.
  std::vector<double> probabilities;
};

/// Track gate tensors that were appended to the tensor network.
struct AppliedTensorOp {
  void *deviceData = nullptr;
  std::optional<NoiseChannelData> noiseChannel;
  std::vector<int32_t> targetQubitIds;
  std::vector<int32_t> controlQubitIds;
  bool isAdjoint;
  bool isUnitary;
  AppliedTensorOp(void *dataPtr, const std::vector<int32_t> &targetQubits,
                  const std::vector<int32_t> &controlQubits, bool adjoint,
                  bool unitary)
      : deviceData(dataPtr), targetQubitIds(targetQubits),
        controlQubitIds(controlQubits), isAdjoint(adjoint), isUnitary(unitary) {
  }

  AppliedTensorOp(const std::vector<int32_t> &qubits,
                  const std::vector<void *> &krausOps,
                  const std::vector<double> &probabilities)
      : targetQubitIds(qubits),
        noiseChannel(NoiseChannelData(krausOps, probabilities)) {}
};

/// @brief Wrapper of cutensornetState_t to provide convenient API's for CUDA-Q
/// simulator implementation.
template <typename ScalarType = double>
class TensorNetState {
  using DataType = std::complex<ScalarType>;
  static constexpr cudaDataType_t cudaDataType =
      std::is_same_v<ScalarType, float> ? CUDA_C_32F : CUDA_C_64F;

protected:
  std::size_t m_numQubits;
  cutensornetHandle_t m_cutnHandle;
  cutensornetState_t m_quantumState;
  /// Track id of gate tensors that are applied to the state tensors.
  std::int64_t m_tensorId = InvalidTensorIndexValue;
  struct TempDevicePtrDeleter {
    void operator()(void *ptr) const {
      if (ptr)
        cudaFree(ptr);
    }
  };

  // Device memory pointers to be cleaned up.
  std::vector<std::shared_ptr<void>> m_tempDevicePtrs;
  // Tensor ops that have been applied to the state.
  std::vector<AppliedTensorOp> m_tensorOps;
  ScratchDeviceMem &scratchPad;
  // Random number generator measurement sampling.
  // This is a reference to the backend random number generator, which can be
  // reseeded by users.
  std::mt19937 &m_randomEngine;
  // True if deterministic path-finding is to be used
  static bool m_deterministic;
  bool m_hasNoiseChannel = false;

public:
  // The number of hyper samples used in the tensor network contraction path
  // finder
  static std::int32_t numHyperSamples;

  /// @brief Constructor
  TensorNetState(std::size_t numQubits, ScratchDeviceMem &inScratchPad,
                 cutensornetHandle_t handle, std::mt19937 &randomEngine);

  /// @brief Constructor (specific basis state)
  TensorNetState(const std::vector<int> &basisState,
                 ScratchDeviceMem &inScratchPad, cutensornetHandle_t handle,
                 std::mt19937 &randomEngine);

  std::unique_ptr<TensorNetState> clone() const;

  /// Default number of trajectories for observe (with noise) in case no option
  /// is provided.
  static inline constexpr int g_numberTrajectoriesForObserve = 1000;

  /// Reconstruct/initialize a state from MPS tensors
  static std::unique_ptr<TensorNetState>
  createFromMpsTensors(const std::vector<MPSTensor> &mpsTensors,
                       ScratchDeviceMem &inScratchPad,
                       cutensornetHandle_t handle, std::mt19937 &randomEngine);

  /// Reconstruct/initialize a tensor network state from a list of tensor
  /// operators.
  static std::unique_ptr<TensorNetState>
  createFromOpTensors(std::size_t numQubits,
                      const std::vector<AppliedTensorOp> &opTensors,
                      ScratchDeviceMem &inScratchPad,
                      cutensornetHandle_t handle, std::mt19937 &randomEngine);

  // Create a tensor network state from the input state vector.
  // Note: this is not the most efficient mode of initialization. However, this
  // is required if users have a state vector that they want to initialize the
  // tensor network simulator with.
  static std::unique_ptr<TensorNetState>
  createFromStateVector(std::span<std::complex<ScalarType>> stateVec,
                        ScratchDeviceMem &inScratchPad,
                        cutensornetHandle_t handle, std::mt19937 &randomEngine);

  /// @brief Apply a unitary gate
  /// @param controlQubits Controlled qubit operands
  /// @param targetQubits Target qubit operands
  /// @param gateDeviceMem Gate unitary matrix in device memory
  /// @param adjoint Apply the adjoint of gate matrix if true
  void applyGate(const std::vector<int32_t> &controlQubits,
                 const std::vector<int32_t> &targetQubits, void *gateDeviceMem,
                 bool adjoint = false);

  /// @brief Apply a unitary channel
  void applyUnitaryChannel(const std::vector<int32_t> &qubits,
                           const std::vector<void *> &krausOps,
                           const std::vector<double> &probabilities);
  /// @brief Apply a general noise channel
  void applyGeneralChannel(const std::vector<int32_t> &qubits,
                           const std::vector<void *> &krausOps);
  /// @brief Apply a projector matrix (non-unitary)
  /// @param proj_d Projector matrix (expected a 2x2 matrix in column major)
  /// @param qubitIdx Qubit operand
  void applyQubitProjector(void *proj_d, const std::vector<int32_t> &qubitIdx);

  /// @brief Add a number of qubits to the state.
  /// The qubits will be initialized to zero.
  void addQubits(std::size_t numQubits);

  /// @brief Add a number of qubits in a specific superposition to the current
  /// state. The size of the wave function determines the number of qubits.
  void addQubits(std::span<DataType> stateVec);

  /// @brief Accessor to the cuTensorNet handle (context).
  cutensornetHandle_t getInternalContext() { return m_cutnHandle; }

  /// @brief Accessor to the underlying `cutensornetState_t`
  cutensornetState_t getInternalState() { return m_quantumState; }

  /// @brief Perform measurement sampling on the quantum state.
  std::unordered_map<std::string, size_t>
  sample(const std::vector<int32_t> &measuredBitIds, int32_t shots,
         bool enableCacheWorkspace);

  /// @brief Contract the tensor network representation to retrieve the state
  /// vector.
  std::vector<DataType>
  getStateVector(const std::vector<int32_t> &projectedModes = {},
                 const std::vector<int64_t> &projectedModeValues = {});

  /// @brief Compute the reduce density matrix on a set of qubits
  ///
  /// The order of the specified qubits (`cutensornet` open state modes) will be
  /// respected when computing the RDM.
  std::vector<DataType> computeRDM(const std::vector<int32_t> &qubits);

  /// Factorize the `cutensornetState_t` into matrix product state form.
  /// Returns MPS tensors in GPU device memory.
  /// Note: the caller assumes the ownership of these pointers, thus needs to
  /// clean them up properly (with cudaFree).
  std::vector<MPSTensor>
  factorizeMPS(int64_t maxExtent, double absCutoff, double relCutoff,
               cutensornetTensorSVDAlgo_t algo,
               const std::optional<cutensornetStateMPSGaugeOption_t> &gauge);

  /// @brief Compute the expectation value of an observable
  /// @param product_terms the terms of the observable (operator sum)
  /// @param numberTrajectories the number of trajectories to use
  std::vector<DataType>
  computeExpVals(const std::vector<cudaq::spin_op_term> &product_terms,
                 const std::optional<std::size_t> &numberTrajectories);

  /// @brief Evaluate the expectation value of a given
  /// `cutensornetNetworkOperator_t`
  DataType computeExpVal(cutensornetNetworkOperator_t tensorNetworkOperator,
                         const std::optional<std::size_t> &numberTrajectories);

  /// @brief Number of qubits that this state represents.
  std::size_t getNumQubits() const { return m_numQubits; }

  /// @brief True if the state contains gate tensors (not just initial qubit
  /// tensors)
  bool isDirty() const { return m_tensorId > 0; }

  /// @brief Helper to reverse qubit order of the input state vector.
  static std::vector<std::complex<ScalarType>>
  reverseQubitOrder(std::span<std::complex<ScalarType>> stateVec);

  /// @brief Apply all the cached ops to the state.
  void applyCachedOps();

  /// @brief Set the state to a zero state
  void setZeroState();

  /// @brief Returns true if the state has at least one general channel applied.
  bool hasGeneralChannelApplied() const;

  /// @brief Destructor
  ~TensorNetState();

private:
  template <typename ScalarTy>
  friend class SimulatorMPS;
  template <typename ScalarTy>
  friend class SimulatorTensorNet;
  template <typename ScalarTy>
  friend class TensorNetSimulationState;
  /// Internal method to contract the tensor network.
  /// Returns device memory pointer and size (number of elements).
  std::pair<void *, std::size_t> contractStateVectorInternal(
      const std::vector<int32_t> &projectedModes,
      const std::vector<int64_t> &projectedModeValues = {});

  /// Internal methods to perform MPS factorize.
  // Note: `factorizeMPS` is an end-to-end API for factorization.
  // This factorization can be split into `cutensornetStateFinalizeMPS` and
  // `cutensornetStateCompute` to facilitate reuse.
  std::vector<MPSTensor> setupMPSFactorize(
      int64_t maxExtent, double absCutoff, double relCutoff,
      cutensornetTensorSVDAlgo_t algo,
      const std::optional<cutensornetStateMPSGaugeOption_t> &gauge);
  void computeMPSFactorize(std::vector<MPSTensor> &mpsTensors);

  /// Internal methods for sampling
  std::pair<cutensornetStateSampler_t, cutensornetWorkspaceDescriptor_t>
  prepareSample(const std::vector<int32_t> &measuredBitIds);

  std::unordered_map<std::string, size_t>
  executeSample(cutensornetStateSampler_t &sampler,
                cutensornetWorkspaceDescriptor_t &workspaceDesc,
                const std::vector<int32_t> &measuredBitIds, int32_t shots,
                bool enableCacheWorkspace);
};
} // namespace nvqir

#include "tensornet_state.inc"
