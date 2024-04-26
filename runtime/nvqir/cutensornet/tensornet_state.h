/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/EigenDense.h"
#include "common/SimulationState.h"
#include "cutensornet.h"
#include "tensornet_utils.h"
#include "timing_utils.h"
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

/// Track gate tensors that were appended to the tensor network.
struct AppliedTensorOp {
  void *deviceData = nullptr;
  std::vector<int32_t> qubitIds;
  bool isAdjoint;
  bool isUnitary;
};

/// @brief Wrapper of cutensornetState_t to provide convenient API's for CUDAQ
/// simulator implementation.
class TensorNetState {

protected:
  std::size_t m_numQubits;
  cutensornetHandle_t m_cutnHandle;
  cutensornetState_t m_quantumState;
  /// Track id of gate tensors that are applied to the state tensors.
  std::int64_t m_tensorId = InvalidTensorIndexValue;
  // Device memory pointers to be cleaned up.
  std::vector<void *> m_tempDevicePtrs;
  // Tensor ops that have been applied to the state.
  std::vector<AppliedTensorOp> m_tensorOps;

public:
  /// @brief Constructor
  TensorNetState(std::size_t numQubits, cutensornetHandle_t handle);

  /// @brief Constructor (specific basis state)
  TensorNetState(const std::vector<int> &basisState,
                 cutensornetHandle_t handle);

  std::unique_ptr<TensorNetState> clone() const;

  /// Reconstruct/initialize a state from MPS tensors
  static std::unique_ptr<TensorNetState>
  createFromMpsTensors(const std::vector<MPSTensor> &mpsTensors,
                       cutensornetHandle_t handle,
                       std::vector<MPSTensor> &outTensors);

  /// Reconstruct/initialize a tensor network state from a list of tensor
  /// operators.
  static std::unique_ptr<TensorNetState>
  createFromOpTensors(std::size_t numQubits,
                      const std::vector<AppliedTensorOp> &opTensors,
                      cutensornetHandle_t handle);

  // Create a tensor network state from the input state vector.
  // Note: this is not the most efficient mode of initialization. However, this
  // is required if users have a state vector that they want to initialize the
  // tensor network simulator with.
  static std::unique_ptr<TensorNetState>
  createFromStateVector(std::span<std::complex<double>> stateVec,
                        cutensornetHandle_t handle);

  /// @brief Apply a unitary gate
  /// @param qubitIds Qubit operands
  /// @param gateDeviceMem Gate unitary matrix in device memory
  /// @param adjoint Apply the adjoint of gate matrix if true
  void applyGate(const std::vector<int32_t> &qubitIds, void *gateDeviceMem,
                 bool adjoint = false);

  /// @brief Apply a projector matrix (non-unitary)
  /// @param proj_d Projector matrix (expected a 2x2 matrix in column major)
  /// @param qubitIdx Qubit operand
  void applyQubitProjector(void *proj_d, const std::vector<int32_t> &qubitIdx);

  /// @brief Add a number of qubits to the state.
  /// The qubits will be initialized to zero.
  void addQubits(std::size_t numQubits);

  /// @brief Add a number of qubits in a specific superposition to the current
  /// state. The size of the wave function determines the number of qubits.
  void addQubits(std::span<std::complex<double>> stateVec);

  /// @brief Accessor to the cuTensorNet handle (context).
  cutensornetHandle_t getInternalContext() { return m_cutnHandle; }

  /// @brief Accessor to the underlying `cutensornetState_t`
  cutensornetState_t getInternalState() { return m_quantumState; }

  /// @brief Perform measurement sampling on the quantum state.
  std::unordered_map<std::string, size_t>
  sample(const std::vector<int32_t> &measuredBitIds, int32_t shots);

  /// @brief Contract the tensor network representation to retrieve the state
  /// vector.
  std::vector<std::complex<double>>
  getStateVector(const std::vector<int32_t> &projectedModes = {},
                 const std::vector<int64_t> &projectedModeValues = {});

  /// @brief Compute the reduce density matrix on a set of qubits
  ///
  /// The order of the specified qubits (`cutensornet` open state modes) will be
  /// respected when computing the RDM.
  std::vector<std::complex<double>>
  computeRDM(const std::vector<int32_t> &qubits);

  /// Factorize the `cutensornetState_t` into matrix product state form.
  /// Returns MPS tensors in GPU device memory.
  /// Note: the caller assumes the ownership of these pointers, thus needs to
  /// clean them up properly (with cudaFree).
  std::vector<MPSTensor> factorizeMPS(
      int64_t maxExtent, double absCutoff, double relCutoff,
      cutensornetTensorSVDAlgo_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ);

  /// @brief  Compute the expectation value w.r.t. a
  /// `cutensornetNetworkOperator_t`
  ///
  /// The `cutensornetNetworkOperator_t` can be constructed from
  /// `cudaq::spin_op`, i.e., representing a sum of Pauli products with
  /// different coefficients.
  std::complex<double>
  computeExpVal(cutensornetNetworkOperator_t tensorNetworkOperator);

  /// @brief Number of qubits that this state represents.
  std::size_t getNumQubits() const { return m_numQubits; }

  /// @brief True if the state contains gate tensors (not just initial qubit
  /// tensors)
  bool isDirty() const { return m_tensorId > 0; }
  /// @brief Destructor
  ~TensorNetState();

private:
  friend class SimulatorMPS;
  friend class TensorNetSimulationState;
  /// Internal method to contract the tensor network.
  /// Returns device memory pointer and size (number of elements).
  std::pair<void *, std::size_t> contractStateVectorInternal(
      const std::vector<int32_t> &projectedModes,
      const std::vector<int64_t> &projectedModeValues = {});
};
} // namespace nvqir
