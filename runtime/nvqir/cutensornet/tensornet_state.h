/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cutensornet.h"
#include "tensornet_utils.h"
#include "timing_utils.h"
#include <unordered_map>

namespace nvqir {
/// @brief Wrapper of cutensornetState_t to provide convenient API's for CUDAQ
/// simulator implementation.
class TensorNetState {
  std::size_t m_numQubits;
  cutensornetHandle_t m_cutnHandle;
  cutensornetState_t m_quantumState;

public:
  /// @brief Constructor
  TensorNetState(std::size_t numQubits, cutensornetHandle_t handle);

  /// @brief Apply a unitary gate
  /// @param qubitIds Qubit operands
  /// @param gateDeviceMem Gate unitary matrix in device memory
  /// @param adjoint Apply the adjoint of gate matrix if true
  void applyGate(const std::vector<int32_t> &qubitIds, void *gateDeviceMem,
                 bool adjoint = false);

  /// @brief Apply a projector matrix (non-unitary)
  /// @param proj_d Projector matrix (expected a 2x2 matrix in column major)
  /// @param qubitIdx Qubit operand
  void applyQubitProjector(void *proj_d, int32_t qubitIdx);

  /// @brief Accessor to the underlying `cutensornetState_t`
  cutensornetState_t getInternalState() { return m_quantumState; }

  /// @brief Perform measurement sampling on the quantum state.
  std::unordered_map<std::string, size_t>
  sample(const std::vector<int32_t> &measuredBitIds, int32_t shots);

  /// @brief Contract the tensor network representation to retrieve the state
  /// vector.
  std::vector<std::complex<double>> getStateVector();

  /// @brief Compute the reduce density matrix on a set of qubits
  ///
  /// The order of the specified qubits (`cutensornet` open state modes) will be
  /// respected when computing the RDM.
  std::vector<std::complex<double>>
  computeRDM(const std::vector<int32_t> &qubits);

  /// Factorize the `cutensornetState_t` into matrix product state form.
  // Returns MPS tensors in GPU device memory.
  // Note: the caller assumes the ownership of these pointers, thus needs to
  // clean them up properly (with cudaFree).
  std::vector<void *> factorizeMPS(
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

  /// @brief Destructor
  ~TensorNetState();
};
} // namespace nvqir
