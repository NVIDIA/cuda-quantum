/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq/operators.h"
#include "cutensornet.h"

namespace nvqir {

/// @brief Utility class converting `cudaq::spin_op` to
/// cutensornetNetworkOperator_t
template <typename ScalarType>
class TensorNetworkSpinOp {
  static constexpr cudaDataType_t cudaDataType =
      std::is_same_v<ScalarType, float> ? CUDA_C_32F : CUDA_C_64F;

  cutensornetHandle_t m_cutnHandle;
  cutensornetNetworkOperator_t m_cutnNetworkOperator;
  std::unordered_map<cudaq::pauli, void *> m_pauli_d;
  std::complex<ScalarType> m_identityCoeff = 0.0;
  std::vector<void *> m_mat_d;

public:
  /// @brief Constructor from a `cudaq::spin_op`
  TensorNetworkSpinOp(const cudaq::spin_op &spinOp, cutensornetHandle_t handle);

  /// @brief Retrieve the cutensornetNetworkOperator_t representation
  cutensornetNetworkOperator_t getNetworkOperator() {
    return m_cutnNetworkOperator;
  }

  /// @brief Get the identity term offset/coefficient.
  /// Note: the cutensornetNetworkOperator_t representation doesn't include the
  /// identity term since its expectation value can be trivially computed (equal
  /// the coefficient)
  std::complex<ScalarType> getIdentityTermOffset() const {
    return m_identityCoeff;
  }

  /// @brief Destructor
  ~TensorNetworkSpinOp();
};
} // namespace nvqir

#include "tensornet_spin_op.inc"
