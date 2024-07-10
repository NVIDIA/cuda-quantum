/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq/spin_op.h"
#include "cutensornet.h"

namespace nvqir {

/// @brief Utility class converting `cudaq::spin_op` to
/// cutensornetNetworkOperator_t
class TensorNetworkSpinOp {
  cutensornetHandle_t m_cutnHandle;
  cutensornetNetworkOperator_t m_cutnNetworkOperator;
  std::unordered_map<cudaq::pauli, void *> m_pauli_d;
  std::complex<double> m_identityCoeff = 0.0;
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
  std::complex<double> getIdentityTermOffset() const { return m_identityCoeff; }

  /// @brief Destructor
  ~TensorNetworkSpinOp();
};
} // namespace nvqir
