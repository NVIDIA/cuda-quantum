/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cutensornet.h"

namespace nvqir {
/// @brief Interface for externally provided expectation calculation executor
/// (optionally used).
/// An external library, implementing this interface, can be provided to perform
/// custom expectation value computation.
///
/// Requirements:
///
/// (1) The extension
/// library should expose a C-API function with the signature:
///
///    CutensornetExecutor* getCutnExecutor();
///
///  The backend shall look for that symbol to retrieve the extension.
///
/// (2) The extension library should be linked against the same (or compatible)
/// `cutensornet` library version.
///
/// (3) Provide CMAKE flag:
/// `-DCUDAQ_CUTENSORNET_PLUGIN_LIB=<path to compiled extension lib>` when
/// building the `tensornet` backends.
struct CutensornetExecutor {
  /// @brief Compute expectation values for a list of Pauli spin operators
  /// @details Spin operators are expressed in binary symplectic form:
  /// a list of boolean vectors, each representing a Pauli product (see
  /// spin_op.h).
  ///
  /// All symplectic boolean vectors shall have the same length,
  /// which is equal to the number of spin qubits times 2. The number of spin
  /// qubits in the operator is less than or equal to the number of qubits in
  /// the state.
  /// @param cutnHandle `cutensornet` handle that the plugin should be using
  /// when calling any `cutensornet` APIs
  /// @param quantumState quantum input state
  /// @param numQubits number of qubits in the quantum state
  /// @param symplecticRepr symplectic representation of the spin operators
  /// @return A list of expectation values (<psi|Op|psi>) in the same order as
  /// the list of input operators
  virtual std::vector<std::complex<double>>
  computeExpVals(cutensornetHandle_t cutnHandle,
                 cutensornetState_t quantumState, std::size_t numQubits,
                 const std::vector<std::vector<bool>> &symplecticRepr) = 0;

  virtual ~CutensornetExecutor() = default;
};
} // namespace nvqir
