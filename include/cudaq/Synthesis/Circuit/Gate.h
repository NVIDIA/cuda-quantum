/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>

namespace cudaq::synth {

/// Single-qubit Clifford+T gate.
///
/// uint8_t backing: sizeof(Gate) == 1, matching the memory density of a
/// char-per-gate string representation.
enum class Gate : uint8_t {
  H = 0, ///< Hadamard
  S,     ///< Phase gate diag(1, i) = T²
  T,     ///< π/8 gate diag(1, e^{iπ/4})
  X,     ///< Pauli-X (bit-flip)
  W,     ///< Global phase ω = e^{iπ/4}
};

} // namespace cudaq::synth
