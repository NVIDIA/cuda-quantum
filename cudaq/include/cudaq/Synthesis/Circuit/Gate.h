/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Gate
//===----------------------------------------------------------------------===//

/// Single-qubit Clifford+T gate. uint8_t backing keeps sizeof(Gate) == 1,
/// matching the memory footprint of a char-per-gate string representation.
enum class Gate : uint8_t {
  H = 0, ///< Hadamard.
  S,     ///< Phase gate diag(1, i) = T^2.
  T,     ///< pi/8 gate diag(1, e^(i*pi/4)).
  X,     ///< Pauli-X bit-flip.
  W,     ///< Global phase omega = e^(i*pi/4).
};

} // namespace cudaq::synth
