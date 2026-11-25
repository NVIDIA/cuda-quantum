/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/gadgets/utils.h"

#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace cudaq::gadgets {
inline namespace v0 {

using OutcomeRef = std::string;

/// Data a standard operation carries.
using Payload = std::variant<ParamValue, PauliString, CliffordImages,
                             ConditionalPauli, DiagonalUnitary, Unitary>;

/// Standard operations. Variable-width ones take their width from a payload.
enum class Opcode : u32 {
  // Paulis.
  I,
  X,
  Y,
  Z,
  // Cliffords.
  H,
  S,
  SX,
  SY,
  CX,
  CZ,
  CY,
  SWAP,
  CLIFFORD, // Variable width: images of X and Z.
  // Preparation of the +1 eigenstate, measurement, and reset.
  PREPARE_X,
  PREPARE_Y,
  PREPARE_Z,
  MEASURE_X, // Destructive.
  MEASURE_Y,
  MEASURE_Z,
  BELL_MEASURE, // Destructive.
  BELL_PREPARE,
  RESET,
  IDLE,
  RANDOM_BIT,        // Produces a uniformly random bit.
  PAULI_MEASUREMENT, // Variable width: joint, by Pauli string.
  CONDITIONAL_PAULI, // Variable width: on the parity of outcomes.
  // Non-Cliffords.
  T,
  SQRT_T,
  CS,
  CCZ,
  CCX,
  CSWAP,
  RX,
  RY,
  RZ,
  DIAGONAL_UNITARY, // Variable width: by its diagonal.
  UNITARY,          // Variable width: by its matrix. Keep last.
};

inline constexpr u32 num_opcodes = static_cast<u32>(Opcode::UNITARY) + 1;

/// The opcode's shape given its payload; a variable-width opcode takes its
/// width from the payload. Input i continues as output i. Throws
/// std::invalid_argument if the payload does not fit the opcode.
Shape get_shape(Opcode opcode,
                const std::optional<Payload> &payload = std::nullopt);

/// The opcode's name, e.g. "CX".
std::string_view to_string(Opcode opcode);

/// The opcode with this name or alias (NOT, CNOT, PREPARE_ZERO, PREPARE_PLUS,
/// TOFFOLI, FREDKIN), if any.
std::optional<Opcode> find_opcode(std::string_view name);

} // namespace v0
} // namespace cudaq::gadgets
