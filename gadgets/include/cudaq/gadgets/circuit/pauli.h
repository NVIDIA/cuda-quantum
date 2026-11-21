/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::gadgets {
inline namespace v0 {

enum class Pauli : std::uint8_t { I, X, Y, Z };

/// Throws std::invalid_argument for an invalid value.
char to_char(Pauli pauli);

/// Accepts IXYZ, and "_" for I; throws std::invalid_argument otherwise.
Pauli pauli_from_char(char c);

/// i^phase_exponent * ops[0] (x) ops[1] (x) ..., one op per qubit.
struct PauliString {
  std::uint8_t phase_exponent = 0; // 0..3
  std::vector<Pauli> ops;

  /// Phase prefixes, indexed by phase_exponent.
  static constexpr std::string_view phase_prefixes[] = {"+", "+i", "-", "-i"};

  /// Phase prefix and ops, e.g. "-iXIZ".
  std::string to_string() const;

  /// Inverse of to_string; the prefix defaults to "+", and "_" means I.
  /// Throws std::invalid_argument on bad input.
  static PauliString from_string(std::string_view text);
};

} // namespace v0
} // namespace cudaq::gadgets
