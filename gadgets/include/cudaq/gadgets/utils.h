/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
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

  std::size_t size() const { return ops.size(); }
  /// Whether the phase is real, i.e. +1 or -1.
  bool is_hermitian() const { return phase_exponent % 2 == 0; }

  /// Phase prefixes, indexed by phase_exponent.
  static constexpr std::string_view phase_prefixes[] = {"+", "+i", "-", "-i"};

  /// Phase prefix and ops, e.g. "-iXIZ".
  std::string to_string() const;

  /// Inverse of to_string; the prefix defaults to "+", and "_" means I.
  /// Throws std::invalid_argument on bad input.
  static PauliString from_string(std::string_view text);
};

using u32 = std::uint32_t;
using Param = std::string;
/// A constant or a parameter name.
using ParamValue = std::variant<double, Param>;

/// Numbers of qubit and classical inputs and outputs of an operation.
struct Shape {
  std::size_t q_in = 0;
  std::size_t q_out = 0;
  std::size_t c_in = 0;
  std::size_t c_out = 0;

  bool operator==(const Shape &other) const {
    return q_in == other.q_in && q_out == other.q_out && c_in == other.c_in &&
           c_out == other.c_out;
  }
  bool operator!=(const Shape &other) const { return !(*this == other); }
};

/// Images of X and Z on each of a Clifford's n qubits.
struct CliffordImages {
  /// Throws std::invalid_argument unless there are n images of each, all on
  /// n qubits.
  CliffordImages(std::vector<PauliString> x, std::vector<PauliString> z)
      : x(std::move(x)), z(std::move(z)) {
    bool square = this->z.size() == size();
    for (const auto *side : {&this->x, &this->z})
      for (const PauliString &image : *side)
        square = square && image.size() == size();
    if (!square)
      throw std::invalid_argument("Clifford images are not n x n");
  }
  std::size_t size() const { return x.size(); }
  const std::vector<PauliString> &get_x() const { return x; }
  const std::vector<PauliString> &get_z() const { return z; }

private:
  std::vector<PauliString> x;
  std::vector<PauliString> z;
};

/// Applies pauli when the parity of the operation's num_bits classical
/// inputs, flipped if offset, is one.
struct ConditionalPauli {
  PauliString pauli;
  std::size_t num_bits = 0;
  bool offset = false;
};

/// A diagonal unitary on n qubits, by its 2^n diagonal entries.
struct DiagonalUnitary {
  /// Throws std::invalid_argument unless there are 2^n entries.
  explicit DiagonalUnitary(std::vector<std::complex<double>> diagonal);
  std::size_t size() const { return n; }
  const std::vector<std::complex<double>> &get_diagonal() const {
    return diagonal;
  }

private:
  std::vector<std::complex<double>> diagonal;
  std::size_t n;
};

/// A unitary on n qubits, by its row-major 2^n x 2^n matrix.
struct Unitary {
  /// Throws std::invalid_argument unless there are 4^n entries.
  explicit Unitary(std::vector<std::complex<double>> matrix);
  std::size_t size() const { return n; }
  const std::vector<std::complex<double>> &get_matrix() const { return matrix; }

private:
  std::vector<std::complex<double>> matrix;
  std::size_t n;
};

using WireId = std::string;

/// Qubit and classical input and output wires, of a node or a circuit.
struct Wires {
  Wires(std::vector<WireId> q_in = {}, std::vector<WireId> q_out = {},
        std::vector<WireId> c_in = {}, std::vector<WireId> c_out = {})
      : q_in(std::move(q_in)), q_out(std::move(q_out)), c_in(std::move(c_in)),
        c_out(std::move(c_out)) {}

  std::vector<WireId> q_in;
  std::vector<WireId> q_out;
  std::vector<WireId> c_in;
  std::vector<WireId> c_out;

  Shape get_shape() const {
    return {q_in.size(), q_out.size(), c_in.size(), c_out.size()};
  }
};

} // namespace v0
} // namespace cudaq::gadgets
