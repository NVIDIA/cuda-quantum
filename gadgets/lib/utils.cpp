/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#include "cudaq/gadgets/utils.h"

#include <stdexcept>

namespace cudaq::gadgets {
inline namespace v0 {

char to_char(Pauli pauli) {
  switch (pauli) {
  case Pauli::I:
    return 'I';
  case Pauli::X:
    return 'X';
  case Pauli::Y:
    return 'Y';
  case Pauli::Z:
    return 'Z';
  }
  throw std::invalid_argument("invalid Pauli");
}

Pauli pauli_from_char(char c) {
  switch (c) {
  case 'I':
  case '_':
    return Pauli::I;
  case 'X':
    return Pauli::X;
  case 'Y':
    return Pauli::Y;
  case 'Z':
    return Pauli::Z;
  default:
    throw std::invalid_argument(std::string("invalid Pauli character '") + c +
                                "'");
  }
}

std::string PauliString::to_string() const {
  std::string result(phase_prefixes[phase_exponent % 4]);
  for (Pauli op : ops)
    result += to_char(op);
  return result;
}

PauliString PauliString::from_string(std::string_view text) {
  PauliString result;
  // Two-character prefixes first, so "+i" is not read as "+".
  for (std::uint8_t exponent : {1, 3, 0, 2}) {
    std::string_view prefix = phase_prefixes[exponent];
    if (text.substr(0, prefix.size()) == prefix) {
      result.phase_exponent = exponent;
      text.remove_prefix(prefix.size());
      break;
    }
  }
  for (char c : text)
    result.ops.push_back(pauli_from_char(c));
  return result;
}

namespace {
/// n such that size == 2^(bits * n); throws std::invalid_argument otherwise.
std::size_t qubits(std::size_t size, std::size_t bits) {
  std::size_t n = 0;
  while ((std::size_t{1} << (bits * n)) < size)
    ++n;
  if ((std::size_t{1} << (bits * n)) != size)
    throw std::invalid_argument("unitary size does not fit whole qubits");
  return n;
}
} // namespace

DiagonalUnitary::DiagonalUnitary(std::vector<std::complex<double>> diagonal)
    : diagonal(std::move(diagonal)), n(qubits(this->diagonal.size(), 1)) {}

Unitary::Unitary(std::vector<std::complex<double>> matrix)
    : matrix(std::move(matrix)), n(qubits(this->matrix.size(), 2)) {}

} // namespace v0
} // namespace cudaq::gadgets
