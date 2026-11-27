/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/APInt.h"
#include <optional>

namespace cudaq {

/// A product i^phase X^x Z^z in a fixed qubit order. A set mask bit
/// selects a Pauli on that qubit. The phase exponent is modulo four.
struct PauliProduct {
  llvm::APInt x;
  llvm::APInt z;
  unsigned phase;

  explicit PauliProduct(unsigned width, unsigned phase = 0)
      : x(width, 0), z(width, 0), phase(phase & 3U) {}

  bool operator==(const PauliProduct &other) const {
    return x.getBitWidth() == other.x.getBitWidth() && x == other.x &&
           z == other.z && phase == other.phase;
  }

  /// Multiply two products in matrix order. Each left Z crossing a right X
  /// contributes a minus sign.
  PauliProduct operator*(const PauliProduct &right) const {
    PauliProduct result(x.getBitWidth());
    result.x = x ^ right.x;
    result.z = z ^ right.z;
    unsigned sign = (z & right.x).popcount() & 1U;
    result.phase = (phase + right.phase + 2U * sign) & 3U;
    return result;
  }

  /// Left-multiply by a Hermitian Pauli on one qubit in constant time.
  /// When both bits are set the multiplier is Y = i X Z.
  void leftMultiplyPauli(unsigned qubit, bool hasX, bool hasZ) {
    unsigned sign = hasZ && x[qubit];
    phase = (phase + (hasX && hasZ) + 2U * sign) & 3U;
    if (hasX)
      x.flipBit(qubit);
    if (hasZ)
      z.flipBit(qubit);
  }

  bool anticommutes(const PauliProduct &other) const {
    return ((x & other.z).popcount() + (z & other.x).popcount()) & 1U;
  }

  /// Return the sign bit relative to the tensor product of I, X, Y and Z,
  /// or no value if the product is anti-Hermitian.
  std::optional<bool> hermitianSign() const {
    unsigned yCount = (x & z).popcount();
    switch ((phase - yCount) & 3U) {
    case 0:
      return false;
    case 2:
      return true;
    default:
      return std::nullopt;
    }
  }
};

} // namespace cudaq
