/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Support/BitVector.h"

#include <cstdint>
#include <vector>

namespace cudaq::synth {

// ---------------------------------------------------------------------------
// PauliProduct
// ---------------------------------------------------------------------------
//
// An n-qubit Pauli product in the symplectic representation:
//
//   qubit i carries Pauli factor:
//     (z[i]=0, x[i]=0) → I
//     (z[i]=1, x[i]=0) → Z
//     (z[i]=0, x[i]=1) → X
//     (z[i]=1, x[i]=1) → Y  (up to phase)
//
//   sign = true → the overall phase is −1 (i.e. the operator is negated).
//
// Internally we keep z and x as separate BitVectors so that bulk operations
// (AND, XOR, popcount) map to direct calls on contiguous BitVector data with
// no deinterleaving needed. This is the most SIMD-friendly layout for the
// hot-path commutation check and multiplication.

class PauliProduct {
public:
  // -- Construction --

  PauliProduct(BitVector z, BitVector x, bool sign) noexcept
      : z_(std::move(z)), x_(std::move(x)), sign_(sign) {}

  // -- Accessors --

  const BitVector &z() const noexcept { return z_; }
  const BitVector &x() const noexcept { return x_; }
  bool sign() const noexcept { return sign_; }

  // -- Commutation --

  /// True iff this Pauli product commutes with `other`.
  ///
  /// Two Pauli products P1 and P2 commute iff the symplectic inner product
  ///   Λ(P1, P2) = popcount((x1 & z2) ^ (z1 & x2)) mod 2
  /// equals 0. This is O(N/256) bulk SIMD operations over all blocks.
  bool is_commuting(const PauliProduct &other) const {
    BitVector ac = (x_ & other.z_) ^ (z_ & other.x_);
    return ac.popcount() % 2 == 0;
  }

  // -- Multiplication --

  /// In-place Pauli product multiplication: *this = (*this) * rhs.
  ///
  /// Uses the standard stabilizer-formalism phase formula. Phase accumulation
  /// follows from the relation XZ = -iY: each qubit where (z1=1, x2=1) or
  /// (x1=1, z2=1) contributes factors of i or -i, tracked via the `ac` and
  /// `x1z2` auxiliary bitvectors.
  PauliProduct &operator*=(const PauliProduct &rhs) {
    // x1z2: qubits where self has Z and rhs has X (contributes i)
    BitVector x1z2 = z_ & rhs.x_;
    // ac: qubits where the two Paulis anti-commute locally
    BitVector ac = (x_ & rhs.z_) ^ x1z2;

    // Update x and z components.
    x_ ^= rhs.x_;
    z_ ^= rhs.z_;

    // Count the i^k phase contributions:
    //   ac counts qubits contributing i^1
    //   x1z2 & (new_x ^ new_z) counts additional i^2 contributions
    x1z2 ^= x_;
    x1z2 ^= z_;
    x1z2 &= ac;

    const int32_t phase = ac.popcount() + 2 * x1z2.popcount();
    sign_ ^= rhs.sign_ ^ ((phase % 4) > 1);
    return *this;
  }

  friend PauliProduct operator*(PauliProduct lhs, const PauliProduct &rhs) {
    return lhs *= rhs;
  }

  // -- Conversion --

  /// Serialize to a boolean vector of length 2 * nb_qubits:
  ///   result = [z[0], z[1], ..., z[nb_qubits-1],
  ///             x[0], x[1], ..., x[nb_qubits-1]]
  ///
  /// Bits beyond nb_qubits in the underlying BitVector are silently dropped.
  std::vector<bool> to_bool_vec(size_t nb_qubits) const {
    std::vector<bool> z_bits = z_.to_bool_vec();
    std::vector<bool> x_bits = x_.to_bool_vec();

    z_bits.resize(nb_qubits, false);
    x_bits.resize(nb_qubits, false);

    z_bits.insert(z_bits.end(), x_bits.begin(), x_bits.end());
    return z_bits;
  }

private:
  BitVector z_;
  BitVector x_;
  bool sign_;
};

} // namespace cudaq::synth
