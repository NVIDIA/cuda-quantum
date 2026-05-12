/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Circuit/PauliProduct.h"
#include "Support/BitVector.h"

#include <cassert>
#include <cstddef>
#include <vector>

namespace cudaq::synth {

// ---------------------------------------------------------------------------
// Tableau
// ---------------------------------------------------------------------------
//
// Aaronson-Gottesman Clifford tableau for n-qubit stabilizer simulation,
// extended to support T-stabilizer generator columns for PBC compilation.
//
// ## Data layout (column-major)
//
// The tableau stores num_generators_ Pauli generators. The first 2n are the
// standard stabilizer/destabilizer pair; additional columns are T-stabs.
//
//   z_[i] : BitVector of length (2n + k)  -- Z-components of qubit i
//   x_[i] : BitVector of length (2n + k)  -- X-components of qubit i
//   signs_ : BitVector of length (2n + k) -- sign bit of each generator
//
// Column layout:
//   [0 .. n-1]       stabilizers     --> emitted as pbc.pp_measure
//   [n .. 2n-1]      destabilizers   (kept for Clifford formalism)
//   [2n .. 2n+k-1]   T-stab columns  --> emitted as pbc.pp_rotation
//
// The initial state |0...0> has:
//   stabilizers  Z_i : z_[i].get(i) = 1, all others 0
//   destabilizers X_i : x_[i].get(i + n) = 1, all others 0
//
// ## Append vs prepend
//
// append_*(g): conjugate all generators by g on the right (U -> U*g).
//   These are bulk SIMD operations on entire row BitVectors -- O(n/256) per
//   call. Importantly, they operate across ALL generator columns, so T-stab
//   columns are correctly conjugated by subsequent Clifford gates.
//
// prepend_*(g): multiply all generators by g on the left (U -> g*U).
//   Implemented via column extract/insert (PauliProduct multiply) -- O(n) per
//   call.

class Tableau {
public:
  // -- Construction --

  /// Construct a tableau for `num_qubits` qubits, pre-allocating capacity for
  /// `num_extra_generators` additional T-stabilizer columns beyond the
  /// standard 2n stabilizer/destabilizer columns.
  explicit Tableau(size_t num_qubits, size_t num_extra_generators = 0)
      : num_qubits_(num_qubits), num_generators_(num_qubits << 1),
        z_(init_z(num_qubits, num_extra_generators)),
        x_(init_x(num_qubits, num_extra_generators)),
        signs_((num_qubits << 1) + num_extra_generators) {}

  size_t num_qubits() const noexcept { return num_qubits_; }
  size_t num_generators() const noexcept { return num_generators_; }
  size_t num_t_stabs() const noexcept {
    return num_generators_ - (num_qubits_ << 1);
  }

  const std::vector<BitVector> &z() const noexcept { return z_; }
  const std::vector<BitVector> &x() const noexcept { return x_; }
  const BitVector &signs() const noexcept { return signs_; }

  // -- Append operations (right-multiply by gate) --
  //
  // These update every generator column in one pass using SIMD bulk BitVector
  // ops. This includes T-stab columns, which is the key property that makes
  // the extended tableau correct for PBC compilation.

  /// Conjugate all generators by X on qubit q: P -> X_q * P * X_q
  /// Effect: Z_q component flips sign iff Z_q is present (XZ = -ZX).
  void append_x(size_t qubit) {
    assert(qubit < num_qubits_);
    signs_ ^= z_[qubit];
  }

  /// Conjugate all generators by Y on qubit q.
  /// Y: X -> -X, Z -> -Z (both anti-commute with Y).
  void append_y(size_t qubit) {
    assert(qubit < num_qubits_);
    signs_ ^= x_[qubit] ^ z_[qubit];
  }

  /// Conjugate all generators by Z on qubit q.
  void append_z(size_t qubit) {
    assert(qubit < num_qubits_);
    signs_ ^= x_[qubit];
  }

  /// Conjugate all generators by S on qubit q (S = diag(1, i)).
  /// S: X -> Y, Y -> -X, Z -> Z.
  void append_s(size_t qubit) {
    assert(qubit < num_qubits_);
    signs_ ^= z_[qubit] & x_[qubit];
    z_[qubit] ^= x_[qubit];
  }

  /// Conjugate all generators by S† (Sdg) on qubit q.
  /// Sdg: X -> -Y, Y -> X, Z -> Z.
  void append_sdg(size_t qubit) {
    assert(qubit < num_qubits_);
    signs_ ^= x_[qubit] ^ (x_[qubit] & z_[qubit]);
    z_[qubit] ^= x_[qubit];
  }

  /// Conjugate all generators by V = S†H = sqrt(X) on qubit q.
  /// V: Z -> X, X -> -Z, Y -> Y.
  void append_v(size_t qubit) {
    assert(qubit < num_qubits_);
    signs_ ^= ~x_[qubit] & z_[qubit];
    x_[qubit] ^= z_[qubit];
  }

  /// Conjugate all generators by H on qubit q.
  /// Decomposed as S*V*S.
  void append_h(size_t qubit) {
    append_s(qubit);
    append_v(qubit);
    append_s(qubit);
  }

  /// Conjugate all generators by CX (CNOT) with given control and target
  /// qubits.
  ///
  /// CX action on Pauli generators:
  ///   Z_ctrl -> Z_ctrl * Z_targ
  ///   X_targ -> X_ctrl * X_targ
  ///   Phase: sign flips when X_ctrl=1, Z_targ=1, and (Z_ctrl XOR X_targ) = 1.
  void append_cx(size_t control, size_t target) {
    assert(control < num_qubits_ && target < num_qubits_ && control != target);
    // Phase correction: signs ^= x[ctrl] & z[targ] & (~z[ctrl] ^ x[targ])
    signs_ ^= x_[control] & z_[target] & (~z_[control] ^ x_[target]);
    z_[control] ^= z_[target];
    x_[target] ^= x_[control];
  }

  /// Conjugate all generators by CZ on the two qubits.
  /// Decomposed as: S(c) S(t) CX(c,t) S(t) Z(t) CX(c,t).
  void append_cz(size_t control, size_t target) {
    assert(control < num_qubits_ && target < num_qubits_ && control != target);
    append_s(control);
    append_s(target);
    append_cx(control, target);
    append_s(target);
    append_z(target);
    append_cx(control, target);
  }

  // -- T-stabilizer management --

  /// Append a new T-stab generator column with Z on `qubit`. If `adj` is
  /// true (Tdg), the sign bit is set (negative Pauli).
  ///
  /// The new column is written at index `num_generators_`, which must be
  /// within the pre-allocated capacity. Subsequent `append_*` calls will
  /// conjugate this column along with all other generators.
  void add_t_stab(size_t qubit, bool adj = false) {
    assert(qubit < num_qubits_);
    assert(num_generators_ < signs_.size() && "T-stab budget exceeded");
    z_[qubit].xor_bit(num_generators_);
    if (adj)
      signs_.xor_bit(num_generators_);
    ++num_generators_;
  }

  /// Append an arbitrary generator column from a PauliProduct. Used to inject
  /// pre-computed T-stab rows (e.g. for Toffoli decomposition).
  void add_stab(const PauliProduct &p) {
    assert(num_generators_ < signs_.size() && "T-stab budget exceeded");
    for (size_t i = 0; i < num_qubits_; ++i) {
      if (p.z().get(i))
        z_[i].xor_bit(num_generators_);
      if (p.x().get(i))
        x_[i].xor_bit(num_generators_);
    }
    if (p.sign())
      signs_.xor_bit(num_generators_);
    ++num_generators_;
  }

  // -- Prepend operations (left-multiply by gate) --
  //
  // These operate column-by-column via PauliProduct extract/insert.

  /// Left-multiply all generators by X on qubit q.
  void prepend_x(size_t qubit) {
    assert(qubit < num_qubits_);
    signs_.xor_bit(qubit);
  }

  /// Left-multiply all generators by Z on qubit q.
  void prepend_z(size_t qubit) {
    assert(qubit < num_qubits_);
    signs_.xor_bit(qubit + num_qubits_);
  }

  /// Left-multiply all generators by S on qubit q.
  /// For each column: destabilizer = destabilizer * stabilizer.
  void prepend_s(size_t qubit) {
    assert(qubit < num_qubits_);
    PauliProduct stab = extract_pauli_product(qubit);
    PauliProduct destab = extract_pauli_product(qubit + num_qubits_);
    destab *= stab;
    insert_pauli_product(destab, qubit + num_qubits_);
  }

  /// Left-multiply all generators by H on qubit q.
  /// Swaps the stabilizer and destabilizer columns.
  void prepend_h(size_t qubit) {
    assert(qubit < num_qubits_);
    PauliProduct stab = extract_pauli_product(qubit);
    PauliProduct destab = extract_pauli_product(qubit + num_qubits_);
    insert_pauli_product(destab, qubit);
    insert_pauli_product(stab, qubit + num_qubits_);
  }

  /// Left-multiply all generators by CX with given control and target.
  void prepend_cx(size_t control, size_t target) {
    assert(control < num_qubits_ && target < num_qubits_ && control != target);
    PauliProduct stab_ctrl = extract_pauli_product(control);
    PauliProduct stab_targ = extract_pauli_product(target);
    PauliProduct destab_ctrl = extract_pauli_product(control + num_qubits_);
    PauliProduct destab_targ = extract_pauli_product(target + num_qubits_);
    stab_targ *= stab_ctrl;
    destab_ctrl *= destab_targ;
    insert_pauli_product(stab_targ, target);
    insert_pauli_product(destab_ctrl, control + num_qubits_);
  }

  // -- Column access --

  /// Extract the Pauli generator stored in column `col` as a PauliProduct.
  /// Valid for any col in [0, num_generators_). O(n) bit reads.
  PauliProduct extract_pauli_product(size_t col) const {
    assert(col < num_generators_);
    BitVector z(num_qubits_);
    BitVector x(num_qubits_);
    for (size_t i = 0; i < num_qubits_; ++i) {
      if (z_[i].get(col))
        z.xor_bit(i);
      if (x_[i].get(col))
        x.xor_bit(i);
    }
    return PauliProduct(std::move(z), std::move(x), signs_.get(col));
  }

  /// Write a PauliProduct into column `col`, XOR-merging with any existing
  /// bits. Bits in `p` that differ from the current column are flipped.
  /// Valid for any col in [0, num_generators_). O(n) bit reads/writes.
  void insert_pauli_product(const PauliProduct &p, size_t col) {
    assert(col < num_generators_);
    for (size_t i = 0; i < num_qubits_; ++i) {
      if (p.z().get(i) != z_[i].get(col))
        z_[i].xor_bit(col);
      if (p.x().get(i) != x_[i].get(col))
        x_[i].xor_bit(col);
    }
    if (p.sign() != signs_.get(col))
      signs_.xor_bit(col);
  }

private:
  size_t num_qubits_;
  size_t num_generators_;
  std::vector<BitVector> z_;
  std::vector<BitVector> x_;
  BitVector signs_;

  static std::vector<BitVector> init_z(size_t num_qubits,
                                       size_t num_extra_generators) {
    std::vector<BitVector> vec;
    vec.reserve(num_qubits);
    const size_t total = (num_qubits << 1) + num_extra_generators;
    for (size_t i = 0; i < num_qubits; ++i) {
      BitVector bv(total);
      bv.xor_bit(i);
      vec.push_back(std::move(bv));
    }
    return vec;
  }

  static std::vector<BitVector> init_x(size_t num_qubits,
                                       size_t num_extra_generators) {
    std::vector<BitVector> vec;
    vec.reserve(num_qubits);
    const size_t total = (num_qubits << 1) + num_extra_generators;
    for (size_t i = 0; i < num_qubits; ++i) {
      BitVector bv(total);
      bv.xor_bit(i + num_qubits);
      vec.push_back(std::move(bv));
    }
    return vec;
  }
};

} // namespace cudaq::synth
