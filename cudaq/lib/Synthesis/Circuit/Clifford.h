/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <iostream>

#include "cudaq/Synthesis/Circuit/Circuit.h"
#include <cstdint>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Axis
//===----------------------------------------------------------------------===//

/// Coset representative label for the Matsumoto-Amano normal form.
///
/// The single-qubit Clifford group splits into three cosets of the subgroup
/// S = <X, S, omega>:
///
///     Cliff = I*S  U  H*S  U  SH*S
///
/// `Axis` names the three coset representatives and is also the syllable
/// prefix selector used by `Clifford::decompose_tconj`: the `a` exponent of
/// the (a, b, c, d) Clifford parametrization picks the coset directly:
///     I  -> a = 0  (element lies in S itself)
///     H  -> a = 1  (Hadamard coset)
///     SH -> a = 2  (S*H coset)
enum class Axis { I = 0, H = 1, SH = 2 };

//===----------------------------------------------------------------------===//
// Lookup tables for Clifford-group arithmetic
//===----------------------------------------------------------------------===//
//
// The single-qubit Clifford group modulo global phase has 24 elements; once
// the eight powers of omega (the global phase factor) are included, the full
// group has 192 elements. Every element is written uniquely as
//
//     E^a * X^b * S^c * omega^d
//
// with a in {0, 1, 2}, b in {0, 1}, c in {0, 1, 2, 3}, d in {0, ..., 7}.
//
// The four tables below precompute the answers to the conjugation and
// inversion subproblems that the (a, b, c, d) arithmetic needs, giving
// constant-time group multiplication and inversion:
//
//   CONJ2_TABLE[2c + b]
//       Conjugate X^b past S^c. Encodes
//           S^-c * X^b * S^c = X^b * S^c' * omega^d'
//       returning (c', d'). The X-exponent is preserved because X maps to
//       X under S-conjugation.
//
//   CONJ3_TABLE[8a + 4b + c]
//       Conjugate X^b * S^c past E^a. Encodes
//           E^-a * X^b * S^c * E^a = E^a' * X^b' * S^c' * omega^d'
//       returning (a', b', c', d').
//
//   CINV_TABLE[8a + 4b + c]
//       Invert E^a * X^b * S^c (i.e. ignoring omega^d). Encodes
//           (E^a * X^b * S^c)^-1 = E^a' * X^b' * S^c' * omega^d'
//       returning (a', b', c', d'); the caller is responsible for the
//       omega^-d step.
//
//   TCONJ_TABLE[2a + b]
//       Matsumoto-Amano syllable lookup. For a trailing Clifford with
//       parameters (a, b, _, _), records which syllable type results from
//       appending T:
//           axis in {I, H, SH} <-> syllable in {T, HT, SHT}
//       together with the (c', d') deltas that are added to the existing
//       (c, d) of the Clifford. Only (a, b) enter the lookup.

// clang-format off
inline constexpr std::array<std::pair<int32_t, int32_t>, 8> CONJ2_TABLE = {{
  {0, 0},
  {0, 0},
  {1, 0},
  {3, 2},
  {2, 0},
  {2, 4},
  {3, 0},
  {1, 6}
}};

inline constexpr std::array<std::array<int32_t, 4>, 24> CONJ3_TABLE = {{
  {0, 0, 0, 0},
  {0, 0, 1, 0},
  {0, 0, 2, 0},
  {0, 0, 3, 0},
  {0, 1, 0, 0},
  {0, 1, 1, 0},
  {0, 1, 2, 0},
  {0, 1, 3, 0},
  {1, 0, 0, 0},
  {2, 0, 3, 6},
  {1, 1, 2, 2},
  {2, 1, 3, 6},
  {1, 0, 2, 0},
  {2, 1, 1, 0},
  {1, 1, 0, 6},
  {2, 0, 1, 4},
  {2, 0, 0, 0},
  {1, 1, 3, 4},
  {2, 1, 0, 0},
  {1, 0, 1, 2},
  {2, 1, 2, 2},
  {1, 1, 1, 0},
  {2, 0, 2, 6},
  {1, 0, 3, 2}
}};

inline constexpr std::array<std::array<int32_t, 4>, 24> CINV_TABLE = {{
  {0, 0, 0, 0},
  {0, 0, 3, 0},
  {0, 0, 2, 0},
  {0, 0, 1, 0},
  {0, 1, 0, 0},
  {0, 1, 1, 6},
  {0, 1, 2, 4},
  {0, 1, 3, 2},
  {2, 0, 0, 0},
  {1, 0, 1, 2},
  {2, 1, 0, 0},
  {1, 1, 3, 4},
  {2, 1, 1, 2},
  {1, 1, 1, 6},
  {2, 0, 2, 2},
  {1, 0, 3, 4},
  {1, 0, 0, 0},
  {2, 1, 3, 6},
  {1, 1, 2, 2},
  {2, 0, 3, 6},
  {1, 0, 2, 0},
  {2, 1, 1, 6},
  {1, 1, 0, 2},
  {2, 0, 1, 6}
}};

inline constexpr std::array<std::array<int32_t, 3>, 6> TCONJ_TABLE = {{
  {static_cast<int32_t>(Axis::I), 0, 0},
  {static_cast<int32_t>(Axis::I), 1, 7},
  {static_cast<int32_t>(Axis::H), 3, 3},
  {static_cast<int32_t>(Axis::H), 2, 0},
  {static_cast<int32_t>(Axis::SH), 0, 5},
  {static_cast<int32_t>(Axis::SH), 1, 4}
}};
// clang-format on

//===----------------------------------------------------------------------===//
// Clifford
//===----------------------------------------------------------------------===//

/// An element of the single-qubit Clifford group (including global phase).
///
/// References:
///   - Giles & Selinger, "Remarks on Matsumoto and Amano's normal form for
///     single-qubit Clifford+T operators", arXiv:1312.6584 (2013).
///   - Ross & Selinger, "Optimal ancilla-free Clifford+T approximation of
///     z-rotations", arXiv:1403.2975 (2014).
///
/// Parametrization. Every element is written uniquely as
///     C = E^a * X^b * S^c * omega^d
/// with the generators
///     E     -- order-3 element; a in {0, 1, 2}. H = E*S*omega^5 gives
///              E = H * S^-1 * omega^-5 = H * S^3 * omega^3.
///     X     -- Pauli-X bit-flip; b in {0, 1}, order 2.
///     S     -- phase gate diag(1, i); c in {0, 1, 2, 3}, order 4.
///     omega -- e^{i*pi/4} global phase; d in {0, ..., 7}, order 8.
/// The product 3 * 2 * 4 * 8 = 192 is the full group order; quotienting by
/// the global phase <omega> gives the 24-element Cliff/<omega> ~= S_4.
///
/// Matsumoto-Amano hook. The three cosets of the subgroup <X, S, omega> are
/// indexed by `a` and correspond to the syllables T, HT, SHT in the normal
/// form:
///     a = 0 -> coset rep I,  prefix empty -> syllable T
///     a = 1 -> coset rep H,  prefix H     -> syllable HT
///     a = 2 -> coset rep SH, prefix S*H   -> syllable SHT
/// `decompose_coset()` extracts the prefix; `decompose_tconj()` produces the
/// updated trailing Clifford after appending a T gate.
///
/// All arithmetic goes through the precomputed lookup tables and is
/// constexpr.
class Clifford {
private:
  int32_t _a, _b, _c, _d;

  /// Bring the four exponents back into their canonical ranges after an
  /// arithmetic step that may have left them unreduced.
  constexpr void normalize() {
    _a = (_a % 3 + 3) % 3; // 0 <= a < 3
    _b = _b & 1;           // 0 <= b < 2
    _c = _c & 0b11;        // 0 <= c < 4
    _d = _d & 0b111;       // 0 <= d < 8
  }

public:
  //===--------------------------------------------------------------------===//
  // Construction
  //===--------------------------------------------------------------------===//

  /// Defaults to the identity (0, 0, 0, 0). Inputs are reduced mod their
  /// respective range so callers do not have to pre-normalize.
  constexpr Clifford(int32_t a = 0, int32_t b = 0, int32_t c = 0, int32_t d = 0)
      : _a(a), _b(b), _c(c), _d(d) {
    normalize();
  }

  //===--------------------------------------------------------------------===//
  // Accessors
  //===--------------------------------------------------------------------===//

  /// Component `a` of the (a, b, c, d) parametrization. Also the Axis index
  /// of the Matsumoto-Amano coset: 0 -> I, 1 -> H, 2 -> SH.
  constexpr int32_t a() const { return _a; }

  /// Component `b`: Pauli-X exponent in {0, 1}.
  constexpr int32_t b() const { return _b; }

  /// Component `c`: phase-gate exponent in {0, 1, 2, 3}.
  constexpr int32_t c() const { return _c; }

  /// Component `d`: omega global-phase exponent in {0, ..., 7}.
  constexpr int32_t d() const { return _d; }

  //===--------------------------------------------------------------------===//
  // I/O
  //===--------------------------------------------------------------------===//

  /// Stream as "E^a X^b S^c omega^d" for debug logging.
  friend std::ostream &operator<<(std::ostream &os, const Clifford &cliff) {
    os << "E^" << cliff._a << " X^" << cliff._b << " S^" << cliff._c
       << " omega^" << cliff._d;
    return os;
  }

  //===--------------------------------------------------------------------===//
  // Conversion
  //===--------------------------------------------------------------------===//

  /// Embed a Clifford `Gate` enumerator into the (a, b, c, d) parametrization.
  /// T is not a Clifford gate; passing Gate::T asserts in debug builds.
  static Clifford from_gate(Gate g) {
    switch (g) {
    case Gate::H:
      return Clifford(1, 0, 1, 5);
    case Gate::S:
      return Clifford(0, 0, 1, 0);
    case Gate::X:
      return Clifford(0, 1, 0, 0);
    case Gate::W:
      return Clifford(0, 0, 0, 1);
    default:
      assert(false && "Gate::T is not a Clifford gate");
      return Clifford(0, 0, 0, 0);
    }
  }

  //===--------------------------------------------------------------------===//
  // Comparison
  //===--------------------------------------------------------------------===//

  constexpr bool operator==(const Clifford &other) const {
    return _a == other._a && _b == other._b && _c == other._c && _d == other._d;
  }

  constexpr bool operator!=(const Clifford &other) const {
    return !(*this == other);
  }

  //===--------------------------------------------------------------------===//
  // Table-lookup helpers
  //===--------------------------------------------------------------------===//

  /// Conjugate X^b past S^c. See CONJ2_TABLE's header comment.
  static constexpr std::pair<int32_t, int32_t> conj2(int32_t c, int32_t b) {
    int32_t index = (c << 1) | b;
    return CONJ2_TABLE[static_cast<size_t>(index)];
  }

  /// Conjugate X^b * S^c past E^a. See CONJ3_TABLE's header comment.
  static constexpr std::array<int32_t, 4> conj3(int32_t b, int32_t c, int32_t a) {
    int32_t index = (a << 3) | (b << 2) | c;
    return CONJ3_TABLE[static_cast<size_t>(index)];
  }

  /// Invert E^a * X^b * S^c (omega^d is handled by the caller). See
  /// CINV_TABLE's header comment.
  static constexpr std::array<int32_t, 4> cinv(int32_t a, int32_t b, int32_t c) {
    int32_t index = (a << 3) | (b << 2) | c;
    return CINV_TABLE[static_cast<size_t>(index)];
  }

  /// Matsumoto-Amano syllable lookup for C * T. See TCONJ_TABLE's header
  /// comment.
  static constexpr std::array<int32_t, 3> tconj(int32_t a, int32_t b) {
    int32_t index = (a << 1) | b;
    return TCONJ_TABLE[static_cast<size_t>(index)];
  }

  //===--------------------------------------------------------------------===//
  // Group arithmetic
  //===--------------------------------------------------------------------===//

  /// Group multiplication via the conjugation tables. O(1), constexpr.
  ///
  /// Computes (E^a X^b S^c omega^d) * (E^a' X^b' S^c' omega^d') by walking
  /// the inner generators past each other:
  ///   1. push X^b * S^c past E^a' via CONJ3_TABLE;
  ///   2. push the resulting S-power past X^b' via CONJ2_TABLE;
  ///   3. accumulate the new exponents and the carry omega phases.
  constexpr Clifford operator*(const Clifford &other) const {
    auto [a1, b1, c1, d1] = conj3(_b, _c, other._a);
    auto [c2, d2] = conj2(c1, other._b);

    int32_t new_a = _a + a1;
    int32_t new_b = b1 + other._b;
    int32_t new_c = c2 + other._c;
    int32_t new_d = d2 + d1 + _d + other._d;

    return Clifford(new_a, new_b, new_c, new_d);
  }

  /// Group inverse via CINV_TABLE. O(1), constexpr.
  constexpr Clifford inv() const {
    auto [a1, b1, c1, d1] = cinv(_a, _b, _c);
    return Clifford(a1, b1, c1, d1 - _d);
  }

  /// Split C as `rep * remainder` where rep is the coset representative
  /// (one of I, H, SH) determined by `a`, and remainder lies entirely in
  /// the subgroup <X, S, omega> (i.e. has a = 0). Used by `to_circuit()`
  /// for gate serialization and by `normalize_gates()` for syllable
  /// classification.
  ///
  /// Defined out of line because it references CLIFFORD_H / CLIFFORD_SH.
  constexpr std::pair<Axis, Clifford> decompose_coset() const;

  /// Matsumoto-Amano syllable extraction: write C * T as
  ///     syllable * C'
  /// where the syllable is one of T (axis I), HT (axis H), SHT (axis SH),
  /// and C' is the new trailing Clifford context. The syllable type is a
  /// function of (a, b) only; the (c, d) deltas from the table are added
  /// onto the existing (c, d) of C.
  ///
  /// This is the core step of `normalize_gates()`.
  constexpr std::pair<Axis, Clifford> decompose_tconj() const {
    auto tconj_result = tconj(_a, _b);
    Axis axis = static_cast<Axis>(tconj_result[0]);
    int32_t c1 = tconj_result[1];
    int32_t d1 = tconj_result[2];

    return {axis, Clifford(0, _b, c1 + _c, d1 + _d)};
  }

  /// Emit this Clifford as a circuit of elementary gates by composing the
  /// coset prefix with the X / S / W generators implied by (b, c, d):
  ///     a = 0 ->                X^b * S^c * W^d
  ///     a = 1 -> H *            X^b * S^c * W^d
  ///     a = 2 -> S * H *        X^b * S^c * W^d
  Circuit to_circuit() const {
    auto [axis, c] = decompose_coset();

    Circuit result;
    if (axis == Axis::H)
      result.push_back(Gate::H);
    else if (axis == Axis::SH) {
      result.push_back(Gate::S);
      result.push_back(Gate::H);
    }

    for (int i = 0; i < c.b(); ++i)
      result.push_back(Gate::X);
    for (int i = 0; i < c.c(); ++i)
      result.push_back(Gate::S);
    for (int i = 0; i < c.d(); ++i)
      result.push_back(Gate::W);

    return result;
  }
};

//===----------------------------------------------------------------------===//
// Predefined Clifford elements
//===----------------------------------------------------------------------===//
//
// Generators:
//   I = (0, 0, 0, 0)    Identity.
//   X = (0, 1, 0, 0)    Pauli-X bit-flip.
//   H = (1, 0, 1, 5)    Hadamard.
//   S = (0, 0, 1, 0)    Phase gate diag(1, i).
//   W = (0, 0, 0, 1)    Global phase omega = e^{i*pi/4}.
//
// Composite Cliffords used by the Matsumoto-Amano machinery:
//   SH  = S * H        (2, 0, 0, 3)  Coset representative for a = 2.
//   HS  = H * S        (1, 0, 2, 5)  Absorbed when merging HT * T syllables.
//   SHS = S * H * S    (2, 0, 1, 3)  Absorbed when merging SHT * T syllables.

inline constexpr Clifford CLIFFORD_I(0, 0, 0, 0);
inline constexpr Clifford CLIFFORD_X(0, 1, 0, 0);
inline constexpr Clifford CLIFFORD_H(1, 0, 1, 5);
inline constexpr Clifford CLIFFORD_S(0, 0, 1, 0);
inline constexpr Clifford CLIFFORD_W(0, 0, 0, 1);
inline constexpr Clifford CLIFFORD_SH = CLIFFORD_S * CLIFFORD_H;
inline constexpr Clifford CLIFFORD_HS = CLIFFORD_H * CLIFFORD_S;
inline constexpr Clifford CLIFFORD_SHS = CLIFFORD_S * CLIFFORD_H * CLIFFORD_S;

constexpr std::pair<Axis, Clifford> Clifford::decompose_coset() const {
  if (_a == 0)
    return {Axis::I, *this};

  if (_a == 1)
    return {Axis::H, CLIFFORD_H.inv() * (*this)};

  // _a == 2
  return {Axis::SH, CLIFFORD_SH.inv() * (*this)};
}

//===----------------------------------------------------------------------===//
// Compile-time verification of Clifford-group identities
//===----------------------------------------------------------------------===//

namespace {
static_assert(CLIFFORD_S * CLIFFORD_S * CLIFFORD_S * CLIFFORD_S == CLIFFORD_I,
              "S^4 = I");
static_assert((CLIFFORD_H * CLIFFORD_H).a() == 0 &&
                  (CLIFFORD_H * CLIFFORD_H).b() == 0 &&
                  (CLIFFORD_H * CLIFFORD_H).c() == 0,
              "H^2 = I (mod phase)");
static_assert(CLIFFORD_S * CLIFFORD_S.inv() == CLIFFORD_I, "S * S^-1 = I");

// Coordinates of the composite Cliffords match the tables above.
static_assert(CLIFFORD_SH.a() == 2 && CLIFFORD_SH.b() == 0 &&
                  CLIFFORD_SH.c() == 0 && CLIFFORD_SH.d() == 3,
              "SH = (2,0,0,3)");
static_assert(CLIFFORD_HS.a() == 1 && CLIFFORD_HS.b() == 0 &&
                  CLIFFORD_HS.c() == 2 && CLIFFORD_HS.d() == 5,
              "HS = (1,0,2,5)");
static_assert(CLIFFORD_SHS.a() == 2 && CLIFFORD_SHS.b() == 0 &&
                  CLIFFORD_SHS.c() == 1 && CLIFFORD_SHS.d() == 3,
              "SHS = (2,0,1,3)");
} // namespace

} // namespace cudaq::synth
