/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <iostream>

#include "cudaq/Synthesis/Math/Types.h"
#include "cudaq/Synthesis/Circuit/Circuit.h"

namespace cudaq::synth {

/// Coset representative label for the Matsumoto-`Amano` normal form.
///
/// The single-qubit Clifford group decomposes into three `cosets` of the
/// subgroup S = ⟨X, S, ω⟩:
///
///   Cliff = I·S  ∪  H·S  ∪  SH·S
///
/// Each Axis value names one `coset` representative. The `a` parameter of
/// the (a,b,c,d) Clifford `parametrization` indexes these `cosets`:
///   I  → a = 0 (element is in the subgroup S itself)
///   H  → a = 1 (Hadamard `coset`)
///   SH → a = 2 (S·H `coset`)
///
/// These labels also identify the syllable prefixes in the Matsumoto-`Amano`
/// normal form: a T gate preceded by Clifford context C decomposes as
/// C·T = (prefix)T · C' where prefix ∈ {ε, H, SH} and C' is the updated
/// trailing Clifford. See `Clifford::decompose_tconj()`.
enum class Axis { I = 0, H = 1, SH = 2 };

// Lookup tables for Clifford group arithmetic.
//
// The Clifford group on one qubit (modulo global phase) has 24 elements.
// Including the 8 powers of ω (global phases), the full group has 192
// elements. Each element is uniquely written as E^a · X^b · S^c · ω^d
// where a ∈ {0,1,2}, b ∈ {0,1}, c ∈ {0,1,2,3}, d ∈ {0,...,7}.
//
// The tables below precompute conjugation and inversion results, enabling
// O(1) group multiplication and inversion.
//
// CONJ2_TABLE[2c + b]: conjugates X^b past S^c.
//   S^{-c} · X^b · S^c = X^b · S^{c'} · ω^{d'}
//   Returns (c', d'); b is preserved (X always maps to X under S-conjugation).
//
// CONJ3_TABLE[8a + 4b + c]: conjugates X^b · S^c past E^a.
//   E^{-a} · X^b · S^c · E^a → E^{a'} · X^{b'} · S^{c'} · ω^{d'}
//   Returns (a', b', c', d').
//
// CINV_TABLE[8a + 4b + c]: inverts E^a · X^b · S^c (ignoring ω^d).
//   (E^a · X^b · S^c)^{-1} = E^{a'} · X^{b'} · S^{c'} · ω^{d'}
//   Returns (a', b', c', d'). The caller subtracts d separately.
//
// TCONJ_TABLE[2a + b]: Matsumoto-`Amano` syllable decomposition.
//   For a trailing Clifford C with parameters (a,b,c,d), determines which
//   syllable type results from C·T:
//     axis ∈ {I, H, SH} selects syllable ∈ {T, HT, SHT}
//     (c', d') are the updated trailing Clifford parameters.
//   Only depends on (a, b), not (c, d), which are added by the caller.

// clang-format off
inline constexpr std::array<std::pair<i32, i32>, 8> CONJ2_TABLE = {{
  {0, 0},
  {0, 0},
  {1, 0},
  {3, 2},
  {2, 0},
  {2, 4},
  {3, 0},
  {1, 6}
}};

inline constexpr std::array<std::array<i32, 4>, 24> CONJ3_TABLE = {{
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

inline constexpr std::array<std::array<i32, 4>, 24> CINV_TABLE = {{
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

inline constexpr std::array<std::array<i32, 3>, 6> TCONJ_TABLE = {{
  {static_cast<i32>(Axis::I), 0, 0},
  {static_cast<i32>(Axis::I), 1, 7},
  {static_cast<i32>(Axis::H), 3, 3},
  {static_cast<i32>(Axis::H), 2, 0},
  {static_cast<i32>(Axis::SH), 0, 5},
  {static_cast<i32>(Axis::SH), 1, 4}
}};
// clang-format on

/// An element of the single-qubit Clifford group (including global phase).
///
/// References:
/// - Giles & Selinger, "Remarks on Matsumoto and `Amano`'s normal form for
///   single-qubit Clifford+T operators", arXiv:1312.6584 (2013).
/// - Ross & Selinger, "Optimal ancilla-free Clifford+T approximation of
///   z-rotations", arXiv:1403.2975 (2014).
///
/// ## Parametrization
///
/// Every element is uniquely written as C = E^a · X^b · S^c · ω^d where
///   E  = order-3 element (a ∈ {0,1,2}); derivable from H = E·S·ω⁵ as
///        E = HS⁻¹ω⁻⁵ = HS³ω³
///   X  = Pauli-X, bit-flip (b ∈ {0,1}, order 2)
///   S  = phase gate `diag`(1, i) (c ∈ {0,1,2,3}, order 4)
///   ω  = e^{iπ/4} global phase (d ∈ {0,...,7}, order 8)
///
/// This spans all 192 = 3·2·4·8 elements. Modulo global phase, the
/// quotient |Cliff/⟨ω⟩| = 24 ≅ S₄.
///
/// The Clifford group is the `normalizer` of the Pauli group in U(2) and
/// is generated by {H, S, ω}.
///
/// ## Matsumoto-`Amano` normal form support
///
/// The three `cosets` of the subgroup S = ⟨X, S, ω⟩ are indexed by `a`:
///   a = 0 → `coset` rep I,  syllable prefix ε   → syllable T
///   a = 1 → `coset` rep H,  syllable prefix H   → syllable HT
///   a = 2 → `coset` rep SH, syllable prefix S·H → syllable SHT
///
/// decompose_coset() extracts the `coset` representative and remainder.
/// `decompose_tconj()` determines which syllable type results from appending
/// T to a circuit ending in this Clifford (the core of normalize_gates).
///
/// All arithmetic is via precomputed lookup tables and is `constexpr`.
class Clifford {
private:
  i32 _a, _b, _c, _d;

  // Reduce each exponent to its canonical range after arithmetic.
  constexpr void normalize() {
    _a = (_a % 3 + 3) % 3; // 0 <= a < 3
    _b = _b & 1;           // 0 <= b < 2
    _c = _c & 0b11;        // 0 <= c < 4
    _d = _d & 0b111;       // 0 <= d < 8
  }

public:
  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  /// Construct from explicit exponents. All parameters are reduced to their
  /// canonical ranges by normalize(). Defaults to the identity element
  /// (0,0,0,0).
  constexpr Clifford(i32 a = 0, i32 b = 0, i32 c = 0, i32 d = 0)
      : _a(a), _b(b), _c(c), _d(d) {
    normalize();
  }

  // -------------------------------------------------------------------------
  // Accessors
  // -------------------------------------------------------------------------

  /// Exponent of E in the `parametrization` C = E^a · X^b · S^c · ω^d.
  /// Identifies the `coset`: 0 → I, 1 → H, 2 → SH.
  constexpr i32 a() const { return _a; }

  /// Exponent of Pauli-X (b ∈ {0, 1}).
  constexpr i32 b() const { return _b; }

  /// Exponent of S (c ∈ {0, 1, 2, 3}).
  constexpr i32 c() const { return _c; }

  /// Exponent of global phase ω (d ∈ {0, ..., 7}).
  constexpr i32 d() const { return _d; }

  // -------------------------------------------------------------------------
  // I/O
  // -------------------------------------------------------------------------

  /// Stream as "E^a X^b S^c ω^d" for debugging.
  friend std::ostream &operator<<(std::ostream &os, const Clifford &cliff) {
    os << "E^" << cliff._a << " X^" << cliff._b << " S^" << cliff._c << " ω^"
       << cliff._d;
    return os;
  }

  // -------------------------------------------------------------------------
  // Conversion
  // -------------------------------------------------------------------------

  /// Convert a Clifford Gate to its Clifford group element.
  ///
  /// Precondition: g must be one of H, S, X, W. T is not a Clifford gate;
  /// passing Gate::T triggers an assertion in debug builds.
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

  // -------------------------------------------------------------------------
  // Comparison
  // -------------------------------------------------------------------------

  /// Two Clifford elements are equal iff all four exponents are identical.
  constexpr bool operator==(const Clifford &other) const {
    return _a == other._a && _b == other._b && _c == other._c && _d == other._d;
  }

  constexpr bool operator!=(const Clifford &other) const {
    return !(*this == other);
  }

  // -------------------------------------------------------------------------
  // Table lookup helpers (used internally by operator*, inv, `decompose_tconj`)
  // -------------------------------------------------------------------------

  /// Conjugate X^b past S^c via CONJ2_TABLE. See table comment for semantics.
  static constexpr std::pair<i32, i32> conj2(i32 c, i32 b) {
    i32 index = (c << 1) | b;
    return CONJ2_TABLE[static_cast<size_t>(index)];
  }

  /// Conjugate X^b · S^c past E^a via CONJ3_TABLE. See table comment for
  /// semantics.
  static constexpr std::array<i32, 4> conj3(i32 b, i32 c, i32 a) {
    i32 index = (a << 3) | (b << 2) | c;
    return CONJ3_TABLE[static_cast<size_t>(index)];
  }

  /// Invert E^a · X^b · S^c (ω^d excluded) via CINV_TABLE. See table comment
  /// for semantics.
  static constexpr std::array<i32, 4> cinv(i32 a, i32 b, i32 c) {
    i32 index = (a << 3) | (b << 2) | c;
    return CINV_TABLE[static_cast<size_t>(index)];
  }

  /// Look up the Matsumoto-`Amano` syllable for C·T via TCONJ_TABLE.
  /// See table comment for semantics.
  static constexpr std::array<i32, 3> tconj(i32 a, i32 b) {
    i32 index = (a << 1) | b;
    return TCONJ_TABLE[static_cast<size_t>(index)];
  }

  // -------------------------------------------------------------------------
  // Group arithmetic
  // -------------------------------------------------------------------------

  /// Group multiplication via conjugation tables. O(1), `constexpr`.
  ///
  /// Computes (E^a · X^b · S^c · ω^d) · (E^a' · X^b' · S^c' · ω^d')
  /// by conjugating the inner generators past each other:
  ///   1. Conjugate (X^b · S^c) past E^{a'} via CONJ3_TABLE.
  ///   2. Conjugate the resulting S-power past X^{b'} via CONJ2_TABLE.
  ///   3. Accumulate the exponents with carry from each conjugation.
  constexpr Clifford operator*(const Clifford &other) const {
    auto [a1, b1, c1, d1] = conj3(_b, _c, other._a);
    auto [c2, d2] = conj2(c1, other._b);

    i32 new_a = _a + a1;
    i32 new_b = b1 + other._b;
    i32 new_c = c2 + other._c;
    i32 new_d = d2 + d1 + _d + other._d;

    return Clifford(new_a, new_b, new_c, new_d);
  }

  /// Group inverse via CINV_TABLE. O(1), `constexpr`.
  constexpr Clifford inv() const {
    auto [a1, b1, c1, d1] = cinv(_a, _b, _c);
    return Clifford(a1, b1, c1, d1 - _d);
  }

  /// Decompose into `coset` representative and subgroup remainder.
  ///
  /// Writes C = rep · remainder where:
  ///   a = 0: rep = I,  remainder = C           (Axis::I)
  ///   a = 1: rep = H,  remainder = H⁻¹ · C    (Axis::H)
  ///   a = 2: rep = SH, remainder = (SH)⁻¹ · C (Axis::SH)
  ///
  /// The remainder always has a = 0, i.e. it lies in ⟨X, S, ω⟩.
  /// Used by to_circuit() for gate serialization and by normalize_gates()
  /// to identify syllable types.
  ///
  /// Defined after CLIFFORD_H and CLIFFORD_SH constants.
  constexpr std::pair<Axis, Clifford> decompose_coset() const;

  /// Matsumoto-`Amano` syllable decomposition: determine which syllable type
  /// results from appending T to a circuit ending with this Clifford.
  ///
  /// Computes C·T = syllable · C' where:
  ///   Axis::I  → syllable = T,   C' has a = 0
  ///   Axis::H  → syllable = HT,  C' has a = 0
  ///   Axis::SH → syllable = SHT, C' has a = 0
  ///
  /// The returned C' is the new trailing Clifford context. The syllable
  /// type depends only on (a, b), not (c, d), which are passed through.
  /// This is the core operation of normalize_gates().
  constexpr std::pair<Axis, Clifford> decompose_tconj() const {
    auto tconj_result = tconj(_a, _b);
    Axis axis = static_cast<Axis>(tconj_result[0]);
    i32 c1 = tconj_result[1];
    i32 d1 = tconj_result[2];

    return {axis, Clifford(0, _b, c1 + _c, d1 + _d)};
  }

  /// Serialize this Clifford element to a Circuit of elementary gates.
  ///
  /// Uses decompose_coset() to produce: coset_rep · X^b · S^c · W^d
  ///   a = 0 →        X^b · S^c · W^d
  ///   a = 1 → H    · X^b · S^c · W^d
  ///   a = 2 → S·H  · X^b · S^c · W^d
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

// Predefined Clifford elements in the (a, b, c, d) `parametrization`.
//
// Basic generators:
//   I = (0, 0, 0, 0)  Identity
//   X = (0, 1, 0, 0)  Pauli-X (bit-flip)
//   H = (1, 0, 1, 5)  Hadamard
//   S = (0, 0, 1, 0)  Phase gate `diag`(1, i)
//   W = (0, 0, 0, 1)  Global phase ω = e^{iπ/4}
//
// Composite Cliffords used by the Matsumoto-`Amano` normal form:
//   SH  = S · H   = (2, 0, 0, 3)  Coset representative for a = 2.
//   HS  = H · S   = (1, 0, 2, 5)  Absorbed when merging HT·T syllables.
//   SHS = S · H · S = (2, 0, 1, 3)  Absorbed when merging SHT·T syllables.

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

// Compile-time verification of Clifford group identities.
namespace {
static_assert(CLIFFORD_S * CLIFFORD_S * CLIFFORD_S * CLIFFORD_S == CLIFFORD_I,
              "S^4 = I");
static_assert((CLIFFORD_H * CLIFFORD_H).a() == 0 &&
                  (CLIFFORD_H * CLIFFORD_H).b() == 0 &&
                  (CLIFFORD_H * CLIFFORD_H).c() == 0,
              "H^2 = I (mod phase)");
static_assert(CLIFFORD_S * CLIFFORD_S.inv() == CLIFFORD_I, "S * S^-1 = I");

// Verify computed (a,b,c,d) coordinates of composite Cliffords.
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
