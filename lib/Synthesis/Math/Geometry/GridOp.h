/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Integer.h"
#include "Math/Ring/Domega.h"
#include "Math/Ring/Zomega.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Support/Result.h"

#include <array>
#include <cassert>
#include <complex>
#include <string>

namespace cudaq::synth {

/// GridOp: A special grid operator G : R² → R², represented as a real 2×2
/// matrix with entries in Z[1/√2].
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §5.3 (Definition 5.10,
/// Lemma 5.11, Proposition 5.13).
///
/// A grid operator G is a real linear map satisfying G(Z[ω]) ⊆ Z[ω].
/// By Lemma 5.11, G has the form:
///
///   G = [[a + a'/√2,  b + b'/√2],
///        [c + c'/√2,  d + d'/√2]]
///
/// where a,b,c,d,a',b',c',d' ∈ Z with a+b+c+d ≡ 0 (mod 2) and
/// a' ≡ b' ≡ c' ≡ d' (mod 2).
///
/// Representation: stored as a pair of ZOmega elements (u0, u1).
/// Each ZOmega u = (a,b,c,d) encodes a column of the matrix via:
///   column = [d + (c-a)/√2, b + (c+a)/√2]^T   (Lemma 5.5)
///
/// A grid operator is "special" if det(G) = ±1 (Definition 5.10).
///
/// Key property (Proposition 5.13): For a special grid operator G, the
/// two-dimensional grid problem for sets A and B is computationally
/// equivalent to the grid problem for G(A) and G●(B), where G● applies
/// the √2-conjugation to each matrix entry. This is the foundation of
/// the "to-upright" algorithm (Theorem 5.16).
///
/// The action on ZOmega elements (operator*(ZOmega)) corresponds to the
/// matrix-vector product when Z[ω] is identified with R² via
///   u = (a,b,c,d) ↦ [Re(u), Im(u)]^T.
class GridOp {
private:
  ZOmega _u0;
  ZOmega _u1;

public:
  /// Construct from first and second column vectors in the Z[ω] encoding.
  ///
  /// u0 encodes column 0 as [Re(u0), Im(u0)]^T and u1 encodes column 1.
  explicit GridOp(const ZOmega &u0, const ZOmega &u1) : _u0(u0), _u1(u1) {}

  /// The 2×2 identity grid operator [[1,0],[0,1]].
  ///
  /// In the ZOmega column encoding, the columns [1,0]ᵀ and [0,1]ᵀ are
  /// represented as ZOmega(0,0,0,1) and ZOmega(0,1,0,0) respectively
  /// (using Re(u)=d+(c-a)/√2 and Im(u)=b+(c+a)/√2 from Lemma 5.5).
  ///
  /// Implemented as a function-local static to avoid static initialization
  /// order issues (same pattern as ZSqrt2::lambda()).
  static const GridOp &identity() {
    static const GridOp value(ZOmega(0, 0, 0, 1), ZOmega(0, 1, 0, 0));
    return value;
  }

  const ZOmega &u0() const { return _u0; }
  const ZOmega &u1() const { return _u1; }

  // Convenience accessors for the individual Z[ω] coefficients of each column.
  const Integer &a0() const { return _u0.a(); }
  const Integer &b0() const { return _u0.b(); }
  const Integer &c0() const { return _u0.c(); }
  const Integer &d0() const { return _u0.d(); }

  const Integer &a1() const { return _u1.a(); }
  const Integer &b1() const { return _u1.b(); }
  const Integer &c1() const { return _u1.c(); }
  const Integer &d1() const { return _u1.d(); }

  // ---------------------------------------------------------------------------
  // Arithmetic operators
  // ---------------------------------------------------------------------------

  /// Composition: (G₁ · G₂)(u) = G₁(G₂(u)).
  ///
  /// Each column of the result is obtained by applying *this to the
  /// corresponding column of other via operator*(ZOmega).
  GridOp operator*(const GridOp &other) const {
    return GridOp((*this) * other.u0(), (*this) * other.u1());
  }

  /// Apply G to u ∈ Z[ω]: matrix-vector product in the Z[ω] column encoding.
  ///
  /// Each Z[ω] element represents a point in R² as [Re(u), Im(u)]^T. The
  /// matrix G acts on this pair, and the result is re-encoded as a Z[ω]
  /// element. The formula is derived from Lemma 5.5 combined with the
  /// constraint that a+b+c+d ≡ 0 (mod 2) and a' ≡ b' ≡ c' ≡ d' (mod 2),
  /// which guarantees that all intermediate half-integer expressions are exact.
  ///
  /// The auxiliary quantities t1..t8 are the eight half-integer sub-expressions
  /// shared across the four output coefficients (CSE to minimise GMP
  /// allocations).
  ZOmega operator*(const ZOmega &other) const {
    const Integer &a0_ = a0(), &b0_ = b0(), &c0_ = c0(), &d0_ = d0();
    const Integer &a1_ = a1(), &b1_ = b1(), &c1_ = c1(), &d1_ = d1();
    const Integer &oa = other.a(), &ob = other.b(), &oc = other.c(),
                  &od = other.d();

    Integer p = c1_ - a1_, q = c0_ - a0_; // → t1 = (p+q)/2, t2 = (p-q)/2
    Integer u = b1_ + d1_, v = b0_ + d0_; // → t3 = (u+v)/2, t4 = (u-v)/2
    Integer r = c1_ + a1_, s = c0_ + a0_; // → t5 = (r+s)/2, t6 = (r-s)/2
    Integer x = b1_ - d1_, y = b0_ - d0_; // → t7 = (x+y)/2, t8 = (x-y)/2

    Integer t1 = floordiv(p + q, 2);
    Integer t2 = floordiv(p - q, 2);
    Integer t3 = floordiv(u + v, 2);
    Integer t4 = floordiv(u - v, 2);
    Integer t5 = floordiv(r + s, 2);
    Integer t6 = floordiv(r - s, 2);
    Integer t7 = floordiv(x + y, 2);
    Integer t8 = floordiv(x - y, 2);

    Integer new_d = d0_ * od + d1_ * ob + t1 * oc + t2 * oa;
    Integer new_c = c0_ * od + c1_ * ob + t3 * oc + t4 * oa;
    Integer new_b = b0_ * od + b1_ * ob + t5 * oc + t6 * oa;
    Integer new_a = a0_ * od + a1_ * ob + t7 * oc + t8 * oa;

    return ZOmega(new_a, new_b, new_c, new_d);
  }

  /// Apply G to x ∈ D[ω], lifting the Z[ω] action to D[ω].
  ///
  /// The denominator exponent k is unchanged since G maps Z[ω] to Z[ω]
  /// (Definition 5.10), so G(u/√2^k) = G(u)/√2^k.
  DOmega operator*(const DOmega &other) const {
    return DOmega((*this) * other.u(), other.k());
  }

  /// Returns the operator as "GridOp(u0=(a,b,c,d), u1=(a,b,c,d))" showing the
  /// two Z[ω] column vectors. Intended for logging and debugging.
  std::string to_string() const {
    return "GridOp(u0=" + _u0.to_string() + ", u1=" + _u1.to_string() + ")";
  }
};

// ---------------------------------------------------------------------------
// Free functions on GridOp
// ---------------------------------------------------------------------------

/// √2-conjugation G●: applies (-)● to each column element.
///
/// By Remark 5.12, G●(u●) = (G·u)● for all u ∈ Z[ω]. This is used in
/// Proposition 5.13 to transform the second set B in the grid problem:
/// the original problem (A, B) is equivalent to (G(A), G●(B)).
inline GridOp conj_sq2(const GridOp &G) {
  return GridOp(G.u0().conj_sq2(), G.u1().conj_sq2());
}

/// Convert G to a floating-point 2×2 complex matrix.
///
/// Each column ui encodes the two real matrix entries as [Re(ui), Im(ui)]^T
/// (Lemma 5.5). The real parts form the first row and the imaginary parts
/// form the second row. The imaginary parts are zero for a valid grid
/// operator (which is a real matrix).
///
/// Prefer to_real_mat() when only the real entries are needed: it avoids
/// allocating the four imaginary GMP objects.
inline std::array<std::array<std::complex<Real>, 2>, 2>
to_mat(const GridOp &G) {
  Real u0_r, u0_i, u1_r, u1_i;
  to_real_imag(G.u0(), u0_r, u0_i);
  to_real_imag(G.u1(), u1_r, u1_i);
  return {{{{u0_r, u1_r}}, {{u0_i, u1_i}}}};
}

/// Convert G to a floating-point 2×2 real matrix.
///
/// Returns only the four real entries of the grid operator matrix (the
/// imaginary parts of a valid grid operator are zero).  Avoids allocating
/// the four imaginary GMP objects that to_mat() would create.
///
/// Layout: result[row][col], identical to the real parts of to_mat().
inline std::array<std::array<Real, 2>, 2> to_real_mat(const GridOp &G) {
  Real u0_r, u0_i, u1_r, u1_i;
  to_real_imag(G.u0(), u0_r, u0_i);
  to_real_imag(G.u1(), u1_r, u1_i);
  return {{{{std::move(u0_r), std::move(u1_r)}},
           {{std::move(u0_i), std::move(u1_i)}}}};
}

/// Inverse of G (valid only for special grid operators, Definition 5.10).
///
/// A grid operator G is special iff det(G) = ±1, where det(G) = Im(u0†·u1)
/// (the imaginary part of the column inner product). The specialness check
/// is:
///   det_element = u0.conj() * u1;
///   special iff det_element.a() + det_element.c() == 0   (no √2 term)
///               && |det_element.b()| == 1                (unit imaginary)
///
/// The inverse formula is derived from the 2×2 adjugate formula divided
/// by det, with the P/Q/R/S sub-expressions shared across the two output
/// columns to minimise GMP allocations.
///
/// Returns failure() if G is not special.
inline FailureOr<GridOp> inv(const GridOp &G) {
  ZOmega det = G.u0().conj() * G.u1();
  if (!((det.a() + det.c() == 0) && (det.b() == 1 || det.b() == -1)))
    return failure();

  const Integer &a0_ = G.a0(), &b0_ = G.b0(), &c0_ = G.c0(), &d0_ = G.d0();
  const Integer &a1_ = G.a1(), &b1_ = G.b1(), &c1_ = G.c1(), &d1_ = G.d1();

  const Integer P = c1_ + a1_, Q = c0_ + a0_;
  const Integer R = c1_ - a1_, S = c0_ - a0_;

  Integer new_c0 = floordiv(P - Q, 2);
  Integer new_a0 = floordiv(-(P + Q), 2);
  ZOmega new_u0(new_a0, -b0_, new_c0, b1_);

  Integer new_c1 = floordiv(S - R, 2);
  Integer new_a1 = floordiv(R + S, 2);
  ZOmega new_u1(new_a1, d0_, new_c1, -d1_);

  if (det.b() == -1) {
    new_u0 = -new_u0;
    new_u1 = -new_u1;
  }

  return GridOp(new_u0, new_u1);
}

/// Integer power G^exp.
///
/// Uses binary exponentiation (O(log|exp|) GridOp multiplications). Negative
/// exponents require G to be special (invertible); this precondition is
/// asserted. The loop is structured to avoid a final wasted squaring on the
/// last iteration (saves one full GridOp multiplication per call).
///
/// The identity element is (u0, u1) = ((0,0,0,1), (0,1,0,0)), corresponding
/// to the identity matrix [[1,0],[0,1]] in the ZOmega column encoding.
inline GridOp pow(const GridOp &G, Integer exp) {
  if (exp < 0) {
    FailureOr<GridOp> inv_or = inv(G);
    assert(succeeded(inv_or) && "pow(GridOp): matrix is not special");
    return pow(*inv_or, -exp);
  }

  GridOp result = GridOp::identity();
  GridOp base = G;

  while (true) {
    if (exp.is_odd())
      result = result * base;
    exp >>= 1;
    if (!(exp > 0))
      break;
    base = base * base;
  }
  return result;
}

} // namespace cudaq::synth
