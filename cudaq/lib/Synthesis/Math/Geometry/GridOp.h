/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Math/Integer.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Ring/Domega.h"
#include "cudaq/Synthesis/Math/Ring/Zomega.h"
#include "llvm/Support/LogicalResult.h"

#include <array>
#include <cassert>
#include <complex>
#include <string>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// GridOp
//===----------------------------------------------------------------------===//

/// A special grid operator G: R^2 -> R^2 represented as a real 2x2 matrix
/// with entries in Z[1/sqrt(2)].
///
/// Reference: Ross & Selinger, arXiv:1403.2975, sec. 5.3 (Definition 5.10,
/// Lemma 5.11, Proposition 5.13).
///
/// Lemma 5.11 gives the entry structure:
///     G = [[ a + a'/sqrt(2),  b + b'/sqrt(2) ],
///          [ c + c'/sqrt(2),  d + d'/sqrt(2) ]]
/// with a, b, c, d, a', b', c', d' integers subject to
///     a + b + c + d == 0 (mod 2)
///     a' == b' == c' == d' (mod 2).
///
/// Internal representation. The matrix is stored as a pair of ZOmega
/// elements (u0, u1), one per column, where each ZOmega u = (a, b, c, d)
/// encodes a column via Lemma 5.5:
///     column = [ d + (c - a) / sqrt(2),  b + (c + a) / sqrt(2) ]^T.
///
/// A grid operator is "special" iff det(G) = +/-1 (Definition 5.10).
/// Proposition 5.13: for a special G, the 2D grid problem on sets (A, B) is
/// equivalent to the one on (G(A), conj_sq2(G)(B)). This is the foundation
/// of the to-upright algorithm (Theorem 5.16).
///
/// The action on ZOmega (operator*(ZOmega)) is the matrix-vector product
/// when Z[omega] is identified with R^2 via u = (a, b, c, d) ->
/// [Re(u), Im(u)]^T.
class GridOp {
private:
  ZOmega _u0;
  ZOmega _u1;

public:
  /// Construct from the two column vectors in the Z[omega] encoding.
  explicit GridOp(const ZOmega &u0, const ZOmega &u1) : _u0(u0), _u1(u1) {}

  /// 2x2 identity. In the ZOmega column encoding (Re(u) = d + (c - a)/sqrt(2),
  /// Im(u) = b + (c + a)/sqrt(2)) the canonical basis vectors [1, 0]^T and
  /// [0, 1]^T become ZOmega(0, 0, 0, 1) and ZOmega(0, 1, 0, 0). Held as a
  /// function-local static to dodge static-initialization-order issues, same
  /// pattern as ZSqrt2::lambda().
  static const GridOp &identity() {
    static const GridOp value(ZOmega(0, 0, 0, 1), ZOmega(0, 1, 0, 0));
    return value;
  }

  const ZOmega &u0() const { return _u0; }
  const ZOmega &u1() const { return _u1; }

  // Per-column shortcuts onto the individual Z[omega] coefficients.
  const Integer &a0() const { return _u0.a(); }
  const Integer &b0() const { return _u0.b(); }
  const Integer &c0() const { return _u0.c(); }
  const Integer &d0() const { return _u0.d(); }

  const Integer &a1() const { return _u1.a(); }
  const Integer &b1() const { return _u1.b(); }
  const Integer &c1() const { return _u1.c(); }
  const Integer &d1() const { return _u1.d(); }

  // -- Arithmetic --

  /// Composition: (G1 * G2)(u) = G1(G2(u)). Each column of the result is
  /// (*this) applied to the corresponding column of `other`.
  GridOp operator*(const GridOp &other) const {
    return GridOp((*this) * other.u0(), (*this) * other.u1());
  }

  /// Apply G to u in Z[omega] as a matrix-vector product in the column
  /// encoding. The structural constraints on (a, b, c, d, a', b', c', d')
  /// from Lemma 5.11 guarantee every intermediate half-integer expression
  /// is exact, so we can use floor-division by 2 throughout.
  ///
  /// The eight t1..t8 temporaries are the half-integer sub-expressions
  /// shared across the four output coefficients (CSE to minimise GMP
  /// allocations).
  ZOmega operator*(const ZOmega &other) const {
    const Integer &a0_ = a0(), &b0_ = b0(), &c0_ = c0(), &d0_ = d0();
    const Integer &a1_ = a1(), &b1_ = b1(), &c1_ = c1(), &d1_ = d1();
    const Integer &oa = other.a(), &ob = other.b(), &oc = other.c(),
                  &od = other.d();

    Integer p = c1_ - a1_, q = c0_ - a0_; // -> t1 = (p+q)/2, t2 = (p-q)/2
    Integer u = b1_ + d1_, v = b0_ + d0_; // -> t3 = (u+v)/2, t4 = (u-v)/2
    Integer r = c1_ + a1_, s = c0_ + a0_; // -> t5 = (r+s)/2, t6 = (r-s)/2
    Integer x = b1_ - d1_, y = b0_ - d0_; // -> t7 = (x+y)/2, t8 = (x-y)/2

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

  /// Lift the Z[omega] action to D[omega]. The denominator exponent k passes
  /// through unchanged because G maps Z[omega] into Z[omega], so
  /// G(u / sqrt(2)^k) = G(u) / sqrt(2)^k.
  DOmega operator*(const DOmega &other) const {
    return DOmega((*this) * other.u(), other.k());
  }

  /// "GridOp(u0=..., u1=...)" rendering for debug logging.
  std::string to_string() const {
    return "GridOp(u0=" + _u0.to_string() + ", u1=" + _u1.to_string() + ")";
  }
};

//===----------------------------------------------------------------------===//
// Free functions on GridOp
//===----------------------------------------------------------------------===//

/// Sqrt(2)-conjugation acting columnwise. By Remark 5.12,
/// conj_sq2(G)(conj_sq2(u)) = conj_sq2(G * u) for all u in Z[omega]; this is
/// what lets Proposition 5.13 transform (A, B) into (G(A), conj_sq2(G)(B)).
inline GridOp conj_sq2(const GridOp &G) {
  return GridOp(G.u0().conj_sq2(), G.u1().conj_sq2());
}

/// Floating-point view of G as a 2x2 complex matrix. Each column u_i carries
/// the two real matrix entries as [Re(u_i), Im(u_i)]^T (Lemma 5.5); the real
/// parts form the first matrix row and the imaginary parts form the second.
/// The imaginary parts are zero for any valid (real-valued) grid operator.
///
/// Prefer `to_real_mat` when only the real entries are needed: it skips
/// allocating the four imaginary GMP objects.
inline std::array<std::array<std::complex<Real>, 2>, 2>
to_mat(const GridOp &G) {
  Real u0_r, u0_i, u1_r, u1_i;
  to_real_imag(G.u0(), u0_r, u0_i);
  to_real_imag(G.u1(), u1_r, u1_i);
  return {{{{u0_r, u1_r}}, {{u0_i, u1_i}}}};
}

/// Floating-point view of G as a 2x2 real matrix. Returns just the four real
/// entries (the imaginary parts of a valid grid operator are zero), avoiding
/// the imaginary-part allocations that `to_mat` performs.
inline std::array<std::array<Real, 2>, 2> to_real_mat(const GridOp &G) {
  Real u0_r, u0_i, u1_r, u1_i;
  to_real_imag(G.u0(), u0_r, u0_i);
  to_real_imag(G.u1(), u1_r, u1_i);
  return {{{{std::move(u0_r), std::move(u1_r)}},
           {{std::move(u0_i), std::move(u1_i)}}}};
}

/// Inverse of G, defined only for special grid operators.
///
/// G is special iff det(G) = +/-1, where det(G) = Im(conj(u0) * u1). The
/// specialness predicate is: let d = conj(u0) * u1; then G is special iff
///     d.a() + d.c() == 0       (no sqrt(2) term)
///     and |d.b()| == 1         (unit imaginary part).
/// Returns failure() if either condition fails.
///
/// The closed-form inverse comes from the 2x2 adjugate divided by det. The
/// shared P, Q, R, S sub-expressions are factored out across the two output
/// columns to minimise GMP allocations.
inline llvm::FailureOr<GridOp> inv(const GridOp &G) {
  ZOmega det = G.u0().conj() * G.u1();
  if (!((det.a() + det.c() == 0) && (det.b() == 1 || det.b() == -1)))
    return llvm::failure();

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

  // det == -1 flips the sign of both columns of the adjugate.
  if (det.b() == -1) {
    new_u0 = -new_u0;
    new_u1 = -new_u1;
  }

  return GridOp(new_u0, new_u1);
}

/// Integer power G^exp by binary exponentiation.
///
/// Cost: O(log|exp|) GridOp multiplications. Negative exponents require G to
/// be invertible (i.e. special); the precondition is asserted. The loop is
/// arranged so the final iteration does not square `base` after the last
/// odd-exponent bit -- saves one full GridOp multiplication per call.
inline GridOp pow(const GridOp &G, Integer exp) {
  if (exp < 0) {
    llvm::FailureOr<GridOp> inv_or = inv(G);
    assert(llvm::succeeded(inv_or) && "pow(GridOp): matrix is not special");
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
