/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <complex>
#include <string>

#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Math/Ring/Domega.h"
#include "llvm/Support/LogicalResult.h"

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// DOmegaUnitary
//===----------------------------------------------------------------------===//

/// A 2x2 unitary with entries in D[omega], parametrised as
///
///     U = [[ z,           -conj(w) * omega^n ],
///          [ w,            conj(z) * omega^n ]]
///
/// where z, w in D[omega] and n in Z/8Z.
///
/// References: Ross & Selinger, arXiv:1403.2975, sec. 7.1, equations (11)
/// and (12); Kliuchnikov, Maslov, Mosca [10].
///
/// Theorem (from [10]): a single-qubit operator is exactly Clifford+T iff
/// every matrix entry lies in D[omega]. The n = 0 specialisation collapses
/// to
///     U = [[ u, -conj(t) ], [ t, conj(u) ]]
/// with conj(u)*u + conj(t)*t = 1. Lemma 7.2 of the paper proves that for
/// epsilon < |1 - e^{i*pi/8}| every approximate-synthesis solution has this
/// form (n = 0).
///
/// T-count. Determined by the least denominator exponent k shared by z and
/// w: T-count = 2*k - 2 for k > 0, or 0 for k = 0 (Lemma 7.3). When the
/// raw answer is 2*k, the equivalent unitary T*U*conj(T) achieves 2*k - 2
/// and approximates equally well because R_z(theta) commutes with T.
///
/// Gate operators. The `mul_by_*_from_left` family applies the standard
/// Clifford+T generators to U by transforming (z, w, n) directly. The
/// exact decomposition algorithm (kmm_synthesize.h) uses these to peel off
/// gates one at a time while driving the denominator exponent down to zero.
///
/// `from_gates` reverses a gate string back to a DOmegaUnitary -- used by
/// the verification path.
class DOmegaUnitary {
private:
  DOmega _z, _w;
  int32_t _n;

public:
  /// Construct from explicit (z, w, n). If `k` is negative (the default),
  /// z and w are auto-aligned to the larger of their two denominator
  /// exponents; otherwise both are renormalised to exactly k.
  DOmegaUnitary(const DOmega &z, const DOmega &w, int32_t n, int32_t k = -1)
      : _z(z), _w(w), _n(n & 0b111) {

    if (k == -1) {
      if (_z.k() > _w.k()) {
        _w = with_denom_exp(_w, _z.k());
      } else if (_z.k() < _w.k()) {
        _z = with_denom_exp(_z, _w.k());
      }
    } else {
      _z = with_denom_exp(_z, k);
      _w = with_denom_exp(_w, k);
    }
  }

  // -- Accessors --

  DOmega z() const { return _z; }
  DOmega w() const { return _w; }
  int32_t n() const { return _n; }
  int32_t k() const { return static_cast<int32_t>(_w.k()); }

  /// 2x2 matrix view with entries in D[omega] (equation (12) instantiated).
  std::array<std::array<DOmega, 2>, 2> to_matrix() const {
    DOmega m00 = _z;
    DOmega m01 = -mul_by_omega_power(_w.conj(), _n);
    DOmega m10 = _w;
    DOmega m11 = mul_by_omega_power(_z.conj(), _n);

    return {{{{m00, m01}}, {{m10, m11}}}};
  }

  /// 2x2 floating-point complex matrix. Avoids the intermediate DOmega
  /// matrix by computing all four entries' real/imag coordinates with a
  /// shared inv_scale and sqrt(2)/2 (`coords_into` amortises the MPFR
  /// work across the four entries).
  std::array<std::array<std::complex<Real>, 2>, 2> to_complex_matrix() const {
    DOmega m00 = _z;
    DOmega m01 = -mul_by_omega_power(_w.conj(), _n);
    DOmega m10 = _w;
    DOmega m11 = mul_by_omega_power(_z.conj(), _n);

    // All four entries share the same denominator exponent k.
    const Real inv_scale = Real(1.0) / pow_sqrt2(_w.k());
    const Real sqrt2_over_2 = Real::sqrt2() / 2;

    Real r00, i00, r01, i01, r10, i10, r11, i11;
    coords_into(m00, inv_scale, sqrt2_over_2, r00, i00);
    coords_into(m01, inv_scale, sqrt2_over_2, r01, i01);
    coords_into(m10, inv_scale, sqrt2_over_2, r10, i10);
    coords_into(m11, inv_scale, sqrt2_over_2, r11, i11);

    return {{{{std::complex<Real>(r00, i00), std::complex<Real>(r01, i01)}},
             {{std::complex<Real>(r10, i10), std::complex<Real>(r11, i11)}}}};
  }

  bool operator==(const DOmegaUnitary &other) const {
    return _z == other._z && _w == other._w && _n == other._n;
  }

  bool operator!=(const DOmegaUnitary &other) const {
    return !(*this == other);
  }

  // -- Left multiplication by standard gates --
  //
  // Each method computes g * U for a Clifford+T generator g. The (z, w, n)
  // transformations follow from the matrix forms of the generators:
  //
  //   T = diag(1, omega)         -> (z,           omega * w,         n + 1)
  //   S = diag(1, i) = T^2       -> (z,           i * w,             n + 2)
  //   H = (1/sqrt(2)) [[1, 1], [1, -1]]
  //                              -> ((z + w)/sqrt(2), (z - w)/sqrt(2), n + 4)
  //   X = [[0, 1], [1, 0]]       -> (w,           z,                 n + 4)
  //   W = omega * I (global phase)
  //                              -> (omega * z,  omega * w,           n + 2)
  //
  // The transformations are forced by the parametrisation
  //     U = [[ z, -conj(w) * omega^n ], [ w, conj(z) * omega^n ]]
  // together with the requirement that g * U must again have this form.

  /// T * U: multiplies the lower-left entry by omega.
  DOmegaUnitary mul_by_T_from_left() const {
    return DOmegaUnitary(_z, mul_by_omega(_w), _n + 1);
  }

  DOmegaUnitary mul_by_T_inv_from_left() const {
    return DOmegaUnitary(_z, mul_by_omega_inv(_w), _n - 1);
  }

  DOmegaUnitary mul_by_T_power_from_left(int32_t m) const {
    m &= 0b111; // mod 8
    return DOmegaUnitary(_z, mul_by_omega_power(_w, m), _n + m);
  }

  DOmegaUnitary mul_by_S_from_left() const {
    return DOmegaUnitary(_z, mul_by_omega_power(_w, 2), _n + 2);
  }

  DOmegaUnitary mul_by_S_power_from_left(int32_t m) const {
    m &= 0b11; // mod 4
    return DOmegaUnitary(_z, mul_by_omega_power(_w, m << 1), _n + (m << 1));
  }

  DOmegaUnitary mul_by_H_from_left() const {
    DOmega new_z = mul_by_inv_sqrt2(_z + _w);
    DOmega new_w = mul_by_inv_sqrt2(_z - _w);
    return DOmegaUnitary(new_z, new_w, _n + 4);
  }

  DOmegaUnitary mul_by_H_and_T_power_from_left(int m) const {
    return mul_by_T_power_from_left(m).mul_by_H_from_left();
  }

  DOmegaUnitary mul_by_X_from_left() const {
    return DOmegaUnitary(_w, _z, _n + 4);
  }

  DOmegaUnitary mul_by_W_from_left() const {
    return DOmegaUnitary(mul_by_omega(_z), mul_by_omega(_w), _n + 2);
  }

  DOmegaUnitary mul_by_W_power_from_left(int32_t m) const {
    m &= 0b111; // mod 8
    return DOmegaUnitary(mul_by_omega_power(_z, m), mul_by_omega_power(_w, m),
                         _n + (m << 1));
  }

  // -- Factories --

  static DOmegaUnitary identity() {
    return DOmegaUnitary(DOmega::from_int(1), DOmega::from_int(0), 0);
  }

  /// Reconstruct a DOmegaUnitary from a Circuit (gate alphabet H, T, S, X, W).
  /// Defined out of line because it needs `with_denom_exp` and `to_lde`,
  /// which are declared after this class.
  static DOmegaUnitary from_gates(const Circuit &circuit);

  /// "DOmegaUnitary(z=..., w=..., n=N)" rendering for debug logging.
  std::string to_string() const {
    return "DOmegaUnitary(z=" + _z.to_string() + ", w=" + _w.to_string() +
           ", n=" + std::to_string(_n) + ")";
  }
};

//===----------------------------------------------------------------------===//
// Free functions on DOmegaUnitary
//===----------------------------------------------------------------------===//

/// Re-express u with both z and w at denominator exponent `new_k`. Overload
/// of `with_denom_exp(DOmega, Integer)` lifted to the unitary.
inline DOmegaUnitary with_denom_exp(const DOmegaUnitary &u, int32_t new_k) {
  return DOmegaUnitary(u.z(), u.w(), u.n(), new_k);
}

/// Reduce u to its least denominator exponent by independently calling
/// to_lde(DOmega) on z and on w.
inline DOmegaUnitary to_lde(const DOmegaUnitary &u) {
  return DOmegaUnitary(to_lde(u.z()), to_lde(u.w()), u.n());
}

//===----------------------------------------------------------------------===//
// Approximation-error metrics
//===----------------------------------------------------------------------===//

/// Operator-norm proxy for |R_z(theta) - U|.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, equation (13).
///
/// Builds E = U - R_z(theta) entry by entry from the complex matrix of u
/// (via `to_complex_matrix`) and returns sqrt(|det(E)|). For small epsilon
/// the two singular values of E are approximately equal, so this proxy is
/// accurate in the regime the synthesizer targets.
inline Real rz_approximation_error(const DOmegaUnitary &u, const Real &theta) {
  Real half = theta / Real(2.0);
  Real c = cos(half);
  Real s = sin(half);

  auto M = u.to_complex_matrix();

  // E = U - R_z(theta). The diagonal entries absorb the subtraction;
  // off-diagonal entries of R_z are zero.
  //   R_z(theta)[0][0] = e^{-i*theta/2} = c - i*s
  //   R_z(theta)[1][1] = e^{+i*theta/2} = c + i*s
  Real e00r = M[0][0].real() - c;
  Real e00i = M[0][0].imag() + s;
  Real e01r = M[0][1].real();
  Real e01i = M[0][1].imag();
  Real e10r = M[1][0].real();
  Real e10i = M[1][0].imag();
  Real e11r = M[1][1].real() - c;
  Real e11i = M[1][1].imag() - s;

  // det(E) = E[0][0]*E[1][1] - E[0][1]*E[1][0], computed in (re, im) pairs.
  Real det_re = (e00r * e11r - e00i * e11i) - (e01r * e10r - e01i * e10i);
  Real det_im = (e00r * e11i + e00i * e11r) - (e01r * e10i + e01i * e10r);

  return sqrt(sqrt(det_re * det_re + det_im * det_im));
}

/// Convenience wrapper: compute |R_z(theta) - U| for a circuit already
/// realised as a Circuit. Theta is consumed as an arbitrary-precision
/// decimal string (so callers can stay in their preferred I/O format).
inline std::string rz_gate_sequence_error(const std::string &theta,
                                          const Circuit &circuit) {
  return rz_approximation_error(DOmegaUnitary::from_gates(circuit), Real(theta))
      .to_string();
}

/// String-API overload: parse the gate string before delegating. Provided
/// for I/O boundaries (test utilities, CLI tools); inside the synthesis
/// pipeline prefer the Circuit overload above.
inline std::string rz_gate_sequence_error(const std::string &theta,
                                          const std::string &gates) {
  llvm::FailureOr<Circuit> circuit_or = Circuit::from_string(gates);
  assert(llvm::succeeded(circuit_or) &&
         "rz_gate_sequence_error: invalid gate string");
  return rz_gate_sequence_error(theta, *circuit_or);
}

//===----------------------------------------------------------------------===//
// Out-of-line DOmegaUnitary members
//===----------------------------------------------------------------------===//

inline DOmegaUnitary DOmegaUnitary::from_gates(const Circuit &circuit) {
  DOmegaUnitary unitary = identity();

  // Right-to-left application: the rightmost gate acts first on the input
  // ket, which means we left-multiply it onto the identity first.
  for (auto it = circuit.rbegin(); it != circuit.rend(); ++it) {
    switch (*it) {
    case Gate::H:
      // H needs one extra factor of sqrt(2) in the denominator to keep
      // every entry inside D[omega].
      unitary = with_denom_exp(unitary, unitary.k() + int32_t(1))
                    .mul_by_H_from_left();
      break;
    case Gate::T:
      unitary = unitary.mul_by_T_from_left();
      break;
    case Gate::S:
      unitary = unitary.mul_by_S_from_left();
      break;
    case Gate::X:
      unitary = unitary.mul_by_X_from_left();
      break;
    case Gate::W:
      unitary = unitary.mul_by_W_from_left();
      break;
    }
  }

  return to_lde(unitary);
}

} // namespace cudaq::synth
