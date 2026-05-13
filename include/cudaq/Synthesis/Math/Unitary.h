/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <complex>
#include <string>

#include "cudaq/Synthesis/Math/Ring/Domega.h"
#include "cudaq/Synthesis/Circuit/Circuit.h"

namespace cudaq::synth {

/// DOmegaUnitary: A 2×2 unitary matrix with entries in D[ω], `parametrized` as:
///
///   U = [[z, -w† · ωⁿ],
///        [w,  z† · ωⁿ]]
///
/// where z, w ∈ D[ω] and n ∈ Z/8Z.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §7.1, equations (11)-(12),
/// and `Kliuchnikov, Maslov, Mosca` [10].
///
/// From [10], a single-qubit operator can be exactly written as a product
/// of Clifford+T operators iff all its matrix entries belong to D[ω].
/// The special form (12) with n = 0 is:
///   U = [[u, -t†], [t, u†]]
/// where u†u + t†t = 1.
///
/// By Lemma 7.2, for ε < |1 - e^{iπ/8}|, all solutions of the
/// approximate synthesis problem have this form (n = 0).
///
/// The T-count of U is determined by the least denominator exponent k
/// of z (equivalently w): T-count = 2k-2 for k > 0, or 0 for k = 0
/// (Lemma 7.3). If the T-count is 2k, then U' = TUT† has T-count 2k-2
/// and approximates equally well (since R_z(θ) commutes with T).
///
/// Gate multiplications (mul_by_*_from_left) implement the action of
/// standard gates on the (z, w, n) representation. These are used by
/// the exact decomposition algorithm (decompose_domega_unitary in
/// kmm_synthesize.h) to peel off gates one at a time while reducing the
/// denominator exponent.
///
/// from_gates() reconstructs U from a gate string, used for verification.
class DOmegaUnitary {
private:
  DOmega _z, _w;
  i32 _n;

public:
  // Constructor
  DOmegaUnitary(const DOmega &z, const DOmega &w, i32 n, i32 k = -1)
      : _z(z), _w(w), _n(n & 0b111) {

    if (k == -1) {
      // Auto-align denominators
      if (_z.k() > _w.k()) {
        _w = with_denom_exp(_w, _z.k());
      } else if (_z.k() < _w.k()) {
        _z = with_denom_exp(_z, _w.k());
      }
    } else {
      // Use specified denominator exponent
      _z = with_denom_exp(_z, k);
      _w = with_denom_exp(_w, k);
    }
  }

  // Properties
  DOmega z() const { return _z; }
  DOmega w() const { return _w; }
  i32 n() const { return _n; }
  i32 k() const { return static_cast<i32>(_w.k()); }

  // Matrix representation as 2x2 array of DOmega
  std::array<std::array<DOmega, 2>, 2> to_matrix() const {
    DOmega m00 = _z;
    DOmega m01 = -mul_by_omega_power(_w.conj(), _n);
    DOmega m10 = _w;
    DOmega m11 = mul_by_omega_power(_z.conj(), _n);

    return {{{{m00, m01}}, {{m10, m11}}}};
  }

  // Complex matrix representation (avoid intermediate arrays and extra
  // temporaries)
  std::array<std::array<std::complex<Real>, 2>, 2> to_complex_matrix() const {
    // Build entries directly and use fast coord extraction with hoisted
    // invariants.
    DOmega m00 = _z;
    DOmega m01 = -mul_by_omega_power(_w.conj(), _n);
    DOmega m10 = _w;
    DOmega m11 = mul_by_omega_power(_z.conj(), _n);

    // All entries share the same denominator exponent k
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

  // Equality operator
  bool operator==(const DOmegaUnitary &other) const {
    return _z == other._z && _w == other._w && _n == other._n;
  }

  bool operator!=(const DOmegaUnitary &other) const {
    return !(*this == other);
  }

  // Gate multiplications from the left.
  //
  // Each method computes g · U where g is a standard gate and U = *this.
  // The transformations on (z, w, n) follow from the matrix forms:
  //   T = `diag(1, ω)`        → (z, ω·w, n+1)
  //   S = `diag(1, i) = T²`   → (z, i·w, n+2)
  //   H = (1/√2)[[1,1],[1,-1]] → ((z+w)/√2, (z-w)/√2, n+4)
  //   X = [[0,1],[1,0]]     → (w, z, n+4)
  //   W = ω·I (global phase)→ (ω·z, ω·w, n+2)
  //
  // These follow from the `parametrization` U = [[z, -w†·ωⁿ], [w, z†·ωⁿ]]
  // and the fact that g·U must again have this form.

  // T·U: T = `diag(1, ω)` multiplies the lower-left entry w by ω.
  DOmegaUnitary mul_by_T_from_left() const {
    return DOmegaUnitary(_z, mul_by_omega(_w), _n + 1);
  }

  DOmegaUnitary mul_by_T_inv_from_left() const {
    return DOmegaUnitary(_z, mul_by_omega_inv(_w), _n - 1);
  }

  DOmegaUnitary mul_by_T_power_from_left(i32 m) const {
    m &= 0b111; // mod 8
    return DOmegaUnitary(_z, mul_by_omega_power(_w, m), _n + m);
  }

  DOmegaUnitary mul_by_S_from_left() const {
    return DOmegaUnitary(_z, mul_by_omega_power(_w, 2), _n + 2);
  }

  DOmegaUnitary mul_by_S_power_from_left(i32 m) const {
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

  DOmegaUnitary mul_by_W_power_from_left(i32 m) const {
    m &= 0b111; // mod 8
    return DOmegaUnitary(mul_by_omega_power(_z, m), mul_by_omega_power(_w, m),
                         _n + (m << 1));
  }

  // Static factory methods
  static DOmegaUnitary identity() {
    return DOmegaUnitary(DOmega::from_int(1), DOmega::from_int(0), 0);
  }

  /// Reconstruct a DOmegaUnitary from a Circuit (H, T, S, X, W).
  /// Defined out-of-line because it requires the with_denom_exp and
  /// to_lde free functions, which are declared after this class.
  static DOmegaUnitary from_gates(const Circuit &circuit);

  /// Returns "DOmegaUnitary(z=..., w=..., n=N)" delegating to
  /// DOmega::to_string() for z and w. Intended for logging and debugging.
  std::string to_string() const {
    return "DOmegaUnitary(z=" + _z.to_string() + ", w=" + _w.to_string() +
           ", n=" + std::to_string(_n) + ")";
  }
};

// ---------------------------------------------------------------------------
// Free functions on DOmegaUnitary
// ---------------------------------------------------------------------------

/// with_denom_exp: Return a copy of u with both z and w re-expressed at
/// denominator exponent new_k. Overloads with_denom_exp(DOmega, Integer).
inline DOmegaUnitary with_denom_exp(const DOmegaUnitary &u, i32 new_k) {
  return DOmegaUnitary(u.z(), u.w(), u.n(), new_k);
}

/// to_lde: Return a copy of u with the minimal denominator exponent
/// by calling to_lde(DOmega) on each of z and w independently.
/// Overloads to_lde(DOmega).
inline DOmegaUnitary to_lde(const DOmegaUnitary &u) {
  return DOmegaUnitary(to_lde(u.z()), to_lde(u.w()), u.n());
}

// ---------------------------------------------------------------------------
// Approximation error metrics
// ---------------------------------------------------------------------------

/// Computes the operator norm approximation error ‖R_z(θ) - U‖.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, equation (13).
///
/// Constructs E = U - R_z(θ) entry-by-entry using the complex matrix of u
/// (via to_complex_matrix()) and returns √|`det`(E)| as a proxy for the
/// operator norm. For small ε, the two singular values of E are approximately
/// equal, so this proxy is accurate in the regime where the synthesizer
/// operates.
///
/// @param u     The approximating Clifford+T unitary
/// @param theta Target rotation angle θ; R_z(θ) = `diag(e^{-iθ/2}, e^{iθ/2})`
/// @return      √|`det`(U - R_z(θ))|
inline Real rz_approximation_error(const DOmegaUnitary &u, const Real &theta) {
  Real half = theta / Real(2.0);
  Real c = cos(half);
  Real s = sin(half);

  auto M = u.to_complex_matrix();

  // E = U - R_z(θ); diagonal entries absorb the subtraction,
  // off-diagonal entries are unchanged.
  //   R_z(θ)[0][0] = e^{-iθ/2} = c - i·s
  //   R_z(θ)[1][1] = e^{+iθ/2} = c + i·s
  Real e00r = M[0][0].real() - c;
  Real e00i = M[0][0].imag() + s;
  Real e01r = M[0][1].real();
  Real e01i = M[0][1].imag();
  Real e10r = M[1][0].real();
  Real e10i = M[1][0].imag();
  Real e11r = M[1][1].real() - c;
  Real e11i = M[1][1].imag() - s;

  // `det(E) = E[0][0]·E[1][1] − E[0][1]·E[1][0]`, computed component-wise.
  Real det_re = (e00r * e11r - e00i * e11i) - (e01r * e10r - e01i * e10i);
  Real det_im = (e00r * e11i + e00i * e11r) - (e01r * e10i + e01i * e10r);

  return sqrt(sqrt(det_re * det_re + det_im * det_im));
}

/// Compute ‖R_z(θ) - U‖ for a circuit already reconstructed as a Circuit.
///
/// @param circuit  Clifford+T circuit
/// @param theta    Target rotation angle θ
/// @return         √|`det`(U - R_z(θ))| as a decimal string
inline std::string rz_gate_sequence_error(const std::string &theta,
                                          const Circuit &circuit) {
  return rz_approximation_error(DOmegaUnitary::from_gates(circuit), Real(theta))
      .to_string();
}

/// String-API overload: parses the gate string before delegating.
///
/// Provided for I/O boundaries (test utilities, print tools) where the
/// circuit is represented as a string. Inside the synthesis pipeline, prefer
/// the Circuit overload above.
///
/// @param theta  Target rotation angle as an arbitrary-precision decimal string
/// @param gates  Clifford+T gate sequence string (characters H, T, S, X, W)
/// @return       √|`det`(U - R_z(θ))| as a decimal string
inline std::string rz_gate_sequence_error(const std::string &theta,
                                          const std::string &gates) {
  auto circuit_or = Circuit::from_string(gates);
  assert(succeeded(circuit_or) &&
         "rz_gate_sequence_error: invalid gate string");
  return rz_gate_sequence_error(theta, *circuit_or);
}

// ---------------------------------------------------------------------------
// Out-of-line DOmegaUnitary member definitions
// ---------------------------------------------------------------------------

inline DOmegaUnitary DOmegaUnitary::from_gates(const Circuit &circuit) {
  DOmegaUnitary unitary = identity();

  // Process gates in reverse order (right-to-left multiplication).
  for (auto it = circuit.rbegin(); it != circuit.rend(); ++it) {
    switch (*it) {
    case Gate::H:
      unitary =
          with_denom_exp(unitary, unitary.k() + i32(1)).mul_by_H_from_left();
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
