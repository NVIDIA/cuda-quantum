/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Synthesis/Synthesis/KmmSynthesize.h"
#include "Circuit/NormalForm.h"
#include "Support/StreamOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cudaq-synth"

namespace cudaq::synth {

namespace {

//===----------------------------------------------------------------------===//
// Denominator-reduction lookup tables
//===----------------------------------------------------------------------===//

// BIT_SHIFT[r] is the index of the lowest set bit of the 4-bit residue r
// (i.e. the trailing-zero count). BIT_COUNT[r] is the popcount of r. The
// reduce_denomexp dispatch uses both to choose the right T^m * H prefix
// that drops the denominator exponent of a DOmegaUnitary by one.
inline constexpr std::array<int32_t, 16> BIT_SHIFT = {0, 0, 1, 0, 2, 0, 1, 3,
                                                  3, 3, 0, 2, 2, 1, 0, 0};

inline constexpr std::array<int32_t, 16> BIT_COUNT = {0, 1, 1, 2, 1, 2, 2, 3,
                                                  1, 2, 2, 3, 2, 3, 3, 4};

//===----------------------------------------------------------------------===//
// reduce_denomexp
//===----------------------------------------------------------------------===//

/// Peel off a single gate (or short gate sequence) g from the left of a
/// DOmegaUnitary so that inv(g) * U has denominator exponent k - 1 (or
/// occasionally k - 2 as a bonus).
///
/// References: Kliuchnikov, Maslov, Mosca [10] (exact synthesis); Ross &
/// Selinger arXiv:1403.2975, sec. 7.3 step 3.
///
/// Key fact from [10]: with k = U's denominator exponent, sqrt(2)^k * U has
/// all entries in Z[omega]. The residues of those entries mod 2 form a
/// 4-bit pattern; the residue of conj(z) * z classifies the four cases
/// below and the gate prefix to apply.
///
/// Case mapping (residue of conj(z) * z mod 2):
///   0b0000: z is divisible by sqrt(2); H alone drops the exponent because
///           H swaps z and w.
///   0b1010: the generic case; apply T^{-m} * H where m aligns the trailing
///           bits of w to those of z.
///   0b0001: split by popcount(z) vs popcount(w). Equal popcounts drop the
///           exponent by 1; unequal popcounts drop by 2 (one bonus rung) so
///           the caller's loop discovers the extra reduction next iteration.
///   other:  default to H (always safe, may not reduce on its own).
///
/// T_POWER_and_H[m] encodes the gate string for T^m * H: m = 0 -> "H",
/// m = 1 -> "TH", m = 2 -> "SH" (since S = T^2), m = 3 -> "TSH".
std::pair<Circuit, DOmegaUnitary>
reduce_denomexp(const DOmegaUnitary &unitary) {
  // Built once at first call so we do not allocate a vector of gate strings
  // per reduce_denomexp invocation.
  static const Circuit T_POWER_and_H[] = {
      Circuit({Gate::H}),
      Circuit({Gate::T, Gate::H}),
      Circuit({Gate::S, Gate::H}), // S = T^2
      Circuit({Gate::T, Gate::S, Gate::H}),
  };

  // residue() encodes (a%2, b%2, c%2, d%2) of the ZOmega numerator into a
  // 4-bit integer. residue_squared_z carries the case label below.
  int32_t residue_z = unitary.z().residue();
  int32_t residue_w = unitary.w().residue();
  int32_t residue_squared_z = (unitary.z().u() * unitary.z().conj().u()).residue();

  // T-power offset that aligns the lowest set bit of w to that of z. The
  // negative branch wraps mod 4 since T has order 8 modulo a sign.
  int32_t m = BIT_SHIFT[static_cast<size_t>(residue_w)] -
          BIT_SHIFT[static_cast<size_t>(residue_z)];
  if (m < 0)
    m += 4;

  DOmegaUnitary new_unitary = unitary;
  Circuit gate_seq;

  if (residue_squared_z == 0b0000) {
    new_unitary = with_denom_exp(unitary.mul_by_H_and_T_power_from_left(0),
                                 unitary.k() - 1);
    gate_seq = T_POWER_and_H[0];
  } else if (residue_squared_z == 0b1010) {
    new_unitary = with_denom_exp(unitary.mul_by_H_and_T_power_from_left(-m),
                                 unitary.k() - 1);
    gate_seq = T_POWER_and_H[m];
  } else if (residue_squared_z == 0b0001) {
    if (BIT_COUNT[static_cast<size_t>(residue_z)] ==
        BIT_COUNT[static_cast<size_t>(residue_w)]) {
      new_unitary = with_denom_exp(unitary.mul_by_H_and_T_power_from_left(-m),
                                   unitary.k() - 1);
      gate_seq = T_POWER_and_H[m];
    } else {
      // Bonus reduction: the exponent drops by 2 on this step, so we do not
      // apply with_denom_exp here and instead let the caller's loop pick up
      // the extra rung on its next iteration.
      new_unitary = unitary.mul_by_H_and_T_power_from_left(-m);
      gate_seq = T_POWER_and_H[m];
    }
  } else {
    // Catch-all: H always reduces the exponent by 1 even if the case
    // analysis above did not match.
    new_unitary = with_denom_exp(unitary.mul_by_H_from_left(), unitary.k() - 1);
    gate_seq = T_POWER_and_H[0];
  }

  return {gate_seq, new_unitary};
}

} // namespace

//===----------------------------------------------------------------------===//
// kmm_synthesize
//===----------------------------------------------------------------------===//

/// Exact decomposition of a DOmegaUnitary into a Clifford+T gate sequence
/// in Matsumoto-Amano normal form.
///
/// References: Ross & Selinger arXiv:1403.2975, sec. 7.3 step 3; the exact
/// synthesis algorithm of Kliuchnikov, Maslov, Mosca [10].
///
/// Three phases:
///
///   1. Denominator reduction. While k > 0, reduce_denomexp peels off one
///      gate (or HT / SHT syllable) from the left, dropping k by 1 (or 2).
///      When k = 0 the components z, w are in Z[omega] and U is a
///      Clifford (possibly with a global phase).
///
///   2. Clifford decomposition. The k = 0 unitary is unwound as a T-gate
///      (when the n phase is odd), an X (when z = 0, swapping the
///      components), an omega-power W^m to match z's omega exponent, and
///      S-gates to clear the remaining n phase.
///
///   3. Matsumoto-Amano normalization. normalize_gates absorbs Cliffords,
///      collapses TT -> S, etc., and emits the canonical minimum-T-count
///      representation.
Circuit kmm_synthesize(DOmegaUnitary unitary) {
  CUDAQ_CUDAQ_SYNTH_OPEN_SUB("kmm_synthesize");
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "denom_exp=" << unitary.k() << " (expected T-count ~"
             << unitary.k() << ")\n");

  Circuit gates;

  // Phase 1: peel syllables off the left until the denominator is gone. The
  // number of iterations is exactly the T-count of the synthesized circuit.
  while (unitary.k() > 0) {
    auto [gate_seq, reduced_unitary] = reduce_denomexp(unitary);
    gates += gate_seq;
    unitary = reduced_unitary;
  }

  // Phase 2: undo the remaining Clifford. Each step here corresponds to a
  // generator (T, X, W^m, S^m) whose inverse we left-multiply onto U; the
  // gates we emit are the originals (right-multiplied onto the circuit so
  // far).
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "Clifford phase: n=" << unitary.n() << ", z=" << unitary.z()
             << ", w=" << unitary.w() << '\n');

  // T contributes 1 to n; if n is currently odd, one T brings it to even.
  if (unitary.n() & 1) {
    gates.push_back(Gate::T);
    unitary = unitary.mul_by_T_inv_from_left();
  }

  // z = 0 means the unitary is off-diagonal; X swaps the components so the
  // omega-power search below has a non-zero z to match against.
  if (unitary.z() == DOmega::from_int(0)) {
    gates.push_back(Gate::X);
    unitary = unitary.mul_by_X_from_left();
  }

  // After the X step z is forced to be a unit in Z[omega], i.e. an
  // omega-power. Find that exponent and divide it out as a global W power.
  int32_t m_W = 0;
  for (int32_t m = 0; m < 8; ++m) {
    if (unitary.z().u() == mul_by_omega_power(ZOmega::from_int(1), m)) {
      m_W = m;
      unitary = unitary.mul_by_W_power_from_left(-m_W);
      break;
    }
  }

  // n is now even; S adds 2 to n, so n / 2 S-gates clear it.
  int32_t m_S = unitary.n() >> 1;
  for (int32_t i = 0; i < m_S; ++i)
    gates.push_back(Gate::S);
  unitary = unitary.mul_by_S_power_from_left(-m_S);

  assert(unitary == DOmegaUnitary::identity() &&
         "unitary should be the identity after Clifford decomposition");

  // Emit the trailing global-phase W gates in the order they were peeled.
  for (int32_t i = 0; i < m_W; ++i)
    gates.push_back(Gate::W);

  // Phase 3: normalize. Cannot fail.
  Circuit result = normalize_gates(gates);
  CUDAQ_CUDAQ_SYNTH_CLOSE_SUCCESS("final " + std::to_string(result.size()) +
                      " gates, T-count=" + std::to_string(result.t_count()));
  return result;
}

} // namespace cudaq::synth
