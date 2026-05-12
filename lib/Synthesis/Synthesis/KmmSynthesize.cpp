/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Synthesis/Synthesis/KmmSynthesize.h"
#include "Circuit/NormalForm.h"
#include "Support/LogMacros.h"

namespace cudaq::synth {

namespace {

// Lookup tables for the denominator reduction algorithm.
// - BIT_SHIFT[r] gives the number of trailing zero bits in the 4-bit residue r.
// - BIT_COUNT[r] gives the popcount of the 4-bit residue r.
//
// These are used to determine which gate sequence reduces the denominator
// exponent of a DOmegaUnitary by 1.

inline constexpr std::array<i32, 16> BIT_SHIFT = {0, 0, 1, 0, 2, 0, 1, 3,
                                                  3, 3, 0, 2, 2, 1, 0, 0};

inline constexpr std::array<i32, 16> BIT_COUNT = {0, 1, 1, 2, 1, 2, 2, 3,
                                                  1, 2, 2, 3, 2, 3, 3, 4};

/// Reduces the denominator exponent of a DOmegaUnitary by applying a single
/// gate (or short gate sequence) from the left.
///
/// Reference: Kliuchnikov, Maslov, Mosca [10] (exact synthesis),
/// and Ross & Selinger, arXiv:1403.2975, §7.3 step 3.
///
/// The key insight from [10]: Given U with denominator exponent k,
/// the matrix √2^k · U has all entries in Z[ω]. The residues of
/// these entries modulo 2 determine which gate g ∈ {H, T^m·H, S·H, T·S·H}
/// makes g† · U have denominator exponent k-1.
///
/// The residue pattern of z†z (mod 2) determines the case:
/// - 0b0000: z is divisible by √2, so H alone suffices (H swaps z↔w).
/// - 0b1010: the "generic" case; apply T^{-m}·H where m aligns the
///   trailing bits of w and z.
/// - 0b0001: subcases depend on whether popcount(z) = popcount(w).
///   If equal, the exponent drops by 1; if not, it drops by 2 (bonus).
/// - default: fallback to H (always safe, may not always reduce by 1).
///
/// T_POWER_and_H[m] encodes the gate string for T^m · H:
///   m=0 → "H", m=1 → "TH", m=2 → "SH" (since S = T²), m=3 → "TSH".
std::pair<Circuit, DOmegaUnitary>
reduce_denomexp(const DOmegaUnitary &unitary) {
  // T^m · H gate sequences for m = 0..3.
  // Static to avoid a heap allocation on every call to reduce_denomexp
  // (previously a std::vector<std::string> was constructed here each time).
  static const Circuit T_POWER_and_H[] = {
      Circuit({Gate::H}),
      Circuit({Gate::T, Gate::H}),
      Circuit({Gate::S, Gate::H}),         // S = T²
      Circuit({Gate::T, Gate::S, Gate::H}),
  };

  // Compute residues modulo 2 of the ZOmega numerators.
  // residue() returns a 4-bit integer encoding (a%2, b%2, c%2, d%2).
  i32 residue_z = unitary.z().residue();
  i32 residue_w = unitary.w().residue();
  // residue of z†z = z · conj(z) determines the case structure.
  i32 residue_squared_z = (unitary.z().u() * unitary.z().conj().u()).residue();

  // m = amount of T-power needed to align the bit patterns of w and z.
  // BIT_SHIFT gives the position of the lowest set bit.
  i32 m = BIT_SHIFT[static_cast<size_t>(residue_w)] -
          BIT_SHIFT[static_cast<size_t>(residue_z)];
  if (m < 0)
    m += 4;

  DOmegaUnitary new_unitary = unitary;
  Circuit gate_seq;

  if (residue_squared_z == 0b0000) {
    // Case: z†z ≡ 0 (mod 2), meaning z is divisible by √2.
    // Applying H swaps z and w, and the new z (= old w) has lower
    // denominator exponent.
    new_unitary = with_denom_exp(unitary.mul_by_H_and_T_power_from_left(0),
                                 unitary.k() - 1);
    gate_seq = T_POWER_and_H[0];
  } else if (residue_squared_z == 0b1010) {
    // Case: z†z ≡ (1,0,1,0) (mod 2). Apply T^{-m}·H to align and reduce.
    new_unitary = with_denom_exp(unitary.mul_by_H_and_T_power_from_left(-m),
                                 unitary.k() - 1);
    gate_seq = T_POWER_and_H[m];
  } else if (residue_squared_z == 0b0001) {
    // Case: z†z ≡ (0,0,0,1) (mod 2). Two subcases based on popcount.
    if (BIT_COUNT[static_cast<size_t>(residue_z)] ==
        BIT_COUNT[static_cast<size_t>(residue_w)]) {
      // Equal popcounts: standard reduction by 1.
      new_unitary = with_denom_exp(unitary.mul_by_H_and_T_power_from_left(-m),
                                   unitary.k() - 1);
      gate_seq = T_POWER_and_H[m];
    } else {
      // Unequal popcounts: denominator drops by 2 (bonus reduction).
      // No explicit with_denomexp—reduce() in the caller will
      // handle the extra reduction on the next iteration.
      new_unitary = unitary.mul_by_H_and_T_power_from_left(-m);
      gate_seq = T_POWER_and_H[m];
    }
  } else {
    // Default fallback: apply H.
    new_unitary = with_denom_exp(unitary.mul_by_H_from_left(), unitary.k() - 1);
    gate_seq = T_POWER_and_H[0];
  }

  return {gate_seq, new_unitary};
}

} // namespace

/// decompose_domega_unitary: Exact decomposition of a DOmegaUnitary into
/// a Clifford+T gate sequence in Matsumoto-Amano normal form.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §7.3 step 3, using
/// the exact synthesis algorithm of Kliuchnikov, Maslov, Mosca [10].
///
/// Algorithm:
/// Phase 1 - Denominator reduction:
///   While k > 0, apply _reduce_denomexp to peel off one gate (or short
///   gate sequence) from the left, reducing k by 1 (sometimes 2).
///   After this loop, U has k = 0, meaning z, w ∈ Z[ω], so U is a
///   Clifford operator (possibly with a global phase).
///
/// Phase 2 - Clifford decomposition:
///   The remaining Clifford U (with k = 0) is decomposed into:
///   - A possible T gate if the phase n is odd (T adjusts ω-phase by 1).
///   - A possible X gate if z = 0 (swaps z and w components).
///   - W^m gates to match the ω-power in z (W = global phase ω).
///   - S^m gates to match the remaining n-phase.
///
/// Phase 3 - Normalization:
///   The raw gate string is converted to Matsumoto-Amano normal form
///   via normalize_gates(), which absorbs Cliffords and simplifies
///   syllable sequences (e.g., TT → S). The result is the canonical
///   gate sequence with minimum T-count.
Circuit kmm_synthesize(DOmegaUnitary unitary) {
  CUDAQ_SYNTH_LOG_DEBUG("synth.kmm",
                   "kmm_synthesize: denom_exp={} (expected T-count ~{})",
                   unitary.k(), unitary.k());

  Circuit gates;

  // Phase 1: Reduce denominator exponent k to 0.
  // Each iteration peels off one T-gate (or HT/SHT syllable),
  // reducing k by 1. The total number of iterations equals the
  // T-count of the synthesized circuit.
  while (unitary.k() > 0) {
    auto [gate_seq, reduced_unitary] = reduce_denomexp(unitary);
    gates += gate_seq;
    unitary = reduced_unitary;
  }

  // Phase 2: Decompose the remaining Clifford (k = 0).
  CUDAQ_SYNTH_LOG_TRACE("synth.kmm",
                   "kmm_synthesize: Clifford phase -- n={}, z={}, w={}",
                   unitary.n(), unitary.z(), unitary.w());
  // If phase n is odd, absorb one T to make it even (T adds 1 to n).
  if (unitary.n() & 1) {
    gates.push_back(Gate::T);
    unitary = unitary.mul_by_T_inv_from_left();
  }

  // If z = 0, the unitary is off-diagonal; apply X to swap z and w.
  if (unitary.z() == DOmega::from_int(0)) {
    gates.push_back(Gate::X);
    unitary = unitary.mul_by_X_from_left();
  }

  // Now z must be a unit in Z[ω], i.e., z = ω^m for some m.
  // Find m and apply W^{-m} (global phase removal).
  i32 m_W = 0;
  for (i32 m = 0; m < 8; ++m) {
    if (unitary.z().u() == mul_by_omega_power(ZOmega::from_int(1), m)) {
      m_W = m;
      unitary = unitary.mul_by_W_power_from_left(-m_W);
      break;
    }
  }

  // Apply S gates to clear the remaining phase exponent n.
  // Since n is now even, m_S = n/2 applications of S (which adds 2 to n).
  i32 m_S = unitary.n() >> 1;
  for (i32 i = 0; i < m_S; ++i) gates.push_back(Gate::S);
  unitary = unitary.mul_by_S_power_from_left(-m_S);

  assert(unitary == DOmegaUnitary::identity() &&
         "unitary should be the identity after Clifford decomposition");

  // Apply W (global phase) gates.
  for (i32 i = 0; i < m_W; ++i) gates.push_back(Gate::W);

  // Phase 3: Convert the raw Circuit to Matsumoto-Amano normal form.
  // This absorbs Cliffords, simplifies TT → S, and produces the
  // canonical representation with minimum T-count. Cannot fail.
  Circuit result = normalize_gates(gates);
  CUDAQ_SYNTH_LOG_DEBUG("synth.kmm",
                   "kmm_synthesize: final circuit has {} gates, T-count={}",
                   result.size(), result.t_count());
  return result;
}

} // namespace cudaq::synth
