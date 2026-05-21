/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <vector>

#include "Circuit/Clifford.h"
#include "cudaq/Synthesis/Circuit/Circuit.h"

namespace cudaq::synth {

// TODO: expose NormalForm as a first-class type alongside Circuit so callers
// can keep the Matsumoto-Amano structure after normalization instead of
// re-serializing to a gate list. The internal representation already lives
// here:
//
//     struct NormalForm {
//       std::vector<Syllable> syllables;  // T-count == syllables.size()
//       Clifford trailing;
//     };
//
// Pros: O(1) T-count; equality modulo global phase is O(syllables); the
//       structure mirrors the paper's algorithm 1:1; kmm_synthesize would
//       no longer need a syllable -> gate serialization step.
// Cons: Syllable and Clifford become public; gate-by-gate consumers need a
//       to_circuit() bridge; not closed under concatenation
//       (re-normalization required after appending).

//===----------------------------------------------------------------------===//
// Syllable
//===----------------------------------------------------------------------===//

namespace {

/// One step of the Matsumoto-Amano decomposition, used by normalize_gates
/// while it scans the input. `I` is never stored in the syllable list; it
/// only exists so the enum has a zero-initialized sentinel value.
///
///     T   -> T
///     HT  -> H * T
///     SHT -> S * H * T
enum class Syllable { I = 0, T = 1, HT = 2, SHT = 3 };

} // anonymous namespace

//===----------------------------------------------------------------------===//
// normalize_gates
//===----------------------------------------------------------------------===//

/// Compute the Matsumoto-Amano normal form of a single-qubit Clifford+T
/// circuit.
///
/// Reference: Giles & Selinger [7]; Ross & Selinger arXiv:1403.2975, sec. 7.3.
///
/// Normal form. Every single-qubit Clifford+T circuit is uniquely equal to
///
///     syllable_1 * syllable_2 * ... * syllable_m * C
///
/// where each syllable is one of {T, HT, SHT} and C is a trailing Clifford.
/// The number of syllables m is exactly the (minimal) T-count of the
/// circuit's equivalence class.
///
/// Algorithm. Walk the input gate list left-to-right and maintain:
///   syllables -- the list of syllables emitted so far.
///   c         -- the "trailing Clifford" absorbing every non-T gate seen
///                since the last syllable boundary.
///
/// Per gate:
///   * Clifford (H, S, X, W): right-multiply into c. This keeps c as the
///     up-to-date Clifford context that the next T gate will be conjugated
///     past.
///   * T: ask `c.decompose_tconj()` for the syllable type that comes out
///     of c * T = syllable * c'.
///       - Axis::I  -- the syllable would be plain T. If the previous
///                     syllable on the list can merge with this T (the
///                     classic TT -> S, HT * T -> HS, SHT * T -> SHS
///                     collapses), pop it and absorb the combined Clifford
///                     into c. Otherwise push Syllable::T.
///       - Axis::H  -- push Syllable::HT, replace c with c'.
///       - Axis::SH -- push Syllable::SHT, replace c with c'.
///
/// After the loop, expand each syllable into its gate-sequence form and
/// append c.to_circuit() for the trailing Clifford.
///
/// @param input Arbitrary Clifford+T circuit. No failure path: the gate
///              alphabet is structurally restricted to {H, S, T, X, W}.
/// @return      Circuit in Matsumoto-Amano normal form. The empty Circuit
///              represents the identity.
inline Circuit normalize_gates(const Circuit &input) {
  std::vector<Syllable> syllables;
  Clifford c;

  for (Gate g : input) {
    if (g == Gate::H || g == Gate::S || g == Gate::X || g == Gate::W) {
      c = c * Clifford::from_gate(g);
      continue;
    }

    // T gate path. Decompose c * T = syllable * c' to find the syllable
    // type and the updated trailing Clifford.
    auto [axis, new_c] = c.decompose_tconj();

    if (axis == Axis::I) {
      // The candidate syllable is a plain T. If the previous syllable can
      // collapse with it, pop the syllable and fold the resulting Clifford
      // (S, HS, or SHS) into the trailing context. Otherwise the T stands
      // on its own.
      if (!syllables.empty() && syllables.back() == Syllable::T) {
        syllables.pop_back();
        c = CLIFFORD_S * new_c;
      } else if (!syllables.empty() && syllables.back() == Syllable::HT) {
        syllables.pop_back();
        c = CLIFFORD_HS * new_c;
      } else if (!syllables.empty() && syllables.back() == Syllable::SHT) {
        syllables.pop_back();
        c = CLIFFORD_SHS * new_c;
      } else {
        syllables.push_back(Syllable::T);
        c = new_c;
      }
    } else if (axis == Axis::H) {
      syllables.push_back(Syllable::HT);
      c = new_c;
    } else { // axis == Axis::SH
      syllables.push_back(Syllable::SHT);
      c = new_c;
    }
  }

  // Materialize the result directly from the syllable list and the
  // trailing Clifford; no intermediate string round-trip needed.
  Circuit result;
  for (Syllable s : syllables) {
    switch (s) {
    case Syllable::T:
      result.push_back(Gate::T);
      break;
    case Syllable::HT:
      result.push_back(Gate::H);
      result.push_back(Gate::T);
      break;
    case Syllable::SHT:
      result.push_back(Gate::S);
      result.push_back(Gate::H);
      result.push_back(Gate::T);
      break;
    case Syllable::I:
      break; // sentinel, never pushed
    }
  }
  result += c.to_circuit();
  return result;
}

} // namespace cudaq::synth
