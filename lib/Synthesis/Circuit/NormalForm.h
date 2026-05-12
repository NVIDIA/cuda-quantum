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

// TODO: Consider exposing NormalForm as a first-class type alongside Circuit
// rather than discarding the Matsumoto-Amano structure after normalization.
// The internal decomposition already exists here:
//
//   struct NormalForm {
//     std::vector<Syllable> syllables; // T-count = syllables.size() (O(1))
//     Clifford trailing;
//   };
//
// Trade-offs:
//   + T-count is O(1); equality ignoring global phase is O(syllables).
//   + Eliminates the syllable-to-Circuit serialization step at the end of
//     normalize_gates, since kmm_synthesize could return NormalForm directly.
//   + Syllable structure maps directly to the paper's algorithm.
//   - Syllable and Clifford become part of the public API surface.
//   - Gate-by-gate consumers need a to_circuit() conversion.
//   - Not closed under concatenation (re-normalization required).

namespace {

// Syllable types used internally by normalize_gates.
//
// Each syllable encodes one "step" in the Matsumoto-Amano decomposition:
//   T   → T
//   HT  → H · T
//   SHT → S · H · T
// Syllable::I is never stored in the list; it exists only to give the
// enum a zero-initialized sentinel.
enum class Syllable { I = 0, T = 1, HT = 2, SHT = 3 };

} // anonymous namespace

/// Compute the Matsumoto-Amano normal form of a single-qubit Clifford+T
/// circuit.
///
/// Reference: Giles & Selinger [7], and Ross & Selinger, arXiv:1403.2975, §7.3.
///
/// ## Normal form structure
///
/// Every single-qubit Clifford+T circuit is uniquely equivalent to:
///
///   syllable_1 · syllable_2 · … · syllable_m · C
///
/// where each syllable ∈ {T, HT, SHT} and C is a trailing Clifford. The number
/// of syllables m equals the T-count of the circuit and is minimal for its
/// equivalence class.
///
/// ## Algorithm
///
/// The function scans the input circuit left-to-right, maintaining two
/// pieces of state:
///
///   syllables — the list of syllables accumulated so far.
///   c         — the "trailing Clifford" absorbing all un-syllabified gates.
///
/// For each gate:
///
///   Clifford gates (H, S, X, W): right-multiply into c.  This keeps c
///     up to date as the Clifford context that the next T gate will be
///     conjugated past.
///
///   T gate: call c.decompose_tconj() to determine the syllable type that
///     results from pushing T past c (i.e. c·T = syllable·c').
///     - Axis::I  → next syllable would be T.  Check if the last syllable
///       already in the list can be merged (T·T → S, HT·T → HS,
///       SHT·T → SHS absorbed into c).  If so, pop it and absorb the
///       combined Clifford into c.  Otherwise, push Syllable::T.
///     - Axis::H  → push Syllable::HT,  set c = c'.
///     - Axis::SH → push Syllable::SHT, set c = c'.
///
/// After processing all gates, expand each syllable to its gate sequence
/// and append c.to_circuit() for the trailing Clifford.
///
/// @param input  Input circuit. Any sequence of Gate values is valid;
///               no failure path exists since the gate type is structurally
///               constrained to {H, S, T, X, W}.
/// @return       Circuit in Matsumoto-Amano normal form. An empty Circuit
///               represents the identity (use Circuit::to_string() for "I").
inline Circuit normalize_gates(const Circuit &input) {
  std::vector<Syllable> syllables;
  Clifford c;

  for (Gate g : input) {
    if (g == Gate::H || g == Gate::S || g == Gate::X || g == Gate::W) {
      // Clifford gate: absorb into the trailing Clifford context.
      c = c * Clifford::from_gate(g);
      continue;
    }
    // Decompose c·T = syllable·c' to determine the next syllable type
    // and the updated trailing Clifford c'.
    auto [axis, new_c] = c.decompose_tconj();

    if (axis == Axis::I) {
      // The result is a plain T syllable. Check whether the previous syllable
      // can be merged with this T to cancel (TT = S absorbed into the
      // Clifford, and similarly for HT·T and SHT·T).
      if (!syllables.empty() && syllables.back() == Syllable::T) {
        syllables.pop_back();
        c = CLIFFORD_S * new_c; // T·T → S absorbed into c
      } else if (!syllables.empty() && syllables.back() == Syllable::HT) {
        syllables.pop_back();
        c = CLIFFORD_HS * new_c; // HT·T → HS absorbed into c
      } else if (!syllables.empty() && syllables.back() == Syllable::SHT) {
        syllables.pop_back();
        c = CLIFFORD_SHS * new_c; // SHT·T → SHS absorbed into c
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

  // Build the result Circuit directly from syllables + trailing Clifford.
  // No string serialization roundtrip needed.
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
