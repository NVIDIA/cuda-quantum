/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Circuit/Gate.h"
#include "cudaq/Synthesis/Support/Result.h"

#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::synth {

/// A single-qubit Clifford+T circuit: an ordered, finite sequence of gates.
///
/// A Circuit represents the product U = g₁ · g₂ · … · gₙ where each gᵢ is
/// one of the standard generators acting on a single qubit. The empty circuit
/// is the identity operator.
///
/// The only valid gate values are those of the Gate enum; structurally invalid
/// sequences cannot be constructed.
class Circuit {
  std::vector<Gate> gates_;

public:
  /// Construct an empty circuit (identity operator).
  Circuit() = default;

  /// Construct a circuit from a brace-enclosed gate list, e.g.
  /// Circuit({Gate::H, Gate::T, Gate::H}).
  Circuit(std::initializer_list<Gate> g) : gates_(g) {}

  // -------------------------------------------------------------------------
  // Iteration
  // -------------------------------------------------------------------------

  /// Forward iteration over gates in application order (left to right).
  auto begin() const { return gates_.begin(); }
  auto end() const { return gates_.end(); }

  /// Reverse iteration, used when reconstructing a unitary from a circuit
  /// via right-to-left gate application (see DOmegaUnitary::from_gates).
  auto rbegin() const { return gates_.rbegin(); }
  auto rend() const { return gates_.rend(); }

  // -------------------------------------------------------------------------
  // Capacity and element access
  // -------------------------------------------------------------------------

  /// Number of gates in the circuit.
  size_t size() const { return gates_.size(); }

  /// True iff the circuit is the identity (no gates).
  bool empty() const { return gates_.empty(); }

  /// Reserve storage for at least n gates to avoid repeated reallocations
  /// when building a circuit incrementally.
  void reserve(size_t n) { gates_.reserve(n); }

  /// Gate at position i (0-indexed, unchecked).
  Gate operator[](size_t i) const { return gates_[i]; }

  /// Last gate in the circuit. Undefined behaviour if the circuit is empty.
  Gate back() const { return gates_.back(); }

  // -------------------------------------------------------------------------
  // Mutation
  // -------------------------------------------------------------------------

  /// Append a single gate at the end of the circuit.
  void push_back(Gate g) { gates_.push_back(g); }

  /// Remove the last gate. Undefined behaviour if the circuit is empty.
  void pop_back() { gates_.pop_back(); }

  /// Append all gates of rhs to the end of this circuit (concatenation).
  Circuit &operator+=(const Circuit &rhs) {
    gates_.insert(gates_.end(), rhs.begin(), rhs.end());
    return *this;
  }

  // -------------------------------------------------------------------------
  // Comparison
  // -------------------------------------------------------------------------

  /// Two circuits are equal iff they contain the same gates in the same order.
  /// Note: unitary equivalence (equality up to global phase or Clifford
  /// relations) is a strictly weaker condition and is not checked here.
  bool operator==(const Circuit &other) const { return gates_ == other.gates_; }
  bool operator!=(const Circuit &other) const { return !(*this == other); }

  // -------------------------------------------------------------------------
  // Metrics
  // -------------------------------------------------------------------------

  /// Number of T gates in the circuit, which equals the T-count and
  /// determines the non-Clifford cost. O(n) scan over all gates.
  int t_count() const {
    int n = 0;
    for (Gate g : gates_)
      if (g == Gate::T)
        ++n;
    return n;
  }

  // -------------------------------------------------------------------------
  // Serialization
  // -------------------------------------------------------------------------

  /// Serialize to a human-readable gate string (e.g. "HTSHTSH").
  /// The empty circuit serializes to "I" (the identity sentinel).
  std::string to_string() const;

  /// Parse a gate string into a Circuit. Each character must be one of
  /// H, S, T, X, W. The character 'I' is accepted as the identity sentinel
  /// and produces an empty Circuit. Any other character returns failure().
  static FailureOr<Circuit> from_string(std::string_view s);
};

inline std::string Circuit::to_string() const {
  if (gates_.empty())
    return "I";
  std::string result;
  result.reserve(gates_.size());
  for (Gate g : gates_) {
    switch (g) {
    case Gate::H:
      result += 'H';
      break;
    case Gate::S:
      result += 'S';
      break;
    case Gate::T:
      result += 'T';
      break;
    case Gate::X:
      result += 'X';
      break;
    case Gate::W:
      result += 'W';
      break;
    }
  }
  return result;
}

inline FailureOr<Circuit> Circuit::from_string(std::string_view s) {
  Circuit result;
  result.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case 'H':
      result.push_back(Gate::H);
      break;
    case 'S':
      result.push_back(Gate::S);
      break;
    case 'T':
      result.push_back(Gate::T);
      break;
    case 'X':
      result.push_back(Gate::X);
      break;
    case 'W':
      result.push_back(Gate::W);
      break;
    case 'I':
      break; // identity sentinel — produces empty Circuit
    default:
      return failure();
    }
  }
  return result;
}

/// Stream a Circuit as its gate string (see Circuit::to_string()).
inline std::ostream &operator<<(std::ostream &os, const Circuit &c) {
  return os << c.to_string();
}

} // namespace cudaq::synth
