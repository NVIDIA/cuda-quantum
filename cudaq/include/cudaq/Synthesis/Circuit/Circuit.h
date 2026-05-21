/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Circuit/Gate.h"
#include "llvm/Support/LogicalResult.h"

#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Circuit
//===----------------------------------------------------------------------===//

/// A single-qubit Clifford+T circuit: an ordered, finite sequence of gates.
///
/// A Circuit represents the product U = g_1 * g_2 * ... * g_n where each
/// g_i is one of the standard single-qubit generators (see `Gate`). The
/// empty circuit is the identity operator.
///
/// The gate alphabet is structurally restricted to the `Gate` enum, so
/// invalid sequences cannot be constructed directly. The `from_string`
/// factory is the only entry point that can fail (unknown character).
class Circuit {
  std::vector<Gate> gates_;

public:
  /// Empty circuit -- the identity operator.
  Circuit() = default;

  /// Brace-init from a gate list: Circuit({Gate::H, Gate::T, Gate::H}).
  Circuit(std::initializer_list<Gate> g) : gates_(g) {}

  // -- Iteration --

  /// Forward iteration in application order (left to right).
  auto begin() const { return gates_.begin(); }
  auto end() const { return gates_.end(); }

  /// Reverse iteration. Used when reconstructing a unitary from a circuit
  /// via right-to-left gate application (see DOmegaUnitary::from_gates).
  auto rbegin() const { return gates_.rbegin(); }
  auto rend() const { return gates_.rend(); }

  // -- Capacity and element access --

  size_t size() const { return gates_.size(); }

  /// True iff the circuit is the identity.
  bool empty() const { return gates_.empty(); }

  /// Reserve storage for at least n gates so incremental builders avoid
  /// repeated reallocations.
  void reserve(size_t n) { gates_.reserve(n); }

  /// Unchecked element access (0-indexed).
  Gate operator[](size_t i) const { return gates_[i]; }

  /// Last gate. Calling this on an empty Circuit is undefined behaviour.
  Gate back() const { return gates_.back(); }

  // -- Mutation --

  void push_back(Gate g) { gates_.push_back(g); }

  /// Drop the last gate. Calling this on an empty Circuit is undefined
  /// behaviour.
  void pop_back() { gates_.pop_back(); }

  /// Concatenate `rhs` onto the end.
  Circuit &operator+=(const Circuit &rhs) {
    gates_.insert(gates_.end(), rhs.begin(), rhs.end());
    return *this;
  }

  // -- Comparison --

  /// Sequence equality. Unitary equivalence (e.g. modulo global phase or
  /// Clifford relations) is a strictly weaker notion and is *not* checked
  /// here.
  bool operator==(const Circuit &other) const { return gates_ == other.gates_; }
  bool operator!=(const Circuit &other) const { return !(*this == other); }

  // -- Metrics --

  /// Number of T gates -- the T-count, i.e. the non-Clifford cost of the
  /// circuit. O(n) scan over the gate list.
  int t_count() const {
    int n = 0;
    for (Gate g : gates_)
      if (g == Gate::T)
        ++n;
    return n;
  }

  // -- Serialization --

  /// Encode the circuit as a gate string (e.g. "HTSHTSH"). The empty
  /// circuit becomes the single-character sentinel "I".
  std::string to_string() const;

  /// Parse a gate string. Each character must be one of H, S, T, X, W; the
  /// special character 'I' is accepted as the identity sentinel and yields
  /// an empty Circuit. Any other character returns failure().
  static llvm::FailureOr<Circuit> from_string(std::string_view s);
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

inline llvm::FailureOr<Circuit> Circuit::from_string(std::string_view s) {
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
      break; // identity sentinel: consumed but not pushed.
    default:
      return llvm::failure();
    }
  }
  return result;
}

/// Stream as the gate string produced by `Circuit::to_string`.
inline std::ostream &operator<<(std::ostream &os, const Circuit &c) {
  return os << c.to_string();
}

} // namespace cudaq::synth
