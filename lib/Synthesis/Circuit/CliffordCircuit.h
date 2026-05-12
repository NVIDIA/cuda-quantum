/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cudaq::synth {

// ---------------------------------------------------------------------------
// CliffordGateKind
// ---------------------------------------------------------------------------
//
// The set of gates needed to express any n-qubit Clifford circuit:
//   Single-qubit: H, S, X, Z
//   Two-qubit:    CX (CNOT), CZ

enum class CliffordGateKind : uint8_t {
  H = 0, ///< Hadamard
  S,     ///< Phase gate diag(1, i)
  X,     ///< Pauli-X (bit-flip)
  Z,     ///< Pauli-Z (phase-flip)
  CX,    ///< Controlled-X (CNOT); control = qubit0, target = qubit1
  CZ,    ///< Controlled-Z; qubit0 and qubit1 are symmetric
};

// ---------------------------------------------------------------------------
// CliffordGate
// ---------------------------------------------------------------------------
//
// A single gate in a CliffordCircuit. For single-qubit gates only qubit0 is
// meaningful; for CX/CZ both qubit0 (control) and qubit1 (target) are used.

struct CliffordGate {
  CliffordGateKind kind;
  size_t qubit0;
  size_t qubit1 = 0;

  // -- Convenience factories --

  static CliffordGate h(size_t q) noexcept { return {CliffordGateKind::H, q}; }
  static CliffordGate s(size_t q) noexcept { return {CliffordGateKind::S, q}; }
  static CliffordGate x(size_t q) noexcept { return {CliffordGateKind::X, q}; }
  static CliffordGate z(size_t q) noexcept { return {CliffordGateKind::Z, q}; }
  static CliffordGate cx(size_t control, size_t target) noexcept {
    return {CliffordGateKind::CX, control, target};
  }
  static CliffordGate cz(size_t control, size_t target) noexcept {
    return {CliffordGateKind::CZ, control, target};
  }

  bool is_two_qubit() const noexcept {
    return kind == CliffordGateKind::CX || kind == CliffordGateKind::CZ;
  }
};

// ---------------------------------------------------------------------------
// CliffordCircuit
// ---------------------------------------------------------------------------
//
// An ordered sequence of CliffordGate values representing an n-qubit Clifford
// circuit. Gate application order is left-to-right (earliest gate at index 0).

class CliffordCircuit {
public:
  explicit CliffordCircuit(size_t nb_qubits) : nb_qubits_(nb_qubits) {}

  size_t nb_qubits() const noexcept { return nb_qubits_; }
  size_t size() const noexcept { return gates_.size(); }
  bool empty() const noexcept { return gates_.empty(); }

  void push_back(CliffordGate g) {
    assert(g.qubit0 < nb_qubits_);
    assert(!g.is_two_qubit() || g.qubit1 < nb_qubits_);
    gates_.push_back(g);
  }

  void reserve(size_t n) { gates_.reserve(n); }

  CliffordGate operator[](size_t i) const { return gates_[i]; }

  auto begin() const noexcept { return gates_.begin(); }
  auto end() const noexcept { return gates_.end(); }
  auto rbegin() const noexcept { return gates_.rbegin(); }
  auto rend() const noexcept { return gates_.rend(); }

private:
  size_t nb_qubits_;
  std::vector<CliffordGate> gates_;
};

} // namespace cudaq::synth
