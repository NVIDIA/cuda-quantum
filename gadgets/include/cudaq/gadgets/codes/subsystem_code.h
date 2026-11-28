/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/gadgets/utils.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::gadgets {
inline namespace v0 {

/// A type alias for a pair of SparsePauliStrings:
/// Representatives of X and Z for one qubit.
using PauliPair = std::pair<SparsePauliString, SparsePauliString>;

/// A subsystem code: k logical qubits and k' gauge qubits (unused logical
/// qubits) encoded in n physical qubits.
struct SubsystemCode {
  /// Throws std::invalid_argument unless:
  /// - independent indexes generators;
  /// - every Pauli acts on qubits below num_qubits;
  /// - r + k + k' = n, for r independent generators.
  /// TODO: Check commutation relations and independence.
  SubsystemCode(std::string name, std::size_t num_qubits,
                std::vector<SparsePauliString> generators,
                std::vector<std::size_t> independent,
                std::vector<PauliPair> logicals,
                std::vector<PauliPair> gauges = {});

  /// Unique name, to refer to the code unambiguously.
  const std::string &get_name() const { return name; }
  /// n, the number of physical qubits.
  std::size_t get_num_qubits() const { return num_qubits; }
  /// Stabilizer generators, possibly over-complete.
  const std::vector<SparsePauliString> &get_generators() const {
    return generators;
  }
  /// Indices of generators that form an independent generating set.
  const std::vector<std::size_t> &get_independent() const {
    return independent;
  }
  /// One X/Z pair per logical qubit; k pairs.
  const std::vector<PauliPair> &get_logicals() const { return logicals; }
  /// One X/Z pair per gauge qubit; k' pairs.
  const std::vector<PauliPair> &get_gauges() const { return gauges; }

private:
  std::string name;
  std::size_t num_qubits;
  std::vector<SparsePauliString> generators;
  std::vector<std::size_t> independent;
  std::vector<PauliPair> logicals;
  std::vector<PauliPair> gauges;
};

} // namespace v0
} // namespace cudaq::gadgets
