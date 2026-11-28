/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#include "cudaq/gadgets/codes/subsystem_code.h"

#include <stdexcept>
#include <utility>

namespace cudaq::gadgets {
inline namespace v0 {

SubsystemCode::SubsystemCode(std::string name, std::size_t num_qubits,
                             std::vector<SparsePauliString> generators,
                             std::vector<std::size_t> independent,
                             std::vector<PauliPair> logicals,
                             std::vector<PauliPair> gauges)
    : name(std::move(name)), num_qubits(num_qubits),
      generators(std::move(generators)), independent(std::move(independent)),
      logicals(std::move(logicals)), gauges(std::move(gauges)) {
  auto fail = [this](const std::string &message) {
    throw std::invalid_argument("code '" + this->name + "': " + message);
  };
  for (std::size_t i : this->independent)
    if (i >= this->generators.size())
      fail("independent generator " + std::to_string(i) + " does not exist");
  auto check_qubits = [&](const SparsePauliString &pauli) {
    for (const auto &[op, qubit] : pauli.ops)
      if (qubit >= num_qubits)
        fail("Pauli acts on qubit " + std::to_string(qubit) +
             " outside the code");
  };
  for (const SparsePauliString &generator : this->generators)
    check_qubits(generator);
  for (const auto *pairs : {&this->logicals, &this->gauges})
    for (const auto &[x, z] : *pairs) {
      check_qubits(x);
      check_qubits(z);
    }
  // Each independent generator, logical pair, and gauge pair accounts for one
  // qubit.
  if (this->independent.size() + this->logicals.size() + this->gauges.size() !=
      num_qubits)
    fail("r + k + k' must be n");
}

} // namespace v0
} // namespace cudaq::gadgets
