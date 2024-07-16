/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <string>
#include <vector>

namespace cudaq {
/// @brief The `pauli_word` is a thin wrapper on a
/// Pauli tensor product string, e.g. `XXYZ` on 4
// qubits.
class pauli_word {
private:
  std::vector<char> term;

public:
  pauli_word() = default;
  pauli_word(const std::string t) : term(t.begin(), t.end()) {}
  std::string str() const { return std::string(term.begin(), term.end()); }
  const std::vector<char> &data() const { return term; }
};
} // namespace cudaq