/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <string>

namespace cudaq {
/// @brief The `pauli_word` is a thin wrapper on a Pauli tensor product string,
/// e.g. `XXYZ` on 4 qubits.
class pauli_word {
private:
  std::string term;

public:
  pauli_word() = default;
  pauli_word(std::string &&t) : term{std::move(t)} {}
  pauli_word(const std::string &t) : term(t) {}
  pauli_word(const char *const p) : term{p} {}
  pauli_word &operator=(const std::string &t) {
    term = t;
    return *this;
  }
  pauli_word &operator=(const char *const p) {
    term = p;
    return *this;
  }

  std::string str() const { return term; }

  // TODO: Obsolete? Used by KernelWrapper.h only.
  const std::vector<char> data() const { return {term.begin(), term.end()}; }
};
} // namespace cudaq
