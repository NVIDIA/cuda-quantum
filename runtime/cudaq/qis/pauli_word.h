/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <ctype.h>
#include <string>

namespace cudaq {

/// @brief The `pauli_word` is a thin wrapper on a Pauli tensor product string,
/// e.g. `XXYZ` on 4 qubits.
class pauli_word {
public:
  pauli_word() = default;
  pauli_word(std::string &&t) : term{std::move(t)} { to_upper_case(); }
  pauli_word(const std::string &t) : term(t) { to_upper_case(); }
  pauli_word(const char *const p) : term{p} { to_upper_case(); }
  pauli_word &operator=(const std::string &t) {
    term = t;
    to_upper_case();
    return *this;
  }
  pauli_word &operator=(const char *const p) {
    term = p;
    to_upper_case();
    return *this;
  }

  std::string str() const { return term; }

  // TODO: Obsolete? Used by KernelWrapper.h only.
  const std::vector<char> data() const { return {term.begin(), term.end()}; }

private:
  // Convert the string member to upper case at construction/assignment.
  // TODO: This should probably verify the string contains only letters valid in
  // this alphabet: I, X, Y, and Z.
  void to_upper_case() {
    std::transform(term.begin(), term.end(), term.begin(), ::toupper);
  }

  // These methods used by the compiler.
  __attribute__((used)) const char *_nvqpp_data() const { return term.data(); }
  __attribute__((used)) std::uint64_t _nvqpp_size() const {
    return term.size();
  }

  std::string term; ///< Pauli words are string-like.
};

namespace details {
static_assert(sizeof(std::string) == sizeof(pauli_word));
// This constant used by the compiler.
static constexpr std::uint64_t _nvqpp_sizeof = sizeof(pauli_word);
} // namespace details
} // namespace cudaq
