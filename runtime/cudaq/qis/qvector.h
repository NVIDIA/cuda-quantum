/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "qview.h"

namespace cudaq {

/// @brief A qreg is a container for qudits. This container can be
/// dynamic or compile-time-size specified. By default,
/// the qreg is constructed as a dynamic register (vector-like)
/// of qubits (2-level). This can be changed via the qreg type
/// template parameters.
template <std::size_t Levels = 2>
class qvector {
public:
  /// @brief Useful typedef indicating the underlying qudit type
  using value_type = qudit<Levels>;

private:
  /// @brief If the size is dynamic, then we use vector of qudits,
  /// if not dynamic, use an array.
  std::vector<value_type> qudits;

public:
  /// @brief Construct a qreg with `size` qudits in the |0> state.
  /// Can only be used for dyn sized qregs
  qvector(std::size_t size)
      : qudits(size) {}

  /// @cond
  /// Nullary constructor
  /// meant to be used with kernel_builder<cudaq::qreg<>>
  qvector()
      : qudits(1) {}

  /// @endcond

  /// @brief qregs cannot be copied
  qvector(qvector const &) = delete;
  /// @brief qregs cannot be moved
  qvector(qvector &&) = delete;

  /// @brief Iterator interface, begin.
  auto begin() { return qudits.begin(); }
  auto end() { return qudits.end(); }

  /// @brief Returns the qudit at `idx`.
  value_type &operator[](const std::size_t idx) { return qudits[idx]; }

  /// @brief Returns the `[0, count)` qudits.
  qview< Levels> front(std::size_t count) {
    return std::span(qudits).subspan(0, count);
  }

  /// @brief  Returns the first qudit.
  value_type &front() { return qudits.front(); }

  /// @brief Returns the `[count, size())` qudits.
  qview< Levels> back(std::size_t count) {
    return std::span(qudits).subspan(size() - count, count);
  }

  /// @brief Returns the last qudit.
  value_type &back() { return qudits.back(); }

  /// @brief Returns the `[start, start+size)` qudits.
  qview<Levels> slice(std::size_t start, std::size_t size) {
    return std::span(qudits).subspan(start, size);
  }

  /// @brief Returns the number of contained qudits.
  std::size_t size() const { return qudits.size(); }

  /// @brief Destroys all contained qudits. Postcondition: `size() == 0`.
  void clear() { qudits.clear(); }
};

// Provide the default qreg q(SIZE) deduction guide
// qreg(std::size_t) -> qreg<dyn, 2>;
} // namespace cudaq
