/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <vector>

namespace cudaq {

/// @brief The QuditIdTracker tracks unique integer
/// indices for allocated qudits. It will recycle indices
/// when the qudit is deallocated, so that it can be reused.
class QuditIdTracker {
private:
  /// @brief The current identifier.
  std::size_t currentId = 0;

  /// @brief The queue of recycled qubits.
  std::vector<std::size_t> recycledQudits;

public:
  QuditIdTracker() = default;
  QuditIdTracker(const QuditIdTracker &) = delete;

  /// @brief Return the next available index,
  /// take from the recycled qudit indentifiers
  /// if possible.
  std::size_t getNextIndex() {
    if (recycledQudits.empty()) {
      std::size_t ret = currentId;
      currentId++;
      return ret;
    }

    auto next = recycledQudits.back();
    recycledQudits.pop_back();
    return next;
  }

  /// @brief Return indices due to qudit deallocation.
  /// If all qudits have been deallocated, reset the
  /// tracker.
  void returnIndex(std::size_t idx) {
    recycledQudits.push_back(idx);
    std::sort(recycledQudits.begin(), recycledQudits.end(),
              std::greater<std::size_t>());
    if (recycledQudits.size() == currentId) {
      currentId = 0;
      recycledQudits.clear();
    }
  }

  /// @brief Return true if all qudits have been deallocated.
  bool allDeallocated() {
    // Either current id is 0 and we don't have recycled bits,
    return currentId == 0 && recycledQudits.empty();
  }

  /// @brief Return the number of qudits allocated.
  std::size_t numAllocated() { return currentId - recycledQudits.size(); }
};

} // namespace cudaq