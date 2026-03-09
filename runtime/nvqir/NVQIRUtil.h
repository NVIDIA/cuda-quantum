/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "QIRTypes.h"

namespace nvqir {
// Singleton struct to track allocated arrays
// This facilitates cleaning up arrays at program end to avoid memory leaks.
struct ArrayTracker {
  // Get a thread-local singleton instance of ArrayTracker.
  // This allows each thread to track its own arrays, e.g., in async. execution.
  static ArrayTracker &getInstance();

  // Track the given array
  void track(Array *arr);

  // Untrack the given array (manually delete outside the tracker)
  void untrack(Array *arr);

  // Clean up all tracked arrays
  void clear();

private:
  ArrayTracker() = default;
  ~ArrayTracker() = default;
  ArrayTracker(const ArrayTracker &) = delete;
  ArrayTracker &operator=(const ArrayTracker &) = delete;
  // The tracked arrays
  std::vector<Array *> allocated_arrays;
};

} // namespace nvqir
