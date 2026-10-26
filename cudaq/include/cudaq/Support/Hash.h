/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <functional>

namespace cudaq::detail {

/// Combine \p val into an existing hash \p seed (boost-style).
template <typename T>
inline void hashCombine(std::size_t &seed, const T &val) {
  std::hash<T> hasher;
  seed ^= hasher(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/// Hash an arbitrary list of values into a single seed.
template <typename... Args>
inline std::size_t hashVal(const Args &...args) {
  std::size_t seed = 0;
  (hashCombine(seed, args), ...);
  return seed;
}

} // namespace cudaq::detail
