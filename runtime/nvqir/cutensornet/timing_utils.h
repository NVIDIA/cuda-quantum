/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/Logger.h"

#ifndef NTIMING
#define LOG_API_TIME() ScopedTraceWithContext(__FUNCTION__);
#else
#define LOG_API_TIME()
#endif

namespace nvqir {
template <typename ItTy>
std::string containerToString(ItTy begin, ItTy end) {
  fmt::basic_memory_buffer<char, 256> buffer;
  fmt::format_to(std::back_inserter(buffer), "[");
  for (ItTy itr = begin; itr != end; ++itr) {
    fmt::format_to(std::back_inserter(buffer), "{}", *itr);
    if (std::next(itr) != end) {
      fmt::format_to(std::back_inserter(buffer), ",");
    }
  }
  fmt::format_to(std::back_inserter(buffer), "]");
  return fmt::to_string(buffer);
}

template <typename ContainerTy>
static inline std::string containerToString(const ContainerTy &container) {
  return containerToString(container.begin(), container.end());
}
} // namespace nvqir
