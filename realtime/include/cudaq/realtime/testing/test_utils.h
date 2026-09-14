/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file test_utils.h
/// @brief Accessors for the address-as-flag ring protocol, for tests and tools
///        that drive a ring buffer by hand instead of through a dispatcher.
///

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace cudaq::realtime::testing {

/// @brief Read slot `slot`'s flag: 0 when free, otherwise the slot address.
inline std::uint64_t load_flag(std::uint64_t flags_addr, unsigned slot) {
  const auto *flags = reinterpret_cast<const std::uint64_t *>(flags_addr);
  return __atomic_load_n(&flags[slot], __ATOMIC_ACQUIRE);
}

/// @brief Publish (`value` = slot address) or release (`value` = 0) a slot.
/// The release ordering is what makes the slot's payload visible to the
/// consumer that observes the flag.
inline void store_flag(std::uint64_t flags_addr, unsigned slot,
                       std::uint64_t value) {
  auto *flags = reinterpret_cast<std::uint64_t *>(flags_addr);
  __atomic_store_n(&flags[slot], value, __ATOMIC_RELEASE);
}

/// @brief First byte of slot `slot`, given the ring's slot stride.
inline std::uint8_t *slot_data(std::uint64_t data_addr, unsigned slot,
                               std::size_t stride) {
  return reinterpret_cast<std::uint8_t *>(data_addr) + slot * stride;
}

/// @brief Poll until slot `slot` is published, or the timeout expires.
/// @return true if the flag became non-zero.
inline bool wait_for_flag(
    std::uint64_t flags_addr, unsigned slot,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (load_flag(flags_addr, slot) != 0)
      return true;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

/// @brief Poll until slot `slot` is released, or the timeout expires.  This is
/// how a producer observes back-pressure: the slot is reusable once the
/// consumer has cleared it.
/// @return true if the flag became zero.
inline bool wait_for_flag_clear(
    std::uint64_t flags_addr, unsigned slot,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (load_flag(flags_addr, slot) == 0)
      return true;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

} // namespace cudaq::realtime::testing
