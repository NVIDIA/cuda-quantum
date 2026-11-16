/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cudaq::realtime::testing::hsb {

constexpr std::uint64_t nanoseconds_per_second = 1'000'000'000;
constexpr unsigned ila_timestamp_bit_offset = 521 % 32;
constexpr std::uint32_t player_timestamp_enable_mask = std::uint32_t{1} << 3;

// The player's ptp_bram_ena bit controls request timestamp insertion.
constexpr bool timestamp_insertion_enabled(std::uint32_t player_enable) {
  return (player_enable & player_timestamp_enable_mask) != 0;
}

// Match the playback tools' {seconds[31:0], nanoseconds[31:0]} encoding.
constexpr std::uint64_t pack_timestamp(std::uint64_t nanoseconds) {
  return (std::uint64_t(std::uint32_t(nanoseconds / nanoseconds_per_second))
          << 32) |
         (nanoseconds % nanoseconds_per_second);
}

// Request and response timestamps use the same host monotonic clock.
inline std::uint64_t monotonic_timestamp() {
  const auto elapsed = std::chrono::steady_clock::now().time_since_epoch();
  return pack_timestamp(
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

// Stamp the outgoing copy, never the cached BRAM image. Like the FPGA player,
// replace any programmed timestamp on every transmission, including replay.
inline void stamp_request(std::uint8_t *data, std::size_t size,
                          std::uint64_t timestamp) {
  if (size >= sizeof(RPCHeader))
    std::memcpy(data + offsetof(RPCHeader, ptp_timestamp), &timestamp,
                sizeof(timestamp));
}

// ILA bits [584:521] span words 16-18. Word 16's low nine bits are controls.
constexpr std::array<std::uint32_t, 3>
ila_timestamp_words(std::uint64_t timestamp) {
  return {std::uint32_t(timestamp << ila_timestamp_bit_offset),
          std::uint32_t(timestamp >> (32 - ila_timestamp_bit_offset)),
          std::uint32_t(timestamp >> (64 - ila_timestamp_bit_offset))};
}

} // namespace cudaq::realtime::testing::hsb
