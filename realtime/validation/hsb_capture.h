/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hsb_capture.h
/// @brief Decode RFSoC ILA samples into complete packets and measured RTTs.
///
/// Kept separate from SDK I/O so lengths, packet boundaries and timestamp
/// decoding can be tested with hand-encoded samples without FPGA hardware.
#pragma once

#include "playback_provider_support.h"

#include <cstddef>
#include <cstdint>

namespace cudaq::realtime::validation::hsb {
inline void check_capture_extent(std::size_t count, std::uint32_t status,
                                 std::size_t previous) {
  // The 13-bit pointer wraps on full; status bit 1 is the sticky ILA DONE bit.
  if ((status & 2) || count >= 8191 || count < previous)
    throw std::runtime_error(
        "ILA capture full, wrapped or reset: incomplete observation");
}

// RFSoC sensor_if_path.sv profile: data[511:0], valid[512], last[513],
// highest asserted tkeep index[520:514], receive PTP[584:521]. This encoding
// requires nonzero, contiguous low tkeep bits; holes are not representable.
// Other FPGA layouts need their own decoder, not reinterpretation of this one.
class CaptureDecoder {
  Record pending; ///< Accumulated bytes until a valid beat marks packet end.

public:
  void append(const std::vector<std::uint32_t> &sample,
              std::deque<Record> &records) {
    if (sample.size() != 19)
      throw std::runtime_error("Invalid ILA sample width");
    if (!(sample[16] & 1))
      return;
    // Bits 520:514 encode the highest valid byte index, not a byte count.
    // Adding one preserves a one-byte final beat as well as a full 64-byte
    // beat.
    std::size_t valid_bytes = ((sample[16] >> 2) & 127) + 1;
    if (valid_bytes > 64 || pending.bytes.size() + valid_bytes > 65536)
      throw std::runtime_error("Invalid or oversized ILA frame");
    for (std::size_t byte_index = 0; byte_index < valid_bytes; ++byte_index)
      pending.bytes.push_back(
          (sample[byte_index / 4] >> (8 * (byte_index % 4))) & 255);
    // A packet may span beats. Keep accumulating until tlast; only that final
    // beat supplies the receive timestamp for the assembled packet.
    if (!(sample[16] & 2))
      return;
    // Short packets still reach reply verification. Only a complete header
    // can carry the transmit stamp needed for RTT calculation.
    if (pending.bytes.size() >= 24) {
      auto transmit_stamp = little_endian(pending.bytes.data() + 16, 8);
      // Receive PTP occupies bits 584:521: 23 bits from word 16, all of
      // word 17 and nine bits from word 18, combined least-significant first.
      std::uint64_t receive_stamp = std::uint64_t(sample[16] >> 9) |
                                    (std::uint64_t(sample[17]) << 23) |
                                    (std::uint64_t(sample[18] & 511) << 55);
      auto transmit_nsec = std::uint32_t(transmit_stamp);
      auto receive_nsec = std::uint32_t(receive_stamp);
      // The captured PTP clock encodes seconds and nanoseconds. Its tick unit
      // is 1 ns; do not divide by the nominal oscillator resolution.
      // Signed subtraction tolerates a one-second rollover, not a PTP reset.
      std::int64_t elapsed = (std::int64_t(receive_stamp >> 32) -
                              std::int64_t(transmit_stamp >> 32)) *
                                 1000000000LL +
                             std::int64_t(receive_nsec) - transmit_nsec;
      // HSB uses an all-zero timestamp when timing is absent. Preserve the
      // packet but leave RTT_UNAVAILABLE if either stamp is missing or invalid.
      // Equal nonzero timestamps still represent a valid zero-duration RTT.
      if (transmit_stamp && receive_stamp && transmit_nsec < 1000000000 &&
          receive_nsec < 1000000000 && elapsed >= 0) {
        pending.rtt_ticks = elapsed;
      }
    }
    if (records.size() >= 8192)
      throw std::runtime_error("HSB observation queue overflow");
    records.push_back(std::move(pending));
    // Reset both bytes and RTT; an untimed next packet must not inherit the
    // previous packet's measurement.
    pending = {};
  }
  // Call at the frozen capture boundary: leftover bytes indicate lost packet
  // data, not a complete short packet that can be handed to the verifier.
  void finish() const {
    if (!pending.bytes.empty())
      throw std::runtime_error("Incomplete HSB frame at observation boundary");
  }
};
} // namespace cudaq::realtime::validation::hsb
