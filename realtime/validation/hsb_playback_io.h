/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hsb_playback_io.h
/// @brief Shared HSB FPGA register, BRAM and ILA control-plane operations.
///
/// Used by direct playback and the HSB playback provider. These helpers encode
/// the RFSoC memory layout; callers own the device session, capture policy and
/// request/reply verification. All device access goes through the HSB SDK.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <hololink/core/hololink.hpp>
#include <hololink/core/timeout.hpp>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

namespace cudaq::realtime::validation::hsb {
// Player control and request storage: 16 banks form one 64-byte playback beat.
constexpr std::uint32_t PLAYER_ADDR = 0x5000'0000;
constexpr std::uint32_t RAM_ADDR = 0x5010'0000;
constexpr std::uint32_t PLAYER_TIMER_OFFSET = 0x0008;
constexpr std::uint32_t PLAYER_WINDOW_SIZE_OFFSET = 0x000C;
constexpr std::uint32_t PLAYER_WINDOW_NUMBER_OFFSET = 0x0010;
constexpr std::uint32_t PLAYER_ENABLE_OFFSET = 0x0004;
constexpr std::uint32_t RAM_NUM = 16;
constexpr std::uint32_t RAM_DEPTH = 512;

constexpr std::uint32_t PLAYER_ENABLE_SINGLEPASS = 0x0000'000D;
// ram_ena | ptp_bram_ena: continuously replay the programmed BRAM payload.
constexpr std::uint32_t PLAYER_ENABLE_LOOP = 0x0000'0009;
constexpr std::uint32_t PLAYER_DISABLE = 0x0000'0000;

constexpr std::uint32_t SIF_TX_THRESHOLD_ADDR = 0x0120'0000;
constexpr std::uint32_t SIF_TX_THRESHOLD_IMMEDIATE = 0x0000'0005;

constexpr std::uint32_t METADATA_PACKET_ADDR = 0x102C;

// The player timer uses 322 ticks per microsecond for this RFSoC profile.
constexpr std::uint32_t DEFAULT_TIMER_SPACING_US = 120;
constexpr std::uint32_t RF_SOC_TIMER_SCALE = 322;

// Capture stores data plus valid/last/byte-count/PTP metadata across RAM banks.
constexpr std::uint32_t ILA_BASE_ADDR = 0x4000'0000;
constexpr std::uint32_t ILA_CTRL_OFFSET = 0x0000;
constexpr std::uint32_t ILA_SAMPLE_ADDR_OFFSET = 0x0084;
constexpr std::uint32_t ILA_W_DATA = 585;
constexpr std::uint32_t ILA_NUM_RAM = (ILA_W_DATA + 31) / 32; // 19
constexpr std::uint32_t ILA_W_ADDR = 13;                      // log2(8192)
constexpr std::uint32_t ILA_W_RAM = 5;                        // ceil(log2(19))

constexpr std::uint32_t ILA_CTRL_ENABLE = 0x0000'0001;
constexpr std::uint32_t ILA_CTRL_RESET = 0x0000'0002;
constexpr std::uint32_t ILA_CTRL_DISABLE = 0x0000'0000;

constexpr std::uint32_t ILA_TVALID_BIT = 512;

constexpr std::uint32_t ROCEV2_UDP_PORT = 4791;

inline std::uint32_t bram_w_sample_addr() {
  std::uint32_t address_width = 0;
  while ((1u << address_width) < RAM_DEPTH)
    ++address_width;
  return address_width;
}

inline std::uint32_t load_le_u32(const std::uint8_t *bytes) {
  return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) |
         (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
}

/// Pad message to 64-byte aligned window and write to BRAM.
inline void write_bram(hololink::Hololink &hololink_device,
                       const std::vector<std::vector<std::uint8_t>> &windows,
                       std::size_t bytes_per_window) {
  if (bytes_per_window % 64 != 0)
    throw std::runtime_error("bytes_per_window must be a multiple of 64");

  std::size_t cycles = bytes_per_window / 64;
  if (cycles == 0)
    throw std::runtime_error("bytes_per_window is too small");

  if (windows.size() * cycles > RAM_DEPTH) {
    std::ostringstream message;
    message << "Requested " << windows.size() << " windows with " << cycles
            << " cycles each exceeds RAM depth " << RAM_DEPTH;
    throw std::runtime_error(message.str());
  }

  const std::uint32_t sample_address_width = bram_w_sample_addr();

  // Bound control-plane write batches; flush any remainder after the loops
  // so the final, partially filled command batch is not left unsent.
  constexpr std::size_t kBatchWrites = 180;
  hololink::Hololink::WriteData write_data;

  for (std::size_t window_index = 0; window_index < windows.size();
       ++window_index) {
    const auto &window = windows[window_index];
    for (std::size_t cycle_index = 0; cycle_index < cycles; ++cycle_index) {
      for (std::size_t bank_index = 0; bank_index < RAM_NUM; ++bank_index) {
        std::size_t word_index = cycle_index * RAM_NUM + bank_index;
        std::size_t byte_offset = word_index * sizeof(std::uint32_t);
        std::uint32_t value = 0;
        if (byte_offset + sizeof(std::uint32_t) <= window.size())
          value = load_le_u32(window.data() + byte_offset);

        // Bank selects a 32-bit lane; the sample address advances by one word
        // per playback beat. Packet windows may contain several such beats.
        auto ram_addr = static_cast<std::uint32_t>(
            bank_index << (sample_address_width + 2));
        auto sample_addr = static_cast<std::uint32_t>(
            (cycle_index + (window_index * cycles)) * 0x4);
        std::uint32_t address = RAM_ADDR + ram_addr + sample_addr;

        write_data.queue_write_uint32(address, value);
        if (write_data.size() >= kBatchWrites) {
          if (!hololink_device.write_uint32(write_data))
            throw std::runtime_error("Failed to write BRAM batch");
          write_data = hololink::Hololink::WriteData();
        }
      }
    }
  }

  if (write_data.size() > 0) {
    if (!hololink_device.write_uint32(write_data))
      throw std::runtime_error("Failed to write BRAM batch");
  }
}

/// Read back BRAM and verify contents.
inline bool verify_bram(hololink::Hololink &hololink_device,
                        const std::vector<std::vector<std::uint8_t>> &windows,
                        std::size_t bytes_per_window) {
  const std::size_t cycles = bytes_per_window / 64;
  const auto total_cycles = static_cast<std::uint32_t>(windows.size() * cycles);
  const std::uint32_t sample_address_width = bram_w_sample_addr();

  bool all_ok = true;
  std::size_t mismatches = 0;

  // Read each lane contiguously, then map its samples back to packet windows.
  // This mirrors hardware bank addressing while comparing caller-owned bytes.
  for (std::uint32_t bank_index = 0; bank_index < RAM_NUM; ++bank_index) {
    std::uint32_t bank_base =
        RAM_ADDR + (bank_index << (sample_address_width + 2));
    auto [ok, readback] = hololink_device.read_uint32(
        bank_base, total_cycles, hololink::Timeout::default_timeout());
    if (!ok) {
      std::cerr << "BRAM readback: failed to read bank " << bank_index << "\n";
      return false;
    }

    for (std::size_t window_index = 0; window_index < windows.size();
         ++window_index) {
      const auto &window = windows[window_index];
      for (std::size_t cycle_index = 0; cycle_index < cycles; ++cycle_index) {
        std::size_t word_index = cycle_index * RAM_NUM + bank_index;
        std::size_t byte_offset = word_index * sizeof(std::uint32_t);
        std::uint32_t expected = 0;
        if (byte_offset + sizeof(std::uint32_t) <= window.size())
          expected = load_le_u32(window.data() + byte_offset);

        std::size_t sample_index = window_index * cycles + cycle_index;
        std::uint32_t actual = readback[sample_index];

        if (actual != expected) {
          if (mismatches < 10) {
            std::cerr << "  BRAM mismatch: bank=" << bank_index
                      << " sample=" << sample_index << " expected=0x"
                      << std::hex << expected << " got=0x" << actual << std::dec
                      << "\n";
          }
          all_ok = false;
          ++mismatches;
        }
      }
    }
  }

  if (mismatches > 10)
    std::cerr << "  ... and " << (mismatches - 10) << " more mismatches\n";

  return all_ok;
}

// ============================================================================
// ILA functions
// ============================================================================

inline void ila_reset(hololink::Hololink &hololink_device) {
  if (!hololink_device.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET,
                                    ILA_CTRL_RESET))
    throw std::runtime_error("ILA reset write failed");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  if (!hololink_device.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET,
                                    ILA_CTRL_DISABLE))
    throw std::runtime_error("ILA disable-after-reset write failed");
}

inline void ila_enable(hololink::Hololink &hololink_device) {
  if (!hololink_device.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET,
                                    ILA_CTRL_ENABLE))
    throw std::runtime_error("ILA enable write failed");
}

inline void ila_disable(hololink::Hololink &hololink_device) {
  if (!hololink_device.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET,
                                    ILA_CTRL_DISABLE))
    throw std::runtime_error("ILA disable write failed");
}

inline std::uint32_t ila_sample_count(hololink::Hololink &hololink_device) {
  return hololink_device.read_uint32(ILA_BASE_ADDR + ILA_SAMPLE_ADDR_OFFSET);
}

/// Read captured ILA samples from RAM banks. Returns
/// vector of samples; each sample is ILA_NUM_RAM uint32 words (LSW-first).
std::vector<std::vector<std::uint32_t>>
ila_dump(hololink::Hololink &hololink_device, std::uint32_t num_samples) {
  constexpr std::uint32_t ctrl_switch = 1u << (ILA_W_ADDR + 2 + ILA_W_RAM);
  // Max entries per block read: (1472 - 6 byte header) / 8 bytes per entry.
  // Use 128 for comfortable margin on both request and reply packets.
  constexpr std::uint32_t kChunkSize = 128;
  auto timeout = hololink::Timeout::default_timeout();

  std::vector<std::vector<std::uint32_t>> bank_data(ILA_NUM_RAM);
  for (std::uint32_t bank_index = 0; bank_index < ILA_NUM_RAM; ++bank_index) {
    std::uint32_t bank_base =
        ILA_BASE_ADDR + ctrl_switch + (bank_index << (ILA_W_ADDR + 2));
    // Read only the frozen sample count supplied by the caller; unused RAM
    // can contain data from an earlier capture and must not become new replies.
    bank_data[bank_index].reserve(num_samples);
    for (std::uint32_t sample_offset = 0; sample_offset < num_samples;
         sample_offset += kChunkSize) {
      std::uint32_t chunk_size =
          std::min(kChunkSize, num_samples - sample_offset);
      auto [ok, data] = hololink_device.read_uint32(
          bank_base + sample_offset * 4, chunk_size, timeout);
      if (!ok)
        throw std::runtime_error("Failed to read ILA bank " +
                                 std::to_string(bank_index));
      bank_data[bank_index].insert(bank_data[bank_index].end(), data.begin(),
                                   data.end());
    }
  }

  // Device reads are bank-major; callers decode one sample across all banks.
  std::vector<std::vector<std::uint32_t>> samples(
      num_samples, std::vector<std::uint32_t>(ILA_NUM_RAM));
  for (std::uint32_t sample_index = 0; sample_index < num_samples;
       ++sample_index)
    for (std::uint32_t bank_index = 0; bank_index < ILA_NUM_RAM; ++bank_index)
      samples[sample_index][bank_index] = bank_data[bank_index][sample_index];

  return samples;
}

/// Extract a single bit from a 585-bit ILA sample stored as 19 uint32 words.
inline bool get_bit(const std::vector<std::uint32_t> &sample,
                    uint32_t bit_position) {
  uint32_t word = bit_position / 32;
  uint32_t offset = bit_position % 32;
  return (sample[word] >> offset) & 1;
}

/// Extract the first 64 bytes (512 bits) of payload from an ILA sample.
std::vector<std::uint8_t>
extract_payload_bytes(const std::vector<std::uint32_t> &sample,
                      std::size_t num_bytes) {
  std::vector<std::uint8_t> bytes(num_bytes, 0);
  for (std::size_t byte_index = 0; byte_index < num_bytes && byte_index < 64;
       ++byte_index) {
    uint32_t word = sample[byte_index / 4];
    bytes[byte_index] =
        static_cast<std::uint8_t>((word >> ((byte_index % 4) * 8)) & 0xFF);
  }
  return bytes;
}

/// Extract the 64-bit current_ptp_timestamp from ILA bits [584:521].
/// Returns {sec[31:0], nsec[31:0]} packed as a uint64.
inline std::uint64_t
extract_ila_ptp_timestamp(const std::vector<std::uint32_t> &sample) {
  // Bits 521..584 span words 16 and 17 (and partially 18).
  // bit 521 = word 16, offset 9
  // We need 64 bits starting at bit 521.
  uint64_t raw = 0;
  for (int bit_index = 0; bit_index < 64; ++bit_index) {
    uint32_t bit_position = 521 + bit_index;
    uint32_t word_index = bit_position / 32;
    uint32_t bit_offset = bit_position % 32;
    if ((sample[word_index] >> bit_offset) & 1)
      raw |= (uint64_t(1) << bit_index);
  }
  return raw;
}

// Decoded FPGA PTP time, used to subtract stamps across a second boundary.
struct PtpTimestamp {
  uint32_t sec; ///< Whole seconds from the captured PTP clock.
  uint32_t nsec; ///< Nanosecond portion within that second.
};

inline PtpTimestamp decode_ptp(uint64_t raw) {
  // {sec[31:0], nsec[31:0]} -- sec in upper 32 bits, nsec in lower 32
  return {static_cast<uint32_t>(raw >> 32),
          static_cast<uint32_t>(raw & 0xFFFF'FFFF)};
}

/// Compute signed nanosecond difference (recv - send).
inline int64_t ptp_delta_ns(PtpTimestamp send, PtpTimestamp recv) {
  int64_t d_sec = static_cast<int64_t>(recv.sec) - send.sec;
  int64_t d_nsec = static_cast<int64_t>(recv.nsec) - send.nsec;
  return d_sec * 1'000'000'000LL + d_nsec;
}

} // namespace cudaq::realtime::validation::hsb
