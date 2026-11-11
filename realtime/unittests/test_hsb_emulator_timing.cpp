/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "../validation/hsb_emulator_timing.h"
#include "../validation/hsb_capture.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>

namespace {
namespace timing = cudaq::realtime::testing::hsb;

// Use the playback reader's bit-wise extraction rather than the writer's
// shifts.
std::uint64_t extract_timestamp(const std::array<std::uint32_t, 19> &words) {
  std::uint64_t timestamp = 0;
  for (unsigned bit = 0; bit < 64; ++bit) {
    const unsigned position = 521 + bit;
    if ((words[position / 32] >> (position % 32)) & 1)
      timestamp |= std::uint64_t{1} << bit;
  }
  return timestamp;
}

std::uint64_t unpack_timestamp(std::uint64_t timestamp) {
  return (timestamp >> 32) * timing::nanoseconds_per_second +
         std::uint32_t(timestamp);
}

TEST(HsbEmulatorTiming, SecondsBoundaryUsesNormalizedNanoseconds) {
  EXPECT_EQ(timing::pack_timestamp(999'999'999), 999'999'999u);
  EXPECT_EQ(timing::pack_timestamp(1'000'000'000), std::uint64_t{1} << 32);
  EXPECT_EQ(timing::pack_timestamp(1'000'000'001),
            (std::uint64_t{1} << 32) | 1);
}

TEST(HsbEmulatorTiming, TimestampInsertionFollowsPlayerEnableBit) {
  for (const std::uint32_t control : {0x0u, 0x1u, 0x4u, 0x5u}) {
    EXPECT_FALSE(timing::timestamp_insertion_enabled(control));
    EXPECT_TRUE(timing::timestamp_insertion_enabled(control | 0x8u));
  }
  EXPECT_FALSE(timing::timestamp_insertion_enabled(0xfffffff7u));
  EXPECT_TRUE(timing::timestamp_insertion_enabled(0xffffffffu));
}

TEST(HsbEmulatorTiming, IlaRoundTripPreservesControlsAndPayload) {
  constexpr std::uint32_t controls = 0b111; // tvalid, tlast, wr_tcnt=1
  for (const auto timestamp :
       {std::uint64_t{1}, std::uint64_t{1} << 23, std::uint64_t{1} << 55,
        std::uint64_t{0x123456783b9ac9ff},
        std::numeric_limits<std::uint64_t>::max()}) {
    std::array<std::uint32_t, 19> words{};
    std::fill_n(words.begin(), 16, 0xabcdef01);
    const auto packed = timing::ila_timestamp_words(timestamp);
    std::copy(packed.begin(), packed.end(), words.begin() + 16);
    words[16] |= controls;
    EXPECT_EQ(extract_timestamp(words), timestamp);
    EXPECT_EQ(words[16] & 0x1ff, controls);
    EXPECT_TRUE(std::all_of(words.begin(), words.begin() + 16,
                            [](auto word) { return word == 0xabcdef01; }));
  }
}

TEST(HsbEmulatorTiming, RefreshesOnlyTimestampInOutgoingCopy) {
  cudaq::realtime::RPCHeader request{cudaq::realtime::RPC_MAGIC_REQUEST, 123, 8,
                                     42, 0};
  // Exercise a deliberately unaligned frame, as well as the bytes after it.
  std::array<std::uint8_t, 64> cached{};
  std::memcpy(cached.data() + 1, &request, sizeof(request));
  std::fill(cached.begin() + 1 + sizeof(request), cached.end(), 0xa5);
  auto transmitted = cached;
  for (const auto timestamp : {timing::pack_timestamp(2'000'000'001),
                               timing::pack_timestamp(3'000'000'002)}) {
    timing::stamp_request(transmitted.data() + 1, transmitted.size() - 1,
                          timestamp);
    cudaq::realtime::RPCHeader stamped{};
    std::memcpy(&stamped, transmitted.data() + 1, sizeof(stamped));
    EXPECT_EQ(stamped.ptp_timestamp, timestamp);
    stamped.ptp_timestamp = 0;
    EXPECT_EQ(std::memcmp(&stamped, &request, sizeof(request)), 0);
    EXPECT_EQ(transmitted.front(), cached.front());
    EXPECT_TRUE(std::equal(transmitted.begin() + 1 + sizeof(request),
                           transmitted.end(),
                           cached.begin() + 1 + sizeof(request)));
  }
  cudaq::realtime::RPCHeader original{};
  std::memcpy(&original, cached.data() + 1, sizeof(original));
  EXPECT_EQ(original.ptp_timestamp, 0u);
}

TEST(HsbEmulatorTiming, ShortFramesAreNotWritten) {
  std::array<std::uint8_t, sizeof(cudaq::realtime::RPCHeader)> frame{};
  frame.fill(0xa5);
  const auto original = frame;
  for (std::size_t size = 0; size < frame.size(); ++size) {
    timing::stamp_request(frame.data(), size, 1);
    EXPECT_EQ(frame, original);
  }
}

TEST(HsbEmulatorTiming, OneClockForRequestEchoAndCapture) {
  cudaq::realtime::RPCHeader request{};
  const auto send = timing::monotonic_timestamp();
  timing::stamp_request(reinterpret_cast<std::uint8_t *>(&request),
                        sizeof(request), send);
  // The dispatcher echoes this field unchanged. The emulator adds no new field.
  cudaq::realtime::RPCResponse response{};
  response.ptp_timestamp = request.ptp_timestamp;
  const auto receive = timing::monotonic_timestamp();
  EXPECT_NE(send, 0u);
  EXPECT_GE(unpack_timestamp(receive),
            unpack_timestamp(response.ptp_timestamp));
  EXPECT_LT(std::uint32_t(receive), timing::nanoseconds_per_second);
}
} // namespace

namespace {
namespace capture = cudaq::realtime::validation::hsb;
using cudaq::realtime::validation::Record;

TEST(HsbCapture, DecodesBytesAndTimestampRollover) {
  capture::CaptureDecoder decoder;
  std::deque<Record> records;
  // Independent wire sample: TX=(7 s, 999999990 ns), RX=(8 s, 30 ns).
  std::vector<uint32_t> sample(19);
  sample[0] = 0x03020100;
  sample[4] = 999999990;
  sample[5] = 7;
  sample[7] = 0xfffefdfc;
  sample[16] = 0x00003c7f; // valid | last | (31 << 2) | (30 << 9)
  sample[17] = 0x00001000; // RX seconds=8 across ILA words.
  decoder.append(sample, records);
  ASSERT_EQ(records.size(), 1);
  ASSERT_EQ(records[0].bytes.size(), 32);
  EXPECT_EQ(records[0].bytes[0], 0);
  EXPECT_EQ(records[0].bytes[3], 3);
  EXPECT_EQ(records[0].bytes[31], 255);
  EXPECT_EQ(records[0].rtt_ticks, 40);
  EXPECT_NO_THROW(decoder.finish());
}

TEST(HsbCapture, MissingTimestampsPreserveReplyBytes) {
  // TX and RX, when present, both encode (8 s, 30 ns): a valid zero RTT.
  // An absent stamp must leave timing unavailable without discarding the reply.
  for (bool has_transmit : {false, true}) {
    for (bool has_receive : {false, true}) {
      capture::CaptureDecoder decoder;
      std::deque<Record> records;
      std::vector<uint32_t> sample(19);
      sample[0] = 0x03020100;
      sample[4] = has_transmit ? 30 : 0;
      sample[5] = has_transmit ? 8 : 0;
      sample[7] = 0xfffefdfc;
      sample[16] = has_receive ? 0x00003c7f : 0x7f;
      sample[17] = has_receive ? 0x00001000 : 0;
      decoder.append(sample, records);
      ASSERT_EQ(records.size(), 1);
      ASSERT_EQ(records[0].bytes.size(), 32);
      EXPECT_EQ(records[0].bytes[0], 0);
      EXPECT_EQ(records[0].bytes[31], 255);
      EXPECT_EQ(records[0].rtt_ticks,
                has_transmit && has_receive
                    ? 0u
                    : CUDAQ_PLAYBACK_PROVIDER_RTT_UNAVAILABLE);
      EXPECT_NO_THROW(decoder.finish());
    }
  }
}

TEST(HsbCapture, PreservesPacketBoundariesAndRejectsCaptureLoss) {
  capture::CaptureDecoder decoder;
  std::deque<Record> records;
  std::vector<uint32_t> sample(19);
  sample[16] = 0xfd; // A full first beat without a final beat is incomplete.
  decoder.append(sample, records);
  EXPECT_TRUE(records.empty());
  EXPECT_THROW(decoder.finish(), std::runtime_error);
  sample[0] = 0x00ccbbaa;
  sample[16] = 0xb; // Three-byte final beat completes one 67-byte packet.
  decoder.append(sample, records);
  ASSERT_EQ(records.size(), 1);
  ASSERT_EQ(records[0].bytes.size(), 67);
  EXPECT_EQ(records[0].bytes[64], 0xaa);
  EXPECT_EQ(records[0].bytes[66], 0xcc);
  EXPECT_NO_THROW(decoder.finish());
  sample[16] = 2; // Ignore invalid beats, but preserve complete short packets.
  decoder.append(sample, records);
  EXPECT_EQ(records.size(), 1);
  sample[16] = 3;
  decoder.append(sample, records);
  ASSERT_EQ(records.size(), 2);
  EXPECT_EQ(records.back().bytes.size(), 1);
  EXPECT_EQ(records.back().rtt_ticks, CUDAQ_PLAYBACK_PROVIDER_RTT_UNAVAILABLE);
  sample[16] = 0x103; // Byte index 64 exceeds the 512-bit bus.
  EXPECT_THROW(decoder.append(sample, records), std::runtime_error);
  EXPECT_NO_THROW(capture::check_capture_extent(100, 0, 99));
  // Full flag, full pointer and a backwards count each indicate capture loss.
  EXPECT_THROW(capture::check_capture_extent(0, 2, 0), std::runtime_error);
  EXPECT_THROW(capture::check_capture_extent(8191, 0, 0), std::runtime_error);
  EXPECT_THROW(capture::check_capture_extent(5, 0, 6), std::runtime_error);
}
} // namespace
