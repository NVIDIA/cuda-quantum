/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_playback_provider.cpp
/// @brief Check provider input boundaries and autonomous UDP playback.
///
/// Invalid HSB settings fail before device access. UDP uses a local socket;
/// no test here requires an FPGA, GPU or application dispatcher.
#include "playback_common.h"
#include <cstddef>
#include <cstdint>
#include <arpa/inet.h>
#include <cerrno>
#include <gtest/gtest.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace cudaq::realtime::validation;
namespace {
class PlaybackProviderTest : public ::testing::TestWithParam<bool> {
protected:
  Plugin plugin; ///< Owns the loaded library and any partial create handle.
  std::vector<std::vector<std::uint8_t>> input =
      requests(1, 8);                               ///< One valid request.
  cudaq_playback_provider_create_params_t params{}; ///< Baseline for each test.
  bool hsb = false; ///< Select HSB input checks; never start its hardware.

  void SetUp() override {
    hsb = GetParam();
    const char *library = UDP_PLAYBACK_PROVIDER_PATH;
#ifdef HSB_PLAYBACK_PROVIDER_PATH
    if (hsb)
      library = HSB_PLAYBACK_PROVIDER_PATH;
#endif
    load_plugin(plugin, library);
    params = {hsb ? "qp=1 rkey=1 buffer_addr=0 peer_ip=192.168.0.2 "
                    "slot_size=512 num_slots=128"
                  : "port=5001 slot_size=96",
              input[0].data(),
              input[0].size(),
              1,
              120000,
              1000000000,
              20000000};
  }

  // A rejected create can own a partial handle; release it before the next
  // case.
  void reject(const cudaq_playback_provider_create_params_t *settings,
              int argument_count = 0, const char *const *arguments = nullptr,
              std::uint64_t timeout_ns = 1000000000) {
    EXPECT_EQ(plugin.api->create(argument_count, arguments, settings,
                                 timeout_ns, &plugin.handle),
              CUDAQ_PLAYBACK_PROVIDER_ERROR);
    EXPECT_NE(*plugin.api->last_error(plugin.handle), '\0');
    ASSERT_EQ(plugin.api->destroy(plugin.handle, 1000000000),
              CUDAQ_PLAYBACK_PROVIDER_OK);
    plugin.handle = nullptr;
  }
};

TEST_P(PlaybackProviderTest, RejectsInvalidBuffersAndDurations) {
  reject(nullptr);
  auto invalid = params;
  invalid.requests = nullptr;
  reject(&invalid);
  // Bounds checks must precede reads from the deliberately small input buffer.
  for (std::size_t size : {std::size_t(0), std::size_t(23), SIZE_MAX}) {
    invalid = params;
    invalid.request_size_bytes = size;
    invalid.request_count = 2;
    reject(&invalid);
  }
  for (std::size_t count : {std::size_t(0), SIZE_MAX}) {
    invalid = params;
    invalid.request_count = count;
    reject(&invalid);
  }
  for (std::uint64_t duration : {std::uint64_t(0), UINT64_MAX}) {
    reject(&params, 0, nullptr, duration);
    invalid = params;
    invalid.run_timeout_ns = duration;
    reject(&invalid);
    invalid = params;
    invalid.request_spacing_ns = duration;
    reject(&invalid);
    invalid = params;
    invalid.capture_quiet_ns = duration;
    reject(&invalid);
  }
  invalid = params;
  invalid.capture_quiet_ns = params.run_timeout_ns;
  reject(&invalid);
}

INSTANTIATE_TEST_SUITE_P(Udp, PlaybackProviderTest, ::testing::Values(false));
#ifdef HSB_PLAYBACK_PROVIDER_PATH
INSTANTIATE_TEST_SUITE_P(Hsb, PlaybackProviderTest, ::testing::Values(true));
#endif

struct Socket {
  int descriptor =
      ::socket(AF_INET, SOCK_DGRAM, 0); ///< Owned local peer socket.
  ~Socket() {
    if (descriptor >= 0)
      ::close(descriptor);
  }
};

TEST(UdpPlayback, ProgressesWithoutStatusPolling) {
  Socket server;
  ASSERT_GE(server.descriptor, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ASSERT_EQ(::bind(server.descriptor, reinterpret_cast<sockaddr *>(&address),
                   sizeof(address)),
            0);
  socklen_t address_size = sizeof(address);
  ASSERT_EQ(::getsockname(server.descriptor,
                          reinterpret_cast<sockaddr *>(&address),
                          &address_size),
            0);
  timeval receive_timeout{
      2, 0}; // Bound reads if playback incorrectly needs polling.
  ASSERT_EQ(::setsockopt(server.descriptor, SOL_SOCKET, SO_RCVTIMEO,
                         &receive_timeout, sizeof(receive_timeout)),
            0);
  Plugin plugin;
  load_plugin(plugin, UDP_PLAYBACK_PROVIDER_PATH);
  auto input = requests(8, 8);
  auto packed = pack_requests(input);
  const auto endpoint = "port=" + std::to_string(ntohs(address.sin_port)) +
                        " slot_size=96 extra=ignored";
  cudaq_playback_provider_create_params_t params{
      endpoint.c_str(), packed.data(),     input[0].size(), input.size(),
      120000,           plugin.timeout_ns, 20000000};
  ASSERT_EQ(plugin.api->create(0, nullptr, &params, plugin.timeout_ns,
                               &plugin.handle),
            CUDAQ_PLAYBACK_PROVIDER_OK);
  // An invalid cleanup budget must leave the prepared handle usable.
  EXPECT_EQ(plugin.api->destroy(plugin.handle, UINT64_MAX),
            CUDAQ_PLAYBACK_PROVIDER_ERROR);
  std::uint8_t buffer[256];
  EXPECT_LT(::recv(server.descriptor, buffer, sizeof(buffer), MSG_DONTWAIT), 0);
  EXPECT_TRUE(errno == EAGAIN ||
              errno == EWOULDBLOCK); // No traffic before start.
  // Stress timer coalescing in the provider worker, without changing the test
  // thread's waits. Socket arrival and timerfd expiration must still wake it.
  const auto previous_slack = ::prctl(PR_GET_TIMERSLACK);
  ASSERT_GT(previous_slack, 0);
  ASSERT_EQ(::prctl(PR_SET_TIMERSLACK, 1000000000UL), 0);
  const auto start_status = plugin.api->start(plugin.handle);
  ASSERT_EQ(::prctl(PR_SET_TIMERSLACK, static_cast<unsigned long>(previous_slack)),
            0);
  ASSERT_EQ(start_status, CUDAQ_PLAYBACK_PROVIDER_OK);
  cudaq_playback_provider_results_t early{};
  EXPECT_EQ(plugin.api->get_results(plugin.handle, &early),
            CUDAQ_PLAYBACK_PROVIDER_ERROR);
  EXPECT_EQ(early.replies, nullptr);
  EXPECT_EQ(early.reply_count, 0);
  // Act only as a wire peer: no status calls while the plugin sends and
  // captures.
  for (const auto &request : input) {
    sockaddr_in peer{};
    socklen_t peer_size = sizeof(peer);
    auto length = ::recvfrom(server.descriptor, buffer, sizeof(buffer), 0,
                             reinterpret_cast<sockaddr *>(&peer), &peer_size);
    ASSERT_EQ(length, 96);
    EXPECT_TRUE(std::equal(request.begin(), request.end(), buffer));
    ASSERT_EQ(::sendto(server.descriptor, buffer, length, 0,
                       reinterpret_cast<sockaddr *>(&peer), peer_size),
              length);
  }
  // Let capture become quiet without polling. The first query finds frozen
  // results.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_EQ(plugin.api->check_complete(plugin.handle),
            CUDAQ_PLAYBACK_PROVIDER_COMPLETE);
  EXPECT_NO_THROW(
      collect_batch(plugin, expected_replies(input, true),
                    [](const cudaq_playback_provider_reply_t &) {}));
  ASSERT_EQ(plugin.api->destroy(plugin.handle, plugin.timeout_ns),
            CUDAQ_PLAYBACK_PROVIDER_OK);
  plugin.handle = nullptr;

  // Cancellation must wake a long socket/timer wait, even with no reply and
  // the next send ten seconds away. destroy's one-second budget bounds the test.
  params.request_spacing_ns = 10000000000;
  params.run_timeout_ns = 30000000000;
  ASSERT_EQ(plugin.api->create(0, nullptr, &params, plugin.timeout_ns,
                               &plugin.handle),
            CUDAQ_PLAYBACK_PROVIDER_OK);
  ASSERT_EQ(plugin.api->start(plugin.handle), CUDAQ_PLAYBACK_PROVIDER_OK);
  ASSERT_EQ(::recv(server.descriptor, buffer, sizeof(buffer), 0), 96);
  ASSERT_EQ(plugin.api->destroy(plugin.handle, 1000000000),
            CUDAQ_PLAYBACK_PROVIDER_OK);
  plugin.handle = nullptr;
}
} // namespace
