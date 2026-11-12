/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file playback_provider_support.h
/// @brief Shared implementation helpers for the reference playback providers.
///
/// These C++ utilities handle deadlines, capture storage and worker completion.
/// They are not part of the public C ABI; another provider can implement the
/// same lifecycle using its own SDK, hardware state machine or worker model.
#pragma once

#include "cudaq/realtime/validation/playback_provider_interface.h"
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <future>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::realtime::validation {
using Clock = std::chrono::steady_clock;
// Integer nanoseconds are used for control deadlines and pacing. Clock and
// wait-library types stay inside these implementation helpers.
inline std::uint64_t elapsed_ns(Clock::time_point started_at) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                           started_at)
          .count());
}
// Internal waits use signed nanoseconds. Zero is allowed here for a
// nonblocking cleanup/status probe; create validates positive durations itself.
inline bool valid_duration_ns(std::uint64_t duration_ns) {
  return duration_ns <= static_cast<std::uint64_t>(INT64_MAX);
}

// Created may own partial setup. Prepared can start once. Finished exposes
// frozen results; Faulted still owns resources that destroy must release.
enum class State { Created, Prepared, Running, Finished, Faulted };
struct Record {
  std::vector<std::uint8_t> bytes; ///< Owned packet at its actual captured length.
  std::uint64_t rtt_ticks = CUDAQ_PLAYBACK_PROVIDER_RTT_UNAVAILABLE; ///< Optional RTT.
};
inline std::uint64_t parse_unsigned_integer(const std::string &text,
                                       std::uint64_t maximum = UINT32_MAX) {
  std::size_t parsed_length = 0;
  if (text.empty() || text[0] == '-')
    throw std::invalid_argument("Invalid unsigned integer: " + text);
  // Base 0 accepts the hexadecimal addresses advertised by RoCE bridges.
  // Require full consumption so a value such as "128bytes" cannot pass.
  auto value = std::stoull(text, &parsed_length, 0);
  if (parsed_length != text.size() || value > maximum)
    throw std::invalid_argument("Invalid unsigned integer: " + text);
  return value;
}
// The reference bridges publish whitespace-separated key=value fields. This
// parser is a convention of these providers, not a requirement of the C ABI.
inline std::map<std::string, std::string> endpoint_fields(const char *text) {
  std::map<std::string, std::string> fields;
  std::istringstream input(text);
  for (std::string word; input >> word;) {
    auto equals_position = word.find('=');
    // emplace reports duplicate keys instead of silently choosing one value.
    // Required keys and transport-specific bounds are checked by each provider.
    if (equals_position == std::string::npos ||
        !fields
             .emplace(word.substr(0, equals_position),
                      word.substr(equals_position + 1))
             .second)
      throw std::invalid_argument("Invalid/duplicate endpoint field: " + word);
  }
  return fields;
}
inline std::pair<std::string, std::string>
parse_provider_option(const char *argument) {
  std::string text(argument);
  auto equals_position = text.find('=');
  if (equals_position == std::string::npos)
    throw std::invalid_argument("Expected --option=value: " + text);
  return {text.substr(0, equals_position), text.substr(equals_position + 1)};
}
inline std::uint64_t little_endian(const std::uint8_t *bytes, std::size_t size) {
  std::uint64_t value = 0;
  for (std::size_t byte_index = 0; byte_index < size; ++byte_index)
    value |= std::uint64_t(bytes[byte_index]) << (8 * byte_index);
  return value;
}
// Providers keep packet storage alive until destroy; build the public array
// only after capture is frozen. No per-packet retrieval or release protocol is
// needed.
struct ResultStorage {
  std::vector<cudaq_playback_provider_reply_t> replies; ///< Views into Record bytes.
  std::uint64_t tick_period_numerator = 0; ///< Seconds-per-tick ratio numerator.
  std::uint64_t tick_period_denominator = 0; ///< Ratio denominator; zero without RTT.
  cudaq_playback_provider_results_t view() const {
    return {replies.data(), replies.size(), tick_period_numerator,
            tick_period_denominator};
  }
};
// Keep the referenced records alive and unchanged until destroy. Only the
// descriptors are copied here, so publishing views must follow final capture.
inline ResultStorage result_views(const std::deque<Record> &records,
                                  std::uint64_t tick_period_numerator,
                                  std::uint64_t tick_period_denominator) {
  ResultStorage results;
  results.replies.reserve(records.size());
  bool measured_rtt = false;
  for (const auto &record : records) {
    results.replies.push_back(
        {record.bytes.data(), record.bytes.size(), record.rtt_ticks});
    measured_rtt |= record.rtt_ticks != CUDAQ_PLAYBACK_PROVIDER_RTT_UNAVAILABLE;
  }
  // Leave both clock fields zero for an entirely untimed capture. A mixed
  // capture still publishes one common scale for its available measurements.
  if (measured_rtt) {
    if (!tick_period_numerator || !tick_period_denominator)
      throw std::runtime_error("Measured RTT needs a positive tick period");
    results.tick_period_numerator = tick_period_numerator;
    results.tick_period_denominator = tick_period_denominator;
  }
  return results;
}

struct SetupTimeout : std::runtime_error {
  SetupTimeout() : std::runtime_error("Setup deadline expired") {}
};
// Share one setup budget across parsing, input copying and device operations.
class SetupDeadline {
  Clock::time_point started_at = Clock::now(); ///< Beginning of create.
  std::uint64_t timeout_ns; ///< Total setup allowance, not a per-operation timeout.

public:
  explicit SetupDeadline(std::uint64_t timeout_ns) : timeout_ns(timeout_ns) {}
  void check() const {
    if (elapsed_ns(started_at) >= timeout_ns)
      throw SetupTimeout();
  }
  std::uint64_t remaining_ns() const {
    const auto elapsed = elapsed_ns(started_at);
    // Check before unsigned subtraction; an expired budget must not wrap to
    // a very large wait passed into the SDK worker join.
    if (elapsed >= timeout_ns)
      throw SetupTimeout();
    return timeout_ns - elapsed;
  }
};
inline bool valid_arguments(int argument_count, const char *const *arguments) {
  if (argument_count < 0 || (argument_count && !arguments))
    return false;
  for (int argument_index = 0; argument_index < argument_count;
       ++argument_index)
    if (!arguments[argument_index])
      return false;
  return true;
}
struct RunTimeout : std::runtime_error {
  RunTimeout()
      : std::runtime_error(
            "Run deadline expired before complete capture/readout") {}
};

// Status may time out while the worker is still inside device I/O. Publish
// diagnostic progress separately so status never reads live packet storage.
struct RunProgress {
  std::atomic<std::size_t> captured{0}; ///< Observed packets, not verified replies.
  std::atomic<const char *> phase{"starting playback"}; ///< Static text only.

  std::string describe(std::size_t expected_count) const {
    return "captured " + std::to_string(captured.load()) +
           " packets (expected " + std::to_string(expected_count) +
           "); phase: " + phase.load() + "; packet contents not verified";
  }
};

// Common finite-batch observation policy. This is an implementation helper,
// not another public API: providers may implement it in their device/SDK code.
class BatchCompletion {
  // Minimum captured count and last observed count, including invalid packets.
  std::size_t expected_count, previous_count = 0;
  // Whole-run budget and required inactivity interval, both in nanoseconds.
  std::uint64_t timeout_ns, quiet_ns;
  // Run origin and most recent count change; each new packet resets quiet time.
  Clock::time_point started_at, last_activity;

public:
  BatchCompletion(std::size_t count, std::uint64_t timeout_ns, std::uint64_t capture_quiet_ns,
                  Clock::time_point started_at)
      : expected_count(count), timeout_ns(timeout_ns),
        quiet_ns(capture_quiet_ns), started_at(started_at),
        last_activity(started_at) {}
  void check_deadline() const {
    if (elapsed_ns(started_at) >= timeout_ns)
      throw RunTimeout();
  }
  // Capture readiness alone proves neither correct replies nor completed TX;
  // providers also account for their own transmission completion requirements.
  bool ready(std::size_t captured_count) {
    check_deadline();
    if (captured_count < previous_count)
      throw std::runtime_error("Capture count moved backwards");
    // Even extra or invalid packets count as activity. Stopping as soon as
    // the expected count arrives could hide a delayed duplicate.
    if (captured_count != previous_count) {
      previous_count = captured_count;
      last_activity = Clock::now();
    }
    // Silence alone is insufficient: missing replies keep the run open until
    // its deadline. Correctness is checked later against the frozen packets.
    return captured_count >= expected_count &&
           elapsed_ns(last_activity) >= quiet_ns;
  }
  // Maximum wait before another completion check. A socket provider can wake
  // sooner on packet arrival. Quiet time matters only after transmission ends.
  std::uint64_t wait_ns(bool transmission_finished) const {
    const auto elapsed = elapsed_ns(started_at);
    std::uint64_t remaining = elapsed >= timeout_ns ? 0 : timeout_ns - elapsed;
    if (transmission_finished && previous_count >= expected_count) {
      const auto quiet_elapsed = elapsed_ns(last_activity);
      remaining = std::min(remaining, quiet_elapsed >= quiet_ns
                                          ? 0
                                          : quiet_ns - quiet_elapsed);
    }
    return remaining;
  }
};
// Check shared buffer geometry and durations before any provider-specific
// parsing. The division guards the later request_size * request_count product.
inline bool
valid_create_params(const cudaq_playback_provider_create_params_t *params) {
  return params && params->requests && params->request_size_bytes >= 24 &&
         params->request_count &&
         params->request_count <= SIZE_MAX / params->request_size_bytes &&
         params->request_spacing_ns &&
         valid_duration_ns(params->request_spacing_ns) &&
         valid_duration_ns(params->run_timeout_ns) &&
         params->capture_quiet_ns &&
         params->capture_quiet_ns < params->run_timeout_ns;
}
// Optional implementation helpers for the two reference providers. The public
// C API does not require vendors to use these C++ worker utilities.
inline bool join_worker(std::future<void> &job, std::uint64_t timeout_ns) {
  if (!job.valid())
    return true;
  const auto started_at = Clock::now();
  std::uint64_t remaining_ns = timeout_ns;
  // Bound each library wait so adding its duration to the clock cannot
  // overflow.
  do {
    const auto wait_ns = std::min<std::uint64_t>(remaining_ns, 1000000000ULL);
    if (job.wait_for(std::chrono::nanoseconds(wait_ns)) ==
        std::future_status::ready) {
      // get() synchronizes the worker's writes and rethrows its exception.
      // It also consumes the future, so subsequent cleanup need not join again.
      job.get();
      return true;
    }
    const auto elapsed = elapsed_ns(started_at);
    if (elapsed >= timeout_ns)
      return false;
    remaining_ns = timeout_ns - elapsed;
  } while (true);
}
// The worker owns I/O and result construction. Serialized API calls own state;
// joining the future publishes the worker's results and propagates exceptions.
template <typename Context>
cudaq_playback_provider_status_t
start_playback(cudaq_playback_provider_t handle) {
  auto *context = static_cast<Context *>(handle);
  if (!context)
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  // Reject repeated starts without touching the live worker or resending data.
  if (context->state != State::Prepared)
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  try {
    context->started_at = Clock::now();
    // Force asynchronous execution: a deferred future would make progress
    // depend on check_complete, contrary to the provider lifecycle.
    context->job =
        std::async(std::launch::async, [context] { context->playback(); });
    context->state = State::Running;
    return CUDAQ_PLAYBACK_PROVIDER_OK;
  } catch (const std::exception &error) {
    context->error = error.what();
    context->state = State::Faulted;
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  }
}
// A zero-budget join only observes progress; callers do not drive playback by
// polling. Once joined successfully, results are immutable and COMPLETE sticks.
template <typename Context>
cudaq_playback_provider_status_t
check_playback_complete(cudaq_playback_provider_t handle) {
  auto *context = static_cast<Context *>(handle);
  if (!context)
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  if (context->state == State::Finished)
    return CUDAQ_PLAYBACK_PROVIDER_COMPLETE;
  if (context->state != State::Running)
    return context->run_timed_out ? CUDAQ_PLAYBACK_PROVIDER_TIMEOUT
                                  : CUDAQ_PLAYBACK_PROVIDER_ERROR;
  try {
    if (!join_worker(context->job, 0)) {
      // The SDK worker may itself be blocked. Status still enforces the run
      // deadline without waiting; destroy retains resources until it can join.
      if (elapsed_ns(context->started_at) >= context->run_timeout_ns)
        throw RunTimeout();
      return CUDAQ_PLAYBACK_PROVIDER_RUNNING;
    }
    // Successful join is the publication boundary. The worker has finished
    // constructing result views, and get_results may now expose their pointers.
    context->state = State::Finished;
    return CUDAQ_PLAYBACK_PROVIDER_COMPLETE;
  } catch (const RunTimeout &error) {
    context->error = std::string(error.what()) + "; " +
                     context->run_progress.describe(context->frames.size());
    context->run_timed_out = true;
    context->state = State::Faulted;
    return CUDAQ_PLAYBACK_PROVIDER_TIMEOUT;
  } catch (const std::exception &error) {
    context->error = error.what();
    context->state = State::Faulted;
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  }
}
// Return local views only after the worker has finished. Clearing the output
// on rejection prevents callers from mistaking an earlier view for new results.
template <typename Context>
cudaq_playback_provider_status_t
get_playback_results(cudaq_playback_provider_t handle,
                     cudaq_playback_provider_results_t *results) {
  if (!results)
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  *results = {};
  auto *context = static_cast<Context *>(handle);
  if (!context || context->state != State::Finished)
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  *results = context->results.view();
  return CUDAQ_PLAYBACK_PROVIDER_OK;
}
} // namespace cudaq::realtime::validation
