/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file playback_common.h
/// @brief Caller-side loading, workload generation and reply verification.
///
/// Shared by playback_app and the provider contract checker so both apply the
/// same packet checks. Provider implementation helpers live in the included
/// support header; this layer never decides how a transport sends requests.
#pragma once

#include "playback_provider_support.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cudaq::realtime::validation {
inline void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
inline void write_little_endian(std::vector<uint8_t> &bytes, size_t offset,
                                uint64_t value, size_t count) {
  for (size_t byte_index = 0; byte_index < count; ++byte_index)
    bytes[offset + byte_index] = (value >> (8 * byte_index)) & 255;
}
inline uint32_t read_little_endian_u32(const std::vector<uint8_t> &bytes,
                                       size_t offset) {
  uint32_t value = 0;
  for (size_t byte_index = 0; byte_index < 4; ++byte_index)
    value |= uint32_t(bytes[offset + byte_index]) << (8 * byte_index);
  return value;
}
// Keep incomplete capture separate from provider operation failures and bad
// reply data so playback_app can report the appropriate outcome.
struct IncompleteRun : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct ProviderError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// Own the dynamic library and its instance together: callbacks and any active
// workers must remain loaded until destroy succeeds.
struct Plugin {
  void *library = nullptr; ///< dlopen handle; released after provider cleanup.
  const cudaq_playback_provider_interface_t *api = nullptr; ///< Library-owned table.
  cudaq_playback_provider_t handle = nullptr; ///< Owned, possibly partial instance.
  uint64_t timeout_ns = 5000000000ULL; ///< Budget for each lifecycle phase.
  ~Plugin() {
    if (handle) {
      if (api->destroy(handle, timeout_ns) != CUDAQ_PLAYBACK_PROVIDER_OK) {
        std::cerr << "Cleanup failed: destroy did not stop activity; retaining "
                     "handle/library\n";
        // Do not dlclose code that an unfinished worker may still execute.
        return;
      }
    }
    if (library)
      dlclose(library);
  }
  void ok(cudaq_playback_provider_status_t status, const char *operation) {
    if (status == CUDAQ_PLAYBACK_PROVIDER_OK)
      return;
    std::string message =
        std::string(operation) + " returned " + std::to_string(status);
    message += ": ";
    // Read the diagnostic before another operation (including cleanup) can
    // replace it. A missing optional callback uses the caller's default text.
    message += api->last_error ? api->last_error(handle) : "Unknown error";
    throw ProviderError(message);
  }
};
// Resolve the versioned getter before calling any provider operation. Only
// last_error is optional; accepting a partial required table would crash later.
inline void load_plugin(Plugin &plugin, const std::string &library) {
  // Resolve dependencies immediately and keep provider symbols local, so one
  // loaded provider cannot accidentally supply another provider's symbols.
  plugin.library = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!plugin.library)
    throw std::runtime_error(dlerror());
  auto getter = reinterpret_cast<cudaq_get_playback_provider_interface_fn_t>(
      dlsym(plugin.library, CUDAQ_PLAYBACK_PROVIDER_GETTER));
  require(getter != nullptr, "Missing provider getter");
  // Version zero is deliberately unsupported: reject a getter that ignores
  // ABI negotiation before interpreting its table layout.
  require(getter(0) == nullptr, "Unsupported ABI must return NULL");
  plugin.api = getter(CUDAQ_PLAYBACK_PROVIDER_ABI_VERSION);
  require(plugin.api != nullptr,
          "Playback provider does not support ABI version " +
              std::to_string(CUDAQ_PLAYBACK_PROVIDER_ABI_VERSION));
  require(plugin.api->create && plugin.api->start &&
              plugin.api->check_complete && plugin.api->get_results &&
              plugin.api->destroy,
          "Missing required provider interface entries");
}

// Generate deterministic requests with unique IDs. The first two payloads
// exercise zero and byte wraparound; subsequent patterns detect misassociation.
inline std::vector<std::vector<uint8_t>>
requests(uint32_t count, uint32_t payload, uint32_t first_id = 1) {
  // Bound memory before allocating and widen arithmetic before multiplication
  // or ID-range checks, so large CLI values cannot wrap past these guards.
  require(count && count <= 1000000 && payload <= 64 * 1024 * 1024 - 24 &&
              uint64_t(first_id) + count <= uint64_t(UINT32_MAX) + 1 &&
              uint64_t(count) * (payload + 24) <= 64 * 1024 * 1024,
          "Invalid request count/size (batch limit 64 MiB)");
  std::vector<std::vector<uint8_t>> result;
  for (uint32_t request_index = 0; request_index < count; ++request_index) {
    uint32_t request_id = first_id + request_index;
    std::vector<uint8_t> frame(24 + payload);
    // Canonical 24-byte header: magic, function ID, payload length, request
    // ID (four bytes each), then an eight-byte provider-owned timing word.
    write_little_endian(frame, 0, 0x43555152, 4);
    write_little_endian(frame, 4, 0x191b6322, 4); // fnv1a("rpc_increment")
    write_little_endian(frame, 8, payload, 4);
    write_little_endian(frame, 12, request_id, 4);
    // Seed the provider-owned timing word to detect accidental workload use;
    // providers may replace it with a transmit timestamp.
    write_little_endian(frame, 16, 0xFEDCBA9800000000ULL + request_id, 8);
    for (uint32_t byte_index = 0; byte_index < payload; ++byte_index)
      frame[24 + byte_index] = request_index == 0 ? 0
                               : request_index == 1
                                   ? 255
                                   : (request_id + byte_index * 37) & 255;
    result.push_back(std::move(frame));
  }
  return result;
}
// This workload uses one size per batch. Packing it once keeps the provider
// API independent of the caller's container and allocation choices.
inline std::vector<uint8_t>
pack_requests(const std::vector<std::vector<uint8_t>> &input) {
  require(!input.empty() && !input.front().empty(), "Empty request batch");
  std::vector<uint8_t> packed;
  packed.reserve(input.size() * input.front().size());
  for (const auto &request : input) {
    require(request.size() == input.front().size(),
            "Requests must have one size per batch");
    packed.insert(packed.end(), request.begin(), request.end());
  }
  return packed;
}
// Index by request ID so arrival order is irrelevant. Native forward echoes
// the request; RPC dispatch changes the response header and increments payload.
inline std::map<uint32_t, std::vector<uint8_t>>
expected_replies(const std::vector<std::vector<uint8_t>> &input, bool forward) {
  std::map<uint32_t, std::vector<uint8_t>> result;
  for (auto frame : input) {
    if (!forward) {
      // A dispatched response replaces request magic/function ID with
      // response magic/status zero. Incrementing uint8_t also tests 255 -> 0.
      write_little_endian(frame, 0, 0x43555153, 4);
      write_little_endian(frame, 4, 0, 4);
      for (size_t byte_index = 24; byte_index < frame.size(); ++byte_index)
        ++frame[byte_index];
    }
    result.emplace(read_little_endian_u32(frame, 12), std::move(frame));
  }
  return result;
}
// Require one reply per expected ID and exact workload bytes. Trailing slot
// padding is allowed, but truncated headers/payloads and duplicates are not.
inline void
verify_reply(const cudaq_playback_provider_reply_t &reply,
             const std::map<uint32_t, std::vector<uint8_t>> &expected,
             std::set<uint32_t> &seen) {
  require(reply.data && reply.size_bytes >= 24 && reply.size_bytes <= 1048576,
          "Truncated/invalid RPC reply");
  std::vector<uint8_t> bytes(reply.data, reply.data + reply.size_bytes);
  uint32_t request_id = read_little_endian_u32(bytes, 12);
  require(expected.count(request_id) && seen.insert(request_id).second,
          "Unexpected, stale or duplicate request ID");
  const auto &wanted = expected.at(request_id);
  // The timing word belongs to the provider. It is not a workload byte and
  // cannot be independently verified by copying it from the received reply.
  require(bytes.size() >= wanted.size() &&
              std::equal(wanted.begin(), wanted.begin() + 16, bytes.begin()) &&
              std::equal(wanted.begin() + 24, wanted.end(), bytes.begin() + 24),
          "RPC header/payload mismatch (or create retained caller memory)");
}
// Missing measurements are valid. A clock ratio is required exactly when at
// least one reply carries a measured RTT; it applies to all measured replies.
inline void verify_timing(const cudaq_playback_provider_results_t &results) {
  bool measured = false;
  for (size_t reply_index = 0; reply_index < results.reply_count; ++reply_index)
    measured |= results.replies[reply_index].rtt_ticks !=
                CUDAQ_PLAYBACK_PROVIDER_RTT_UNAVAILABLE;
  if (measured)
    require(results.tick_period_numerator && results.tick_period_denominator,
            "Measured RTT needs a positive tick period");
  else
    require(!results.tick_period_numerator && !results.tick_period_denominator,
            "Unavailable RTT has nonempty timing metadata");
}
// Status checks never retrieve packets or drive playback. Once the provider
// has frozen and downloaded capture, verify the entire stable result batch.
template <typename Observe>
cudaq_playback_provider_results_t
collect_batch(Plugin &plugin,
              const std::map<uint32_t, std::vector<uint8_t>> &expected,
              Observe observe) {
  using Clock = std::chrono::steady_clock;
  const auto started_at = Clock::now();
  while (true) {
    const auto before = Clock::now();
    // This check detects a slow callback after it returns. The shell runner's
    // external watchdog handles a callback that never returns at all.
    const auto status = plugin.api->check_complete(plugin.handle);
    require(Clock::now() - before < std::chrono::seconds(1),
            "Nonblocking check_complete exceeded watchdog tolerance");
    if (status == CUDAQ_PLAYBACK_PROVIDER_TIMEOUT) {
      // Preserve provider progress before destroy can replace the diagnostic.
      // A timeout does not make partial packet storage available for checking.
      throw IncompleteRun(plugin.api->last_error
                              ? plugin.api->last_error(plugin.handle)
                              : "Run deadline expired; provider supplies no "
                                "timeout details");
    }
    if (elapsed_ns(started_at) >= plugin.timeout_ns)
      throw IncompleteRun(
          "Run deadline expired in playback_app before the provider reported "
          "completion; packet contents not verified");
    if (status == CUDAQ_PLAYBACK_PROVIDER_COMPLETE)
      break;
    // check_complete has its own status vocabulary. OK cannot mean either
    // still running or complete, so accepting it would hide a provider bug.
    if (status != CUDAQ_PLAYBACK_PROVIDER_RUNNING) {
      require(status != CUDAQ_PLAYBACK_PROVIDER_OK,
              "check_complete returned OK instead of run status");
      plugin.ok(status, "check_complete");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  cudaq_playback_provider_results_t results{};
  plugin.ok(plugin.api->get_results(plugin.handle, &results), "get_results");
  // Bound traversal and require array storage before reading any descriptors.
  // A provider may return extra packets; verify_reply will reject their IDs.
  require(results.reply_count <= 2000064 &&
              (!results.reply_count || results.replies),
          "Invalid result array");
  verify_timing(results);
  std::set<uint32_t> seen;
  for (size_t reply_index = 0; reply_index < results.reply_count;
       ++reply_index) {
    verify_reply(results.replies[reply_index], expected, seen);
    // Observers collect timing only from packets that passed the byte/ID
    // checks. A later bad or missing reply still fails the entire batch.
    observe(results.replies[reply_index]);
  }
  require(seen.size() == expected.size(),
          "Completed batch has missing replies");
  return results;
}
} // namespace cudaq::realtime::validation
