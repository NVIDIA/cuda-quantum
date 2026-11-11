/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file playback_provider_contract.cpp
/// @brief Check a dynamically loaded provider against the playback ABI.
///
/// Runs two batches with different IDs, checks lifecycle and input ownership,
/// then verifies replies and immutable result storage. A real provider needs
/// its compatible RTH/peer running.
#include "playback_common.h"
#include "playback_provider_support.h"

using namespace cudaq::realtime::validation;

int main(int argument_count, char **arguments) {
  try {
    Plugin plugin;
    std::string library, endpoint;
    bool forward = false;
    uint64_t request_spacing_ns = 120000;
    std::vector<const char *> provider_args;
    for (int argument_index = 1; argument_index < argument_count;
         ++argument_index) {
      const std::string argument = arguments[argument_index];
      // Everything after -- is provider-owned; preserve spaces and token
      // boundaries so the checker can load third-party implementations too.
      if (argument == "--") {
        for (++argument_index; argument_index < argument_count;
             ++argument_index)
          provider_args.push_back(arguments[argument_index]);
        break;
      }
      if (argument.rfind("--plugin=", 0) == 0)
        library = argument.substr(9);
      else if (argument.rfind("--endpoint=", 0) == 0)
        endpoint = argument.substr(11);
      else if (argument.rfind("--timeout-ns=", 0) == 0)
        plugin.timeout_ns =
            parse_unsigned_integer(argument.substr(13), INT64_MAX);
      else if (argument.rfind("--spacing-ns=", 0) == 0)
        request_spacing_ns =
            parse_unsigned_integer(argument.substr(13), INT64_MAX);
      else if (argument == "--forward")
        forward = true;
      else
        throw std::runtime_error("Unknown argument: " + argument);
    }
    require(!library.empty() && plugin.timeout_ns > 0 && request_spacing_ns > 0,
            "Usage: playback_provider_contract --plugin=LIB [--endpoint=TEXT] "
            "[--forward] [--spacing-ns=N] [--timeout-ns=N] -- [provider "
            "options]");
    load_plugin(plugin, library);
    // These calls need no peer traffic: null cleanup is legal, but an empty
    // batch must fail. A rejected create may still return a partial handle.
    plugin.ok(plugin.api->destroy(nullptr, plugin.timeout_ns), "destroy(NULL)");
    cudaq_playback_provider_create_params_t empty{};
    require(plugin.api->create(int(provider_args.size()), provider_args.data(),
                               &empty, plugin.timeout_ns,
                               &plugin.handle) == CUDAQ_PLAYBACK_PROVIDER_ERROR,
            "Empty batch must be rejected");
    if (plugin.handle) {
      plugin.ok(plugin.api->destroy(plugin.handle, plugin.timeout_ns),
                "destroy partial create");
      plugin.handle = nullptr;
    }

    // Each batch gets a fresh handle. No reusable-handle protocol is required.
    for (unsigned run = 0; run < 2; ++run) {
      // Disjoint IDs make replies retained from the previous handle visible
      // as stale data instead of accidentally matching the next batch.
      auto input = requests(8, 8, 1000 * run + 1);
      auto expected = expected_replies(input, forward);
      auto packed = pack_requests(input);
      std::string borrowed_endpoint = endpoint;
      cudaq_playback_provider_create_params_t params{
          borrowed_endpoint.empty() ? nullptr : borrowed_endpoint.c_str(),
          packed.data(),
          input.front().size(),
          input.size(),
          request_spacing_ns,
          plugin.timeout_ns,
          200000000};
      plugin.ok(plugin.api->create(int(provider_args.size()),
                                   provider_args.data(), &params,
                                   plugin.timeout_ns, &plugin.handle),
                "create");
      require(plugin.handle, "Successful create returned NULL");
      // Pre-fill every result field so an early get_results must actively
      // clear it. A zero-initialized output would not test that obligation.
      cudaq_playback_provider_reply_t sentinel{};
      cudaq_playback_provider_results_t results{&sentinel, 1, 1, 1};
      require(plugin.api->check_complete(plugin.handle) ==
                  CUDAQ_PLAYBACK_PROVIDER_ERROR,
              "check_complete before start must fail without changing state");
      require(plugin.api->get_results(plugin.handle, &results) ==
                      CUDAQ_PLAYBACK_PROVIDER_ERROR &&
                  !results.replies && !results.reply_count &&
                  !results.tick_period_numerator &&
                  !results.tick_period_denominator,
              "get_results before completion must fail with empty results");
      // Keep allocations live while overwriting them to expose illegal borrows
      // without making a faulty provider dereference freed test memory.
      std::fill(packed.begin(), packed.end(), 0xa5);
      std::fill(borrowed_endpoint.begin(), borrowed_endpoint.end(), 'X');
      plugin.ok(plugin.api->start(plugin.handle), "start");
      require(plugin.api->start(plugin.handle) == CUDAQ_PLAYBACK_PROVIDER_ERROR,
              "Repeated start must fail without retransmitting");
      // Verification after the rejected second start also checks that the
      // first run stayed intact and no duplicate transmission was introduced.
      collect_batch(plugin, expected,
                    [](const cudaq_playback_provider_reply_t &) {});
      require(plugin.api->check_complete(plugin.handle) ==
                  CUDAQ_PLAYBACK_PROVIDER_COMPLETE,
              "Completed batch must remain COMPLETE");
      plugin.ok(plugin.api->get_results(plugin.handle, &results),
                "get_results");
      // Snapshot both descriptors and packet bytes: checking pointer equality
      // alone would miss a provider mutating storage after returning COMPLETE.
      const std::vector<cudaq_playback_provider_reply_t> saved_replies(
          results.replies, results.replies + results.reply_count);
      std::vector<std::vector<uint8_t>> saved_bytes;
      for (const auto &reply : saved_replies)
        saved_bytes.emplace_back(reply.data, reply.data + reply.size_bytes);
      cudaq_playback_provider_results_t repeated{};
      plugin.ok(plugin.api->get_results(plugin.handle, &repeated),
                "repeat get_results");
      require(results.replies == repeated.replies &&
                  results.reply_count == repeated.reply_count &&
                  results.tick_period_numerator ==
                      repeated.tick_period_numerator &&
                  results.tick_period_denominator ==
                      repeated.tick_period_denominator,
              "Repeated retrieval must preserve result storage");
      // Compare all bytes here, including the provider-owned timing word:
      // workload verification skips it, but completed storage must be immutable.
      std::set<uint32_t> seen;
      for (size_t reply_index = 0; reply_index < results.reply_count;
           ++reply_index) {
        const auto &reply = results.replies[reply_index];
        const auto &saved = saved_replies[reply_index];
        require(reply.data == saved.data &&
                    reply.size_bytes == saved.size_bytes &&
                    reply.rtt_ticks == saved.rtt_ticks &&
                    std::equal(saved_bytes[reply_index].begin(),
                               saved_bytes[reply_index].end(), reply.data),
                "Result bytes or timing changed after COMPLETE");
        verify_reply(reply, expected, seen);
      }
      plugin.ok(plugin.api->destroy(plugin.handle, plugin.timeout_ns),
                "destroy");
      plugin.handle = nullptr;
      std::cout << "Batch " << run << ": 8/8 verified\n";
    }
    std::cout << "PASS: provider "
                 "ABI/create/start/check_complete/get_results/destroy/"
                 "input-copy/reply "
                 "contracts\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
