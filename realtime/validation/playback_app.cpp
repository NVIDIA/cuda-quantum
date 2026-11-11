/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file playback_app.cpp
/// @brief Generate a validation batch, drive a playback provider and report it.
///
/// This app owns the workload and reply checks. A runtime-loaded provider owns
/// transmission, capture and timing. The optional start-file gate lets the
/// runner prepare both ends before traffic begins; no provider SDK is linked.
#include "playback_common.h"
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>

using namespace cudaq::realtime::validation;

int main(int argument_count, char **arguments) {
  try {
    // Own the library and opaque handle for the entire run. On exceptions,
    // Plugin attempts destroy before unloading; failed cleanup retains
    // active resources so provider code cannot be unloaded while still in use.
    std::cout.imbue(std::locale::classic());
    Plugin plugin;
    plugin.timeout_ns = 30000000000ULL;
    // Keep common workload controls here; options after -- belong solely to
    // the provider. start_file is a runner synchronization file, not input data.
    std::string library, endpoint, start_file;
    uint32_t count = 100, payload = 8;
    uint64_t capture_quiet_ns = 1000000000;
    uint64_t request_spacing_ns = 120000;
    bool forward = false, size_selected = false;
    std::vector<const char *> provider_args;
    for (int argument_index = 1; argument_index < argument_count;
         ++argument_index) {
      std::string argument = arguments[argument_index];
      if (argument == "--") {
        // Only create() interprets provider options. Forward them verbatim.
        for (++argument_index; argument_index < argument_count;
             ++argument_index)
          provider_args.push_back(arguments[argument_index]);
        break;
      }
      auto equals_position = argument.find('=');
      auto key = argument.substr(0, equals_position);
      auto value = equals_position == std::string::npos
                       ? ""
                       : argument.substr(equals_position + 1);
      // The same parser serves counts/sizes and wider nanosecond durations.
      // Reject signs, trailing text and overflow before narrowing the value.
      auto number = [&](uint64_t maximum = UINT32_MAX) {
        size_t parsed_length = 0;
        require(!value.empty() && value[0] != '-',
                "Expected unsigned integer: " + argument);
        auto parsed_value = std::stoull(value, &parsed_length);
        require(parsed_length == value.size() && parsed_value <= maximum,
                "Invalid integer: " + argument);
        return uint64_t(parsed_value);
      };
      if (key == "--playback-provider" || key == "--provider" ||
          key == "--plugin")
        library = value;
      else if (key == "--start-file")
        start_file = value;
      else if (key == "--endpoint")
        endpoint = value;
      else if (key == "--num-messages")
        count = number();
      else if (key == "--message-size" || key == "--payload-size") {
        // These are two spellings of one setting. Check the full message size
        // before subtracting the header, avoiding unsigned underflow.
        require(!size_selected, "Specify message size only once");
        size_selected = true;
        const auto size = number();
        require(key == "--payload-size" || size >= 24,
                "--message-size includes the 24-byte RPC header");
        payload = key == "--message-size" ? size - 24 : size;
      } else if (key == "--timeout-ns")
        plugin.timeout_ns = number(INT64_MAX);
      else if (key == "--spacing-ns")
        request_spacing_ns = number(INT64_MAX);
      else if (key == "--quiet-ns")
        capture_quiet_ns = number(INT64_MAX);
      else if (argument == "--forward")
        forward = true;
      else if (argument == "--help") {
        std::cout
            << "Usage: playback_app --playback-provider=LIB [--endpoint=TEXT] "
               "[options] "
               "-- [provider options]\n"
               "  --start-file=PATH  Runner gate: prepare, then wait for this "
               "file\n"
               "  --endpoint=TEXT    RTH endpoint, when required by the "
               "provider\n"
               "  --num-messages=N   Batch size (default 100)\n"
               "  --message-size=N   RPC header + payload bytes, excluding "
               "padding "
               "(default 32)\n"
               "  --payload-size=N   Alternative: payload bytes only (default "
               "8)\n"
               "  --timeout-ns=N     Setup/run/destroy budgets in ns (default "
               "30000000000)\n"
               "  --spacing-ns=N     Request start interval in ns (default "
               "120000)\n"
               "  --quiet-ns=N       Capture quiet period in ns (default "
               "1000000000)\n"
               "  --forward          Expect an unchanged native echo\n"
               "Start the consumer and wait for App Ready before invoking "
               "playback.\n";
        return 0;
      } else
        throw std::runtime_error("Unknown app option: " + argument);
    }
    require(
        !library.empty() && plugin.timeout_ns && request_spacing_ns &&
            capture_quiet_ns && capture_quiet_ns < plugin.timeout_ns,
        "--playback-provider=LIB, positive spacing/timeout and a shorter quiet "
        "period are required");
    // The app defines the workload and expected answers. Plugins receive
    // canonical RPC bytes, including the function ID, rather than generating
    // their own test requests or deciding whether replies are correct.
    auto input = requests(count, payload);
    auto expected = expected_replies(input, forward);
    auto packed = pack_requests(input);
    // 1. Load the exact ABI requested by this app. The versioned getter
    // rejects incompatible tables; no provider-specific dispatch is needed.
    load_plugin(plugin, library);

    // 2. Create a playback instance: parse opaque options, copy
    // requests/endpoint, configure hardware and prepare capture. No requests
    // are sent yet. Even a failed create can return a partial handle; Plugin
    // destroys it during exception unwinding.
    cudaq_playback_provider_create_params_t params{
        endpoint.empty() ? nullptr : endpoint.c_str(),
        packed.data(),
        input.front().size(),
        input.size(),
        request_spacing_ns,
        plugin.timeout_ns,
        capture_quiet_ns};
    plugin.ok(plugin.api->create(int(provider_args.size()),
                                 provider_args.data(), &params,
                                 plugin.timeout_ns, &plugin.handle),
              "create");
    require(plugin.handle, "Successful create returned NULL");

    // 3. Create may complete a handshake needed by the RTH's connect(). The
    // runner starts both sides, then releases this gate only when both are
    // ready. Standalone use requires the caller to ensure RTH readiness.
    std::cout << "CUDAQ_PLAYBACK_READY" << std::endl;
    // Preparation and readiness have separate budgets. Waiting for the RTH
    // must not consume the provider's run budget, which starts in start().
    const auto ready_started_at = Clock::now();
    while (!start_file.empty() && !std::ifstream(start_file).good()) {
      if (elapsed_ns(ready_started_at) >= plugin.timeout_ns)
        throw IncompleteRun("RTH start gate deadline expired");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    plugin.ok(plugin.api->start(plugin.handle), "start");
    // 4. Wait for frozen results, verify every response, and copy timing before
    // destroy releases provider-owned storage. Keep the integer ticks
    // untouched.
    std::vector<std::pair<uint32_t, uint64_t>> samples;
    const auto results = collect_batch(
        plugin, expected, [&](const cudaq_playback_provider_reply_t &reply) {
          // Missing RTT is valid; keep the correct packet in the reply count
          // and omit only its timing sample. A measured value of zero is valid.
          if (reply.rtt_ticks != CUDAQ_PLAYBACK_PROVIDER_RTT_UNAVAILABLE)
            samples.emplace_back(little_endian(reply.data + 12, 4),
                                 reply.rtt_ticks);
        });
    // Convert only for presentation. Raw integer ticks and their exact clock
    // ratio remain in the output so callers can recompute the measurements.
    const long double tick_period_seconds =
        results.tick_period_denominator
            ? static_cast<long double>(results.tick_period_numerator) /
                  results.tick_period_denominator
            : 0;
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
              << "WORKLOAD name="
              << (forward ? "native_forward" : "rpc_increment")
              << " samples=" << count
              << " message_size_bytes=" << input.front().size()
              << " payload_size_bytes=" << payload
              << " request_spacing_ns=" << request_spacing_ns << '\n'
              << "REPLIES PASS: " << expected.size() << '/' << count
              << " packets received correctly\n";
    if (!samples.empty())
      std::cout << "CLOCK tick_period_numerator="
                << results.tick_period_numerator << " tick_period_denominator="
                << results.tick_period_denominator
                << " tick_period_seconds=" << tick_period_seconds << '\n';
    for (const auto &[request_id, ticks] : samples)
      std::cout << "LATENCY request_id=" << request_id << " ticks=" << ticks
                << " ns=" << ticks * tick_period_seconds * 1000000000 << '\n';
    std::cout.flush();
    // Timing availability is separate from reply correctness. Preserve measured
    // samples, but do not report full-run statistics for a partial set.
    if (samples.size() != count) {
      if (samples.empty())
        std::cout << "LATENCY UNAVAILABLE: provider supplied no timing "
                     "measurements; statistics cannot be calculated\n";
      else
        std::cout << "LATENCY PARTIAL: measured " << samples.size() << '/'
                  << count << "; full-run statistics cannot be calculated\n";
    } else {
      // Sort integer ticks so large counters retain their exact ordering. For
      // an even sample count the median is the average of the two middle
      // values.
      std::vector<uint64_t> ticks;
      for (const auto &sample : samples)
        ticks.push_back(sample.second);
      std::sort(ticks.begin(), ticks.end());
      long double median_ticks = ticks[ticks.size() / 2];
      if (ticks.size() % 2 == 0)
        median_ticks =
            static_cast<long double>(ticks[ticks.size() / 2 - 1]) / 2 +
            static_cast<long double>(ticks[ticks.size() / 2]) / 2;
      // Sum in a wider floating type to avoid overflowing a uint64_t total.
      // Individual samples remain exact integers in the raw output above.
      long double total_ticks = 0;
      for (const auto value : ticks)
        total_ticks += value;
      const auto nanoseconds_per_tick = tick_period_seconds * 1000000000;
      // HSB report parsers expect this heading, field order and nanosecond units.
      std::cout << "\n=== PTP Round-Trip Latency ===\n"
                << "  Samples:  " << ticks.size() << '\n'
                << "  Min:      " << std::fixed << std::setprecision(0)
                << ticks.front() * nanoseconds_per_tick << " ns\n"
                << "  Max:      " << ticks.back() * nanoseconds_per_tick
                << " ns\n"
                << "  Avg:      " << std::fixed << std::setprecision(1)
                << total_ticks / ticks.size() * nanoseconds_per_tick << " ns\n"
                << std::defaultfloat
                << std::setprecision(std::numeric_limits<double>::max_digits10)
                << "  Median:   " << median_ticks * nanoseconds_per_tick
                << " ns\n";
    }

    // 5. Destroy only after copying/reporting results. A cleanup failure must
    // not produce PASS; a failed destroy retains the handle/library for retry.
    plugin.ok(plugin.api->destroy(plugin.handle, plugin.timeout_ns), "destroy");
    plugin.handle = nullptr;
    std::cout << "PASS: " << expected.size() << '/' << count
              << " replies; latency_samples=" << samples.size() << "/" << count
              << "; provider=" << library << '\n';
    return 0;
  // Keep incomplete capture, provider failures and bad data distinguishable
  // for the runner. Unavailable timing follows the successful path above.
  } catch (const IncompleteRun &error) {
    std::cerr << "INCOMPLETE: " << error.what() << '\n';
    return 2;
  } catch (const ProviderError &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 3;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
