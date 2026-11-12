/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hsb_playback_provider.cpp
/// @brief Control HSB FPGA playback and capture through the provider ABI.
///
/// create configures the RoCE destination, stages requests in BRAM and arms
/// ILA capture. start launches single-pass playback; a worker waits for capture
/// stabilization and downloads results. Workload generation and reply checking
/// belong to the caller; this adapter handles the RFSoC register/capture layout.
#include "hsb_capture.h"
#include "playback_provider_support.h"
#include "hsb_playback_io.h"
#include <cstddef>
#include <cstdint>
#include <arpa/inet.h>
#include <atomic>
#include <condition_variable>
#include <future>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <memory>
#include <mutex>

using namespace cudaq::realtime::validation;
using namespace cudaq::realtime::validation::hsb;
namespace {
// Diagnostics before a handle exists belong to the calling thread.
thread_local std::string create_error;
// One context owns one batch. Serialized API calls manage lifecycle state;
// The worker owns I/O and capture until its future is joined. Cancellation
// and diagnostic progress are atomic; packet storage survives until destroy.
struct Context {
  State state = State::Created; ///< Lifecycle state visible to API calls.
  // FPGA control address from the RTH endpoint, and latest failure diagnostic.
  std::string peer_ip, error;
  // RTH ring stride/count and FPGA request interval in hardware timer ticks.
  std::uint32_t slot_size = 0, num_slots = 0, timer_ticks = 0;
  std::uint32_t qp = 0, rkey = 0; ///< RTH destination QP and remote memory key.
  std::uint64_t address = 0; ///< RTH receive buffer address advertised by the bridge.
  std::vector<std::vector<std::uint8_t>> frames; ///< Copied, transport-padded requests.
  std::deque<Record> records; ///< Owned capture bytes in arrival order.
  ResultStorage results; ///< Frozen views into records after worker completion.
  RunProgress run_progress; ///< Timeout diagnostics readable while I/O is active.
  // Whole-run deadline and inactivity interval required to finish capture.
  std::uint64_t run_timeout_ns = 0, capture_quiet_ns = 0;
  bool run_timed_out = false; ///< Preserve TIMEOUT on subsequent status checks.
  Clock::time_point started_at; ///< Set by start before launching the worker.
  std::shared_ptr<hololink::Hololink> hololink_device; ///< Owned SDK control session.
  std::future<void> job; ///< Worker completion and exception propagation.
  std::atomic<bool> cancel{false}; ///< destroy requests worker cancellation.
  std::mutex mutex; ///< Protects the interruptible worker wait.
  std::condition_variable capture_condition; ///< Interrupt capture wait on cancel.
  std::size_t captured_packets = 0; ///< Last checked ILA count for this one-beat profile.

  // Prepare hardware without transmitting. The RTH ring stride is separate
  // from each 64-byte BRAM window used by this FPGA playback profile.
  void configure() {
    auto metadata = hololink::Enumerator::find_channel(peer_ip);
    hololink::DataChannel::use_sensor(metadata, 0);
    hololink::DataChannel channel(metadata);
    hololink_device = channel.hololink();
    hololink_device->start();
    hololink_device->reset();
    if (cancel)
      return;
    if (!hololink_device->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET,
                                       PLAYER_DISABLE))
      throw std::runtime_error("Disable player failed");
    // Program the destination memory credentials and ring geometry published
    // by the RTH. The FPGA will place successive requests into that receive ring.
    channel.authenticate(qp, rkey);
    channel.configure_roce(address, 64, slot_size, num_slots, ROCEV2_UDP_PORT);
    hololink::Hololink::WriteData config;
    // One request occupies one BRAM window. Window count bounds this single
    // pass; the timer controls spacing independently of response arrival.
    config.queue_write_uint32(PLAYER_ADDR + PLAYER_WINDOW_SIZE_OFFSET, 64);
    config.queue_write_uint32(PLAYER_ADDR + PLAYER_WINDOW_NUMBER_OFFSET,
                              frames.size());
    config.queue_write_uint32(PLAYER_ADDR + PLAYER_TIMER_OFFSET, timer_ticks);
    if (!hololink_device->write_uint32(config))
      throw std::runtime_error("Configure player failed");
    // Readback catches control-plane staging errors before any timed traffic.
    write_bram(*hololink_device, frames, 64);
    if (!verify_bram(*hololink_device, frames, 64))
      throw std::runtime_error("BRAM readback mismatch");
    // Clear old observations and arm capture during setup, before start can
    // send the first request. Readout belongs to the playback worker below.
    ila_disable(*hololink_device);
    ila_reset(*hololink_device);
    auto value = hololink_device->read_uint32(METADATA_PACKET_ADDR);
    if (!hololink_device->write_uint32(METADATA_PACKET_ADDR,
                                       value | (1u << 16)) ||
        !hololink_device->write_uint32(SIF_TX_THRESHOLD_ADDR,
                                       SIF_TX_THRESHOLD_IMMEDIATE))
      throw std::runtime_error("Configure SIF failed");
    ila_enable(*hololink_device);
  }
  std::size_t capture_count() {
    const auto count = ila_sample_count(*hololink_device);
    const auto status = hololink_device->read_uint32(ILA_BASE_ADDR + 0x80);
    // A full/wrapped capture may have lost replies even if its count looks
    // plausible. Check both hardware status and monotonic progress.
    check_capture_extent(count, status, captured_packets);
    // This profile requires each response to fit one 512-bit ILA beat.
    // Check that assumption during readout before returning results.
    captured_packets = count;
    run_progress.captured = count;
    run_progress.phase = count >= frames.size()
                             ? "waiting for capture quiet period"
                             : "waiting for replies";
    return count;
  }
  void playback() {
    if (!hololink_device->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET,
                                       PLAYER_ENABLE_SINGLEPASS))
      throw std::runtime_error("Start player failed");
    run_progress.phase = "waiting for replies";
    BatchCompletion completion(frames.size(), run_timeout_ns, capture_quiet_ns,
                               started_at);
    // HSB exposes capture progress, not a usable TX-complete register. Apply
    // the supplied stabilization policy internally; the app checks reply
    // contents.
    while (!cancel) {
      if (completion.ready(capture_count()))
        break;
      // This is control-plane polling, not the timing source. RTT comes from
      // FPGA timestamps; waking less often only delays result availability.
      std::unique_lock<std::mutex> lock(mutex);
      capture_condition.wait_for(lock, std::chrono::milliseconds(10),
                                 [&] { return cancel.load(); });
    }
    if (cancel)
      return;
    // RAM reads are gated during capture and disabling clears the count.
    // Preserve this prefix boundary before disabling, then download it.
    const auto count = captured_packets;
    run_progress.phase = "reading capture";
    ila_disable(*hololink_device);
    auto samples = ila_dump(*hololink_device, count);
    run_progress.phase = "decoding capture";
    CaptureDecoder decoder;
    for (const auto &sample : samples) {
      // The completion loop treated each sample as one packet. Require both
      // valid and last flags so that assumption is checked before publishing.
      if ((sample[16] & 3) != 3)
        throw std::runtime_error(
            "HSB profile requires one valid, final beat per captured packet");
      decoder.append(sample, records);
    }
    decoder.finish();
    // Captured PTP values encode seconds/nanoseconds; one unit is 1 ns.
    results = result_views(records, 1, 1000000000);
    completion.check_deadline();
    // Do not disable the player here: a capture count is not TX-completion
    // proof. The app verifies these bytes first; destroy then cleans up or
    // cancels.
  }
};
// Parse provider options before publishing an owned handle. Later setup
// errors return that handle so the caller can clean up partial resources.
cudaq_playback_provider_status_t
allocate_context(int argument_count, const char *const *arguments,
                 cudaq_playback_provider_t *out_provider,
                 const SetupDeadline &deadline) {
  try {
    auto context = std::make_unique<Context>();
    // This hardware profile needs no provider options: all connection values
    // come from the RTH endpoint and all workload settings from create params.
    if (argument_count)
      throw std::invalid_argument("HSB accepts no provider options: " +
                                  std::string(arguments[0]));
    deadline.check();
    *out_provider = context.release();
    return CUDAQ_PLAYBACK_PROVIDER_OK;
  } catch (const SetupTimeout &error) {
    create_error = error.what();
    return CUDAQ_PLAYBACK_PROVIDER_TIMEOUT;
  } catch (const std::exception &error) {
    create_error = error.what();
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  }
}
cudaq_playback_provider_status_t
stage_batch(cudaq_playback_provider_t handle,
            const cudaq_playback_provider_create_params_t *params,
            const SetupDeadline &deadline) {
  auto *context = static_cast<Context *>(handle);
  // This profile has 512 BRAM entries and uses one 64-byte beat per request.
  // Reject workloads it cannot stage before making control-plane calls.
  if (params->request_count > 512)
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  if (params->request_size_bytes > 64 || params->request_spacing_ns < 1000 ||
      params->request_spacing_ns > 1000000000ULL ||
      params->request_spacing_ns % 1000)
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  // This profile accepts whole microseconds. Validate before conversion;
  // the bounded product fits the FPGA's 32-bit timer register.
  context->timer_ticks = static_cast<std::uint32_t>(
      (params->request_spacing_ns / 1000) * RF_SOC_TIMER_SCALE);
  context->run_timeout_ns = params->run_timeout_ns;
  context->capture_quiet_ns = params->capture_quiet_ns;
  try {
    deadline.check();
    auto fields = endpoint_fields(params->rth_endpoint);
    // RoCE QP numbers are 24-bit; keys are 32-bit and buffer addresses 64-bit.
    // Parsing each width explicitly prevents truncation during SDK setup.
    auto qp = parse_unsigned_integer(fields.at("qp"), 0xffffff),
         rkey = parse_unsigned_integer(fields.at("rkey"));
    auto address = parse_unsigned_integer(fields.at("buffer_addr"), UINT64_MAX);
    if (!qp || address % 128)
      return CUDAQ_PLAYBACK_PROVIDER_ERROR;
    // The RTH owns its peer and receive ring. Use the published values rather
    // than accepting independent copies through provider options.
    const auto &peer_ip = fields.at("peer_ip");
    in_addr peer_address{};
    if (inet_pton(AF_INET, peer_ip.c_str(), &peer_address) != 1 ||
        peer_address.s_addr == INADDR_ANY ||
        peer_address.s_addr == INADDR_BROADCAST)
      throw std::invalid_argument("HSB endpoint needs a valid peer_ip");
    context->slot_size =
        parse_unsigned_integer(fields.at("slot_size"), 1048576);
    context->num_slots = parse_unsigned_integer(fields.at("num_slots"), 4096);
    // HSB ring addressing requires 128-byte strides and power-of-two slot
    // counts. These constraints concern RTH memory, not the BRAM window size.
    if (context->slot_size < 128 || context->slot_size % 128 ||
        !context->num_slots || (context->num_slots & (context->num_slots - 1)))
      throw std::invalid_argument("Invalid HSB endpoint ring geometry");
    // Copy every request before starting asynchronous setup: create's caller
    // may reuse its input storage as soon as this API call returns.
    context->peer_ip = peer_ip;
    for (std::size_t request_index = 0; request_index < params->request_count;
         ++request_index) {
      deadline.check();
      const auto *request =
          params->requests + request_index * params->request_size_bytes;
      context->frames.emplace_back(request,
                                   request + params->request_size_bytes);
      context->frames.back().resize(64, 0);
    }
    context->qp = qp;
    context->rkey = rkey;
    context->address = address;
    context->state =
        State::Faulted; // Partial hardware setup requires destroy on error.
    // SDK calls may block. Wait only within the create budget; on timeout the
    // context retains the worker and destroy can wait for it safely.
    context->job =
        std::async(std::launch::async, [context] { context->configure(); });
    if (!join_worker(context->job, deadline.remaining_ns())) {
      context->state = State::Faulted;
      context->error = "HSB create deadline";
      return CUDAQ_PLAYBACK_PROVIDER_TIMEOUT;
    }
    deadline.check();
    context->state = State::Prepared;
    return CUDAQ_PLAYBACK_PROVIDER_OK;
  } catch (const SetupTimeout &error) {
    context->error = error.what();
    context->state = State::Faulted;
    return CUDAQ_PLAYBACK_PROVIDER_TIMEOUT;
  } catch (const std::logic_error &error) {
    context->error = error.what();
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  } catch (const std::exception &error) {
    context->error = error.what();
    context->state = State::Faulted;
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  }
}
// Never delete context while its worker can access it. A timeout retains the
// handle and loaded library so the caller can retry cleanup.
cudaq_playback_provider_status_t
destroy_provider(cudaq_playback_provider_t handle,
                 std::uint64_t cleanup_timeout_ns) {
  auto *context = static_cast<Context *>(handle);
  if (!context)
    return CUDAQ_PLAYBACK_PROVIDER_OK;
  if (!valid_duration_ns(cleanup_timeout_ns)) {
    context->error = "Cleanup duration exceeds supported nanosecond range";
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  }
  // Parsing can fail before a session or worker exists. Such a partial handle
  // needs only local deletion, not a device cleanup attempt.
  if (context->state == State::Created && !context->hololink_device) {
    delete context;
    return CUDAQ_PLAYBACK_PROVIDER_OK;
  }
  context->cancel = true;
  context->capture_condition.notify_all();
  const auto cleanup_started_at = Clock::now();
  try {
    try {
      if (!join_worker(context->job, cleanup_timeout_ns)) {
        context->state = State::Faulted;
        context->error = "HSB destroy deadline: worker is still active";
        return CUDAQ_PLAYBACK_PROVIDER_TIMEOUT;
      }
    } catch (const std::exception &error) {
      // The future has joined even when its task threw. Continue stopping any
      // session that setup managed to create before the failure.
      context->error = error.what();
    }
    // Stop hardware only after the setup/playback worker has joined. Both
    // joining and device cleanup consume the same destroy-call budget.
    if (context->hololink_device) {
      context->job = std::async(std::launch::async, [context] {
        if (!context->hololink_device->write_uint32(
                PLAYER_ADDR + PLAYER_ENABLE_OFFSET, PLAYER_DISABLE))
          throw std::runtime_error("HSB stop failed");
        ila_disable(*context->hololink_device);
        context->hololink_device->stop();
        // Clear the session only after stop succeeds. Failed cleanup retains
        // it in the context so a later destroy can retry.
        context->hololink_device.reset();
      });
      const auto elapsed = elapsed_ns(cleanup_started_at);
      const auto remaining =
          elapsed >= cleanup_timeout_ns ? 0 : cleanup_timeout_ns - elapsed;
      if (!join_worker(context->job, remaining)) {
        context->state = State::Faulted;
        context->error =
            "HSB destroy deadline: hardware cleanup is still active";
        return CUDAQ_PLAYBACK_PROVIDER_TIMEOUT;
      }
    }
    delete context;
    return CUDAQ_PLAYBACK_PROVIDER_OK;
  } catch (const std::exception &error) {
    context->error = error.what();
    context->state = State::Faulted;
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  }
}
cudaq_playback_provider_status_t
create_provider(int argument_count, const char *const *arguments,
                const cudaq_playback_provider_create_params_t *params,
                std::uint64_t setup_timeout_ns,
                cudaq_playback_provider_t *out_provider) {
  const SetupDeadline deadline(setup_timeout_ns);
  if (!out_provider) {
    create_error = "out_provider is required";
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  }
  // A rejection before allocation returns no handle. Once allocation
  // succeeds, stage_batch leaves the partial handle available for destroy.
  *out_provider = nullptr;
  if (!setup_timeout_ns || !valid_duration_ns(setup_timeout_ns) ||
      !valid_arguments(argument_count, arguments) ||
      (!valid_create_params(params) || !params->rth_endpoint ||
       !*params->rth_endpoint)) {
    create_error =
        "Invalid arguments, RTH endpoint, request buffer/size/count or timeout";
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  }
  auto status =
      allocate_context(argument_count, arguments, out_provider, deadline);
  if (status != CUDAQ_PLAYBACK_PROVIDER_OK)
    return status;
  // Partial setup retains its handle for cleanup; no worker borrows caller
  // inputs.
  return stage_batch(*out_provider, params, deadline);
}
const char *last_error(cudaq_playback_provider_t handle) {
  if (!handle)
    return create_error.c_str();
  // Simple validation/lifecycle rejections may have no detailed exception
  // text. The callback still returns a nonempty diagnostic as the ABI requires.
  const auto &message = static_cast<Context *>(handle)->error;
  return message.empty()
             ? "Invalid lifecycle, endpoint or unsupported batch configuration"
             : message.c_str();
}
} // namespace
// The only required export: return a static table for the requested ABI.
// Common helpers implement status/result calls without transport I/O in them.
extern "C" const cudaq_playback_provider_interface_t *
cudaq_get_playback_provider_interface(std::uint32_t version) {
  static const cudaq_playback_provider_interface_t api = {
      create_provider,
      start_playback<Context>,
      check_playback_complete<Context>,
      get_playback_results<Context>,
      destroy_provider,
      last_error};
  return version == CUDAQ_PLAYBACK_PROVIDER_ABI_VERSION ? &api : nullptr;
}
