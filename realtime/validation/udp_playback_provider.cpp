/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file udp_playback_provider.cpp
/// @brief Play a finite request batch against the UDP RTH bridge.
///
/// create copies and pads requests to the advertised slot size. A worker sends
/// them at the requested interval, captures replies and measures RTT using one
/// host clock. Direct socket reads preserve actual datagram lengths for checks;
/// the transceiver ring API pads short datagrams and hides their original size.
#include "playback_provider_support.h"
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <cerrno>
#include <future>
#include <memory>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/timerfd.h>
#include <unistd.h>

using namespace cudaq::realtime::validation;
namespace {
// Diagnostics before a handle exists belong to the calling thread.
thread_local std::string create_error;
// One context owns one batch. Serialized API calls manage lifecycle state;
// The worker owns I/O and capture until its future is joined. Cancellation
// and diagnostic progress are atomic; packet storage survives until destroy.
struct Context {
  State state = State::Created; ///< Lifecycle state visible to API calls.
  // Destination host override and diagnostic for the latest failed operation.
  std::string host = "127.0.0.1", error;
  uint64_t request_spacing_ns = 0; ///< Minimum send start-to-start interval.
  int socket_fd = -1; ///< Owned connected, nonblocking UDP socket.
  int timer_fd = -1; ///< One-shot monotonic timer for sends and completion checks.
  int cancel_fd = -1; ///< Wake the socket wait when destroy requests cancellation.
  // Reuse scratch storage even for empty polls. Captured records allocate only
  // their actual datagram length, not the maximum UDP packet size.
  std::array<uint8_t, 65536> receive_buffer;
  std::vector<std::vector<uint8_t>> frames; ///< Copied, transport-padded requests.
  std::deque<Record> records; ///< Owned capture bytes in arrival order.
  ResultStorage results; ///< Frozen views into records after worker completion.
  RunProgress run_progress; ///< Timeout diagnostics readable while I/O is active.
  // Whole-run deadline and inactivity interval required to finish capture.
  uint64_t run_timeout_ns = 0, capture_quiet_ns = 0;
  bool run_timed_out = false; ///< Preserve TIMEOUT on subsequent status checks.
  Clock::time_point started_at; ///< Set by start before launching the worker.
  std::map<uint32_t, Clock::time_point> sent; ///< Send time by request ID.
  size_t next = 0; ///< Index of the next unsent request.
  std::atomic<bool> cancel{false}; ///< destroy requests worker cancellation.
  std::future<void> job; ///< Worker completion and exception propagation.
  Clock::time_point last_send_at{}; ///< Origin for the next spacing interval.
  bool send_blocked = false; ///< Wait for socket writability after backpressure.
  ~Context() {
    if (socket_fd >= 0)
      ::close(socket_fd);
    if (timer_fd >= 0)
      ::close(timer_fd);
    if (cancel_fd >= 0)
      ::close(cancel_fd);
  }
  // Bound each drain so inbound traffic cannot starve sends or deadline checks.
  // Keep malformed/unmatched datagrams; the caller, not capture, rejects them.
  void receive(size_t budget = 64) {
    for (size_t packet_index = 0; packet_index < budget; ++packet_index) {
      // Hold the largest IPv4 UDP datagram so recv cannot silently truncate
      // it to the expected request size. Short and empty packets also survive.
      auto byte_count = ::recv(socket_fd, receive_buffer.data(),
                               receive_buffer.size(), MSG_DONTWAIT);
      // Timestamp immediately after recv; later decoding/allocation must not
      // inflate the host-observed RTT for this datagram.
      const auto now = Clock::now();
      if (byte_count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        return;
      if (byte_count < 0 && errno == EINTR)
        continue;
      if (byte_count < 0)
        throw std::runtime_error("UDP recv: " +
                                 std::string(std::strerror(errno)));
      Record record;
      record.bytes.assign(receive_buffer.begin(),
                          receive_buffer.begin() + byte_count);
      // Match an ID only to assign timing. This is not a validity check:
      // duplicate or corrupt packets still reach the application's verifier.
      if (byte_count >= 24) {
        uint32_t request_id = little_endian(record.bytes.data() + 12, 4);
        auto sent_request = sent.find(request_id);
        if (sent_request != sent.end()) {
          record.rtt_ticks = (now - sent_request->second).count();
        }
      }
      // Allow extra packets so the verifier can diagnose duplicates. Beyond
      // this bounded storage, report capture failure instead of dropping data.
      if (records.size() >= frames.size() * 2 + 64)
        throw std::runtime_error("UDP capture queue overflow");
      records.push_back(std::move(record));
      run_progress.captured = records.size();
    }
    // Ordinary progress may stop after its small drain budget. The larger
    // final drain must actually reach EAGAIN before capture can be frozen.
    if (budget > 64)
      throw std::runtime_error(
          "UDP capture did not drain at completion boundary");
  }
  void progress() {
    // Service receive first on every pass, including while more sends remain;
    // request pacing must not leave replies queued until transmission finishes.
    receive();
    if (next < frames.size() &&
        (next == 0 || elapsed_ns(last_send_at) >= request_spacing_ns)) {
      auto &frame = frames[next];
      auto before = Clock::now();
      auto byte_count =
          ::send(socket_fd, frame.data(), frame.size(), MSG_DONTWAIT);
      // Retry this same frame later on temporary backpressure. Do not advance
      // its index or assign a send timestamp until the datagram is accepted.
      if (byte_count < 0 &&
          (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
        send_blocked = errno != EINTR;
        return;
      }
      if (byte_count != static_cast<ssize_t>(frame.size()))
        throw std::runtime_error("UDP send failed");
      const auto request_id = little_endian(frame.data() + 12, 4);
      sent[request_id] = before;
      // Start-to-start pacing is independent of reply arrival. If scheduling
      // or socket backpressure delayed this send, do not catch up with a burst.
      last_send_at = before;
      send_blocked = false;
      ++next;
    }
  }

  // Socket readiness wakes immediately for a reply, independently of pacing.
  // Use timerfd for the next deadline: timed poll/condition-variable waits may
  // add timer slack, which is significant at microsecond request intervals.
  void wait_for_activity(uint64_t wait_ns) {
    if (!wait_ns || cancel)
      return;
    itimerspec timer{};
    timer.it_value.tv_sec = wait_ns / 1000000000;
    timer.it_value.tv_nsec = wait_ns % 1000000000;
    if (::timerfd_settime(timer_fd, 0, &timer, nullptr))
      throw std::runtime_error("UDP timer arm failed");
    pollfd descriptors[] = {
        {socket_fd, short(POLLIN | (send_blocked ? POLLOUT : 0)), 0},
        {timer_fd, POLLIN, 0},
        {cancel_fd, POLLIN, 0}};
    if (::poll(descriptors, 3, -1) < 0 && errno != EINTR)
      throw std::runtime_error("UDP readiness wait failed");
    // The next pass drains replies and checks deadlines/cancellation. Rearming
    // the one-shot timer clears its previous expiration; no timer count is used.
  }

  void playback() {
    BatchCompletion completion(frames.size(), run_timeout_ns, capture_quiet_ns,
                               started_at);
    // Playback and completion progress without calls into check_complete.
    while (!cancel) {
      progress();
      if (next < frames.size())
        run_progress.phase = "sending requests";
      else if (records.size() < frames.size())
        run_progress.phase = "waiting for replies";
      else
        run_progress.phase = "waiting for capture quiet period";
      if (completion.ready(records.size()) && next == frames.size()) {
        // Include datagrams already queued at the proposed boundary. New
        // activity restarts stabilization; never truncate an unbounded drain.
        receive(frames.size() * 2 + 65);
        if (completion.ready(records.size()))
          break;
      }
      // Sleep until a reply, cancellation, the next send or a completion
      // deadline. Late sends establish the next interval; never catch up with
      // a burst or let reply arrival determine request spacing.
      uint64_t wait_ns = completion.wait_ns(next == frames.size());
      if (next < frames.size()) {
        if (!send_blocked) {
          const auto since_send_ns =
              next ? elapsed_ns(last_send_at) : request_spacing_ns;
          wait_ns = std::min(wait_ns, since_send_ns >= request_spacing_ns
                                          ? 0
                                          : request_spacing_ns - since_send_ns);
        }
      }
      wait_for_activity(wait_ns);
    }
    // Cancellation is a cleanup path, not a completed capture. Only normal
    // completion constructs result views for check_complete to publish.
    if (cancel)
      return;
    run_progress.phase = "preparing results";
    results = result_views(records, Clock::period::num, Clock::period::den);
    completion.check_deadline();
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
    for (int argument_index = 0; argument_index < argument_count;
         ++argument_index) {
      deadline.check();
      auto [key, value] = parse_provider_option(arguments[argument_index]);
      if (key == "--host")
        context->host = value;
      else
        throw std::invalid_argument("Unknown UDP option: " + key);
    }
    // Numeric addresses avoid an unbounded DNS lookup during timed setup.
    in_addr address{};
    if (inet_pton(AF_INET, context->host.c_str(), &address) != 1)
      throw std::invalid_argument("UDP requires a numeric IPv4 host address");
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
  context->request_spacing_ns = params->request_spacing_ns;
  context->run_timeout_ns = params->run_timeout_ns;
  context->capture_quiet_ns = params->capture_quiet_ns;
  try {
    deadline.check();
    auto fields = endpoint_fields(params->rth_endpoint);
    // The RTH reports its chosen port and slot size, including an ephemeral
    // port assigned during bind. Missing keys fail before any socket is opened.
    auto port = parse_unsigned_integer(fields.at("port"), 65535);
    if (!port)
      return CUDAQ_PLAYBACK_PROVIDER_ERROR;
    // 65,507 is the IPv4 UDP payload limit. Apply the batch memory limit to
    // padded slots, since padding can make storage much larger than RPC bytes.
    const auto slot_size =
        parse_unsigned_integer(fields.at("slot_size"), 65507);
    if (slot_size < params->request_size_bytes ||
        params->request_count > 1000000 ||
        params->request_count * slot_size > 64 * 1024 * 1024)
      throw std::invalid_argument(
          "UDP endpoint slot size or batch exceeds limits");
    // The bridge receives fixed-size slots; canonical request bytes exclude
    // this transport padding and remain unchanged at the front of each slot.
    std::vector<std::vector<uint8_t>> frames;
    for (size_t request_index = 0; request_index < params->request_count;
         ++request_index) {
      deadline.check();
      const auto *request =
          params->requests + request_index * params->request_size_bytes;
      frames.emplace_back(request, request + params->request_size_bytes);
      frames.back().resize(slot_size, 0);
    }
    context->socket_fd =
        ::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (context->socket_fd < 0)
      throw std::runtime_error("UDP socket failed");
    context->timer_fd =
        ::timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
    context->cancel_fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (context->timer_fd < 0 || context->cancel_fd < 0)
      throw std::runtime_error("UDP wait descriptor creation failed");
    sockaddr_in peer{};
    peer.sin_family = AF_INET;
    peer.sin_port = htons(port);
    inet_pton(AF_INET, context->host.c_str(), &peer.sin_addr);
    // UDP connect selects the peer and filters received datagrams; it sends
    // no validation request and needs no remote handshake.
    if (::connect(context->socket_fd, reinterpret_cast<sockaddr *>(&peer),
                  sizeof(peer)))
      throw std::runtime_error("UDP connect failed");
    context->frames = std::move(frames);
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
                 uint64_t cleanup_timeout_ns) {
  auto *context = static_cast<Context *>(handle);
  if (!context)
    return CUDAQ_PLAYBACK_PROVIDER_OK;
  if (!valid_duration_ns(cleanup_timeout_ns)) {
    context->error = "Cleanup duration exceeds supported nanosecond range";
    return CUDAQ_PLAYBACK_PROVIDER_ERROR;
  }
  // Wake a worker sleeping between sends before waiting for its completion.
  // The socket remains open until the worker can no longer access it.
  context->cancel = true;
  if (context->cancel_fd >= 0) {
    const uint64_t notification = 1;
    ssize_t written;
    do {
      written = ::write(context->cancel_fd, &notification, sizeof(notification));
    } while (written < 0 && errno == EINTR);
    // EAGAIN means an earlier notification already made the descriptor ready.
    if (written < 0 && errno != EAGAIN) {
      context->error = "UDP cancellation wake failed";
      return CUDAQ_PLAYBACK_PROVIDER_ERROR;
    }
  }
  try {
    if (!join_worker(context->job, cleanup_timeout_ns)) {
      context->error = "UDP destroy deadline: worker is still active";
      context->state = State::Faulted;
      return CUDAQ_PLAYBACK_PROVIDER_TIMEOUT;
    }
  } catch (...) {
    // get() joined the failed worker; destruction still releases the socket.
  }
  delete context;
  return CUDAQ_PLAYBACK_PROVIDER_OK;
}
cudaq_playback_provider_status_t
create_provider(int argument_count, const char *const *arguments,
                const cudaq_playback_provider_create_params_t *params,
                uint64_t setup_timeout_ns,
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
cudaq_get_playback_provider_interface(uint32_t version) {
  static const cudaq_playback_provider_interface_t api = {
      create_provider,
      start_playback<Context>,
      check_playback_complete<Context>,
      get_playback_results<Context>,
      destroy_provider,
      last_error};
  return version == CUDAQ_PLAYBACK_PROVIDER_ABI_VERSION ? &api : nullptr;
}
