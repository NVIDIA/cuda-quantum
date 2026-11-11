/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file playback_provider_interface.h
/// @brief Control-plane C ABI for runtime-loaded validation playback providers.
///
/// The caller supplies canonical requests; the provider stages/transmits them
/// and captures replies. The caller verifies the returned bytes and timing.
/// Typical use: create, start, check_complete until COMPLETE, get_results,
/// destroy. Setup failures may also require destroy for a partial handle.
///
/// Each instance handles one finite validation batch. Control durations are
/// uint64_t nanoseconds; latency measurements use integer clock ticks with a
/// seconds-per-tick ratio.
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDAQ_PLAYBACK_PROVIDER_ABI_VERSION 1u
#define CUDAQ_PLAYBACK_PROVIDER_GETTER "cudaq_get_playback_provider_interface"
#define CUDAQ_PLAYBACK_PROVIDER_RTT_UNAVAILABLE UINT64_MAX

typedef void *cudaq_playback_provider_t;
typedef int32_t cudaq_playback_provider_status_t;
// Zero means success; other outcomes use positive named codes. Compare the
// constants explicitly; RUNNING and COMPLETE apply only to check_complete.
enum {
  CUDAQ_PLAYBACK_PROVIDER_OK = 0,
  CUDAQ_PLAYBACK_PROVIDER_RUNNING =
      1, ///< check_complete: playback/readout is still active.
  CUDAQ_PLAYBACK_PROVIDER_COMPLETE =
      2, ///< check_complete: immutable results are available.
  CUDAQ_PLAYBACK_PROVIDER_ERROR =
      3, ///< Operation failed; see last_error and destroy.
  CUDAQ_PLAYBACK_PROVIDER_TIMEOUT =
      4 ///< Deadline expired; destroy any returned handle.
};

/// Request data and shared run settings, borrowed during create only. The
/// plugin copies all inputs it needs, including on failed create. Only the
/// provider interprets the opaque RTH endpoint.
typedef struct {
  /// Opaque connection information published by the realtime host (RTH).
  /// NULL when the bridge publishes none; the provider decides whether it is
  /// required.
  const char *rth_endpoint;
  /// Consecutive equal-sized RPC requests, with no transport padding.
  /// Each request has its own ID and payload. Bytes 16..23 are provider-owned
  /// timing storage; preserve all other bytes. Caller supplies readable storage
  /// for request_size_bytes * request_count bytes; the product must fit in
  /// size_t.
  const uint8_t *requests;
  /// Bytes per request: 24-byte header plus payload.
  size_t request_size_bytes;
  size_t request_count; ///< Nonzero, unique request IDs.
  /// Positive interval between request starts, in nanoseconds. Schedule
  /// requests independently of replies; do not send catch-up bursts if late.
  uint64_t request_spacing_ns;
  uint64_t run_timeout_ns; ///< From start through capture and result download.
  /// After at least request_count packets, keep capturing until no new packet
  /// arrives for this interval. HSB watches the ILA count; UDP keeps receiving
  /// datagrams. Every new packet restarts the wait. Positive and shorter than
  /// run_timeout_ns.
  uint64_t capture_quiet_ns;
} cudaq_playback_provider_create_params_t;

/// One complete observed packet, including malformed/duplicate/unexpected ones.
/// Preserve actual length; trailing transport padding is allowed. Never hide
/// capture overflow or incomplete packets. RTT uses one clock, not host
/// download time. The provider decodes native timing or subtracts timestamps
/// into integer RTT ticks, preserving measurement precision. Packet bytes stay
/// unchanged; the app uses the shared tick period to report nanoseconds.
/// Use RTT_UNAVAILABLE for a packet with no measured RTT.
typedef struct {
  const uint8_t *data; ///< Captured packet bytes owned by the provider.
  size_t size_bytes; ///< Actual captured length, including any transport padding.
  uint64_t rtt_ticks; ///< Round-trip ticks, or RTT_UNAVAILABLE when not measured.
} cudaq_playback_provider_reply_t;

/// Entire capture in acquisition order. All pointers and packet bytes
/// are plugin-owned, immutable after COMPLETE, and valid until successful
/// destroy.
typedef struct {
  const cudaq_playback_provider_reply_t *replies; ///< Stable capture array.
  size_t reply_count; ///< Captured packets, including invalid or duplicate ones.
  /// Shared by every measured RTT in this run. The clock fields are empty when
  /// no reply has measured RTT; individual replies may still have no RTT.
  /// One tick is tick_period_numerator / tick_period_denominator seconds.
  /// Both are positive when any RTT is measured, otherwise both are zero.
  /// This exact scale describes timestamp units, not clock accuracy/resolution.
  uint64_t tick_period_numerator;
  uint64_t tick_period_denominator;
} cudaq_playback_provider_results_t;

/// One handle owns one batch; calls are serialized and exceptions never cross
/// this ABI. COMPLETE means results are ready. The app checks request IDs,
/// headers and payloads before reporting PASS or FAIL.
typedef struct {
  /// Validate settings, copy inputs, connect/configure and stage requests.
  /// Send no test requests. Capture must be ready before start transmits.
  /// Set *out_provider=NULL first. A failed/timed-out setup may return a
  /// partial handle, which the caller must destroy. setup_timeout_ns bounds
  /// setup separately from the batch's run deadline. It must be positive and
  /// covers the whole call, including parsing and copying. argument_count must
  /// be nonnegative; each argument is a non-NULL NUL-terminated string.
  /// arguments may be NULL only when argument_count is zero. No input pointers
  /// survive this call.
  cudaq_playback_provider_status_t (*create)(
      int argument_count, const char *const *arguments,
      const cudaq_playback_provider_create_params_t *params,
      uint64_t setup_timeout_ns, cudaq_playback_provider_t *out_provider);

  /// Trigger/schedule the prepared batch exactly once; return promptly.
  /// The RTH must be ready. Hardware or a provider worker owns playback,
  /// capture, stabilization and result download independently of status checks.
  /// Repeated start returns ERROR without retransmitting or changing the run.
  cudaq_playback_provider_status_t (*start)(cudaq_playback_provider_t provider);

  /// Nonblocking status only: RUNNING, COMPLETE, ERROR or TIMEOUT.
  /// Before start: ERROR without changing the prepared state. COMPLETE is
  /// stable and means capture has stopped and all its results are in plugin
  /// memory. For this finite validation profile, observe at least
  /// request_count packets followed by capture_quiet_ns without capture
  /// activity before freezing. New activity restarts the quiet period. Counts
  /// include invalid/duplicate packets and never establish correctness or TX
  /// completion. Never disable an active transmitter just because capture
  /// stabilized. A provider unable to implement this policy must reject the
  /// batch. A run deadline or capture loss/overflow cannot yield COMPLETE. No
  /// private timer may shorten capture.
  cudaq_playback_provider_status_t (*check_complete)(
      cudaq_playback_provider_t provider);

  /// After COMPLETE, return OK with the entire result batch. Local read only;
  /// no SDK/peer I/O. Repeated calls return the same pointers and content.
  /// Before COMPLETE: ERROR without changing the run. Clear *results on error.
  cudaq_playback_provider_status_t (*get_results)(
      cudaq_playback_provider_t provider,
      cudaq_playback_provider_results_t *results);

  /// Stop owned activity and release resources, including before
  /// start or after an error. On OK the handle is invalid. On ERROR/TIMEOUT it
  /// remains live for cleanup retry; keep the library loaded. destroy(NULL) is
  /// OK.
  cudaq_playback_provider_status_t (*destroy)(
      cudaq_playback_provider_t provider, uint64_t cleanup_timeout_ns);

  /// Optional callback; set to NULL when diagnostics are not provided.
  /// As the final field, it may be omitted from a C or C++ aggregate initializer
  /// and is then automatically null. No diagnostic implementation is needed.
  /// After a failed operation, returns a non-NULL, nonempty NUL-terminated
  /// diagnostic string, valid until the next operation on the handle.
  /// A NULL provider selects the calling thread's latest create error. No device
  /// I/O. The app uses "Unknown error" only when the callback is absent.
  /// Status codes determine success or failure.
  const char *(*last_error)(cudaq_playback_provider_t provider);
} cudaq_playback_provider_interface_t;

typedef const cudaq_playback_provider_interface_t *(
    *cudaq_get_playback_provider_interface_fn_t)(uint32_t requested_abi);

/// Only required export. Return NULL for unsupported ABI versions. The table
/// layout is fixed by the version and remains valid while the library is
/// loaded.
const cudaq_playback_provider_interface_t *
cudaq_get_playback_provider_interface(uint32_t requested_abi);

#ifdef __cplusplus
}
#endif
