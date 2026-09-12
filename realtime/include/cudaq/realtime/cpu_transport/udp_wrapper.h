/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file udp_wrapper.h
/// @brief C interface to a loopback/Ethernet UDP ring transceiver.
///
/// The plain-UDP counterpart of CpuRoceTransceiver (roce_wrapper.h) for
/// systems without an RDMA NIC, usable over loopback or a real UDP network:
/// it exposes the same ring-buffer contract the
/// RoCE transport provides -- an RX ring the far end's frames land in
/// (rx_flag[slot] = slot data address when fresh, 0 when free) and a TX ring
/// whose published slots (tx_flag[slot] = slot data address) are shipped to
/// the peer and cleared -- so a host dispatcher or DeviceCallChannel wired to
/// these rings is transport-agnostic between UDP and RoCE.
///
/// Wire behavior: one datagram carries one full slot stride. (This is
/// simpler than the RoCE transport, whose TX SGE covers only the actual
/// frame bytes -- cu_frame_size, not the slot stride; this transport has no
/// separate frame-size parameter.) Inbound datagrams larger than this end's
/// slot stride are dropped with a one-time `stderr` warning, so BOTH ENDS MUST
/// USE THE SAME page_size.
/// Arriving frames fill RX slots in strict ring order (the same FIFO the
/// RoCE recv-WQE path provides), with back-pressure until the in-order slot
/// is fully recycled (rx flag cleared by the consumer AND tx response slot
/// drained).
///
/// Deliberately no `ibverbs`, no CUDA: buildable and runnable anywhere.
///
/// UNIFIED MODE
/// ============
///
/// By default this transceiver is an active agent: it moves bytes wire<->ring
/// on its own two threads and a consumer only ever touches the rings.
/// cpu_udp_set_unified inverts that. No threads are started, and the consumer's
/// single thread drives the wire itself through cpu_udp_rx_poll and
/// cpu_udp_tx_publish. It exists for consumers that already own a dispatch loop
/// and would otherwise be running two -- in-tree, the library's unified host
/// dispatcher, reached through the `udp` bridge provider's `--unified` mode.
///
/// The ring contract above then changes in two ways that are not optional:
///
///   * rx_flags is UNUSED. cpu_udp_rx_poll reports a slot by return value
///     instead of publishing an address, and a consumer of this shape never
///     clears an rx flag -- so a flag set here would wedge that slot for good.
///     Slot occupancy is tracked through tx_flags alone.
///
///   * Slot addresses are DERIVED, not carried. The consumer computes
///     rx_data + slot * page_size itself, and cpu_udp_rx_poll places the
///     datagram at exactly that address.
///
/// Every unified call must come from that one thread, which is what licenses
/// the absence of locking on the peer address and the RX cursor. Mixing the two
/// shapes is refused rather than raced: the hooks fail unless unified mode is
/// set, and setting it is refused once running.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a UdpTransceiver.
typedef void *cpu_udp_transceiver_t;

//==============================================================================
// Lifecycle
//==============================================================================

/// Construct a new transceiver with `num_pages` ring slots of `page_size`
/// bytes each (both rings). Returns NULL on invalid arguments or allocation
/// failure. Does not open a socket; call cpu_udp_bind()/cpu_udp_bind_to() or
/// cpu_udp_connect() next.
cpu_udp_transceiver_t cpu_udp_create_transceiver(size_t page_size,
                                                 unsigned num_pages);

/// External-rings variant of cpu_udp_create_transceiver: the caller supplies
/// (and owns; freed by the caller AFTER destroy) the four zero-initialized
/// ring buffers -- e.g. CUDA pinned+mapped memory so a GPU consumer can poll
/// the rings while this transport moves the datagrams.  Buffer sizes:
/// flags = num_pages * 8 bytes; data = num_pages * page_size bytes.
cpu_udp_transceiver_t cpu_udp_create_transceiver_ext(
    size_t page_size, unsigned num_pages, volatile uint64_t *rx_flags,
    volatile uint64_t *tx_flags, uint8_t *rx_data, uint8_t *tx_data);

/// Destroy the transceiver. Idempotent. Implicitly calls cpu_udp_close if the
/// transceiver is still running.
void cpu_udp_destroy_transceiver(cpu_udp_transceiver_t handle);

/// Service end: bind a UDP endpoint for receiving requests. `host` selects
/// the local interface address to listen on ("0.0.0.0" for all interfaces);
/// NULL or "" binds the loopback interface. `port` 0 selects an ephemeral
/// port; read it back with cpu_udp_get_port. Responses go to the source
/// address of the most recent inbound datagram. Returns 1 on success.
int cpu_udp_bind_to(cpu_udp_transceiver_t handle, const char *host,
                    uint16_t port);

/// Convenience form of cpu_udp_bind_to that binds the loopback interface.
int cpu_udp_bind(cpu_udp_transceiver_t handle, uint16_t port);

/// Caller end: connect the socket to the service endpoint. Returns 1 on
/// success.
int cpu_udp_connect(cpu_udp_transceiver_t handle, const char *host,
                    uint16_t port);

/// Bound local UDP port (valid after cpu_udp_bind / cpu_udp_connect).
uint16_t cpu_udp_get_port(cpu_udp_transceiver_t handle);

/// Switch to unified (thread-free) mode; see "Unified mode" above. MUST precede
/// cpu_udp_start, which is what it changes; returns 0 if already running. Once
/// set, cpu_udp_start starts no pump threads and the two hooks below are the
/// only data path.
int cpu_udp_set_unified(cpu_udp_transceiver_t handle, int unified);

/// Start the RX and TX pump threads. Returns 1 on success. Under unified mode
/// there are no such threads: this only marks the transceiver running.
int cpu_udp_start(cpu_udp_transceiver_t handle);

/// Stop the pump threads and close the socket. Idempotent. Under unified mode
/// the consumer MUST stop driving the hooks first -- they use this socket.
void cpu_udp_close(cpu_udp_transceiver_t handle);

//==============================================================================
// Unified-mode data plane (see "Unified mode" above; both refused unless
// cpu_udp_set_unified succeeded)
//==============================================================================

/// Non-blocking RX. Returns 1 and sets *out_slot when a datagram has been
/// placed in that RX slot (at rx_data + slot * page_size, zero-filled past the
/// received bytes). Returns 0 when nothing was ready, when the in-order slot is
/// still occupied (tx_flags[slot] != 0, i.e. a response still in flight), or
/// when the datagram exceeded this end's page_size (dropped, warned once).
/// Never touches rx_flags.
int cpu_udp_rx_poll(cpu_udp_transceiver_t handle, uint32_t *out_slot);

/// Ship TX slot `slot` to the most recent inbound peer as one full-stride
/// datagram. Returns 1 on success. Slot-addressed rather than FIFO, so
/// responses may be published in any order. Does NOT touch tx_flags: the
/// consumer owns them in this shape.
int cpu_udp_tx_publish(cpu_udp_transceiver_t handle, uint32_t slot);

//==============================================================================
// Ring access (addresses are host pointers, same contract as roce_wrapper.h)
//==============================================================================

uint64_t cpu_udp_get_rx_ring_flag_addr(cpu_udp_transceiver_t handle);
uint64_t cpu_udp_get_rx_ring_data_addr(cpu_udp_transceiver_t handle);
uint64_t cpu_udp_get_tx_ring_flag_addr(cpu_udp_transceiver_t handle);
uint64_t cpu_udp_get_tx_ring_data_addr(cpu_udp_transceiver_t handle);

#ifdef __cplusplus
}
#endif
