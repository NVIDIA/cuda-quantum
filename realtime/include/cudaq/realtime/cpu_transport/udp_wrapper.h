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

#ifndef CPU_UDP_WRAPPER_H
#define CPU_UDP_WRAPPER_H

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

/// Start the RX and TX pump threads. Returns 1 on success.
int cpu_udp_start(cpu_udp_transceiver_t handle);

/// Stop the pump threads and close the socket. Idempotent.
void cpu_udp_close(cpu_udp_transceiver_t handle);

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

#endif // CPU_UDP_WRAPPER_H
