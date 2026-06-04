/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file roce_wrapper.h
/// @brief C interface to cudaq::realtime::bridge::CpuRoceTransceiver.
///
/// Parallels hololink_wrapper.h (which wraps GpuRoceTransceiver) so the
/// existing bridge-tool patterns can swap in the CPU transport with
/// minimal changes.  Used by:
///   - hsb_bridge_cpu     (Phase 1, FPGA-facing; default tx_mode_send_for_fpga)
///   - CpuRoceChannel     (Phase 2, plugged into PR #4565's DeviceCallChannel
///                         framework; tx_mode_write_with_imm_for_peer)
///   - cpu_roce_test_daemon (Phase 2 service-end;
///   tx_mode_write_with_imm_for_peer)

#ifndef CPU_ROCE_WRAPPER_H
#define CPU_ROCE_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to CpuRoceTransceiver.
typedef void *cpu_roce_transceiver_t;

/// Which RDMA wire verb the TX path uses.  Mirrors CpuRoceTxMode in
/// roce_transceiver.hpp.
typedef enum {
  /// Phase 1: bridge-to-FPGA.  TX issues IBV_WR_SEND because the FPGA's
  /// HSB IP can only receive Sends.
  CPU_ROCE_TX_MODE_SEND_FOR_FPGA = 0,
  /// Phase 2: cpu_roce caller ↔ daemon.  TX issues
  /// IBV_WR_RDMA_WRITE_WITH_IMM targeting the peer's rx_data using the
  /// peer's `rkey`; slot index carried in the IMM.
  CPU_ROCE_TX_MODE_WRITE_WITH_IMM_FOR_PEER = 1,
} cpu_roce_tx_mode_t;

//==============================================================================
// Lifecycle
//==============================================================================

/// Construct a new transceiver.  Returns NULL on invalid arguments
/// (mutually-exclusive forward/rx_only/tx_only/unified, missing peer_rx_*
/// for kWriteWithImmForPeer, non-power-of-two num_pages).  Does not start
/// the transport; call cpu_roce_start() next.
///
/// `forward`, `rx_only`, `tx_only`, `unified` are bool-as-int (0 = false,
/// non-zero = true).  At most one may be true.
///
/// `peer_rx_base_addr` and `peer_rx_rkey` are ignored unless `tx_mode` is
/// CPU_ROCE_TX_MODE_WRITE_WITH_IMM_FOR_PEER, in which case they must be
/// non-zero (the addresses the peer's `cpu_roce_get_rx_ring_data_addr`
/// returned + `cpu_roce_get_rkey` returned, exchanged out-of-band).
cpu_roce_transceiver_t cpu_roce_create_transceiver(
    const char *device_name, int ib_port, unsigned tx_ibv_qp, size_t frame_size,
    size_t page_size, unsigned num_pages, const char *peer_ip, int forward,
    int rx_only, int tx_only, int unified, cpu_roce_tx_mode_t tx_mode,
    uint64_t peer_rx_base_addr, uint32_t peer_rx_rkey);

/// Destroy the transceiver.  Idempotent.  Implicitly calls cpu_roce_close
/// if the transceiver is still running.
void cpu_roce_destroy_transceiver(cpu_roce_transceiver_t handle);

/// Open the `ibv` device, build PD/CQs/QP/MRs, allocate rings, pre-post `recv`
/// WQEs, transition to RTS.  Returns 1 on success, 0 on failure.  Uses the
/// peer QP/ip/rkey fixed at construction (Phase 1 path).
int cpu_roce_start(cpu_roce_transceiver_t handle);

/// Phase 2 split bring-up for the bidirectional RDMA handshake.
/// cpu_roce_setup(): build everything up to QP INIT (peer not needed yet);
/// after success, cpu_roce_get_qp_number()/cpu_roce_get_rkey() are valid to
/// exchange with the peer.  Returns 1 on success, 0 on failure.
int cpu_roce_setup(cpu_roce_transceiver_t handle);

/// cpu_roce_connect(): transition the setup() QP to RTR/RTS using the now-
/// known peer QP number, peer IPv4 (RoCEv2 GID derived from it), and (for
/// tx_mode=WRITE_WITH_IMM_FOR_PEER) the peer's rx_data rkey.  Returns 1 on
/// success, 0 on failure.
int cpu_roce_connect(cpu_roce_transceiver_t handle, unsigned peer_qp,
                     const char *peer_ip, uint32_t peer_rx_rkey);

/// Signal exit, join I/O threads, release `ibv`/memory resources.
/// Idempotent.  Safe to call from any thread.
void cpu_roce_close(cpu_roce_transceiver_t handle);

/// Spawn the configured I/O thread(s) and block until cpu_roce_close().
/// Idempotent.  Throws on start() not having succeeded (returned to the
/// C ABI as a return-without-blocking).
void cpu_roce_blocking_monitor(cpu_roce_transceiver_t handle);

/// Per-slot dispatch callback for unified mode.  Returns number of bytes
/// written to tx_slot (0 = drop without sending).  See
/// CpuRoceTransceiver::UnifiedDispatchFn in roce_transceiver.hpp.
typedef size_t (*cpu_roce_unified_dispatch_fn_t)(void *context,
                                                 const void *rx_slot,
                                                 void *tx_slot,
                                                 size_t slot_size);

/// Install the unified-mode dispatch callback + opaque context.  No-op
/// when the transceiver was not constructed with unified=1.  Caller
/// retains ownership of `context`; it must outlive
/// cpu_roce_blocking_monitor.
void cpu_roce_set_unified_dispatch(cpu_roce_transceiver_t handle,
                                   cpu_roce_unified_dispatch_fn_t fn,
                                   void *context);

/// Optionally pin the local source GID to a specific IPv4 address (must be
/// called before cpu_roce_setup/cpu_roce_start).  Unset/null/empty => first
/// IPv4-mapped RoCEv2 GID on the port (correct on single-IP ports); set it on a
/// multi-IP port so the source GID matches the intended address.
void cpu_roce_set_local_ip(cpu_roce_transceiver_t handle, const char *local_ip);

//==============================================================================
// QP / rkey for rendezvous
//==============================================================================

/// Local QP number assigned by ibv_create_qp.  Pass to peer.
uint32_t cpu_roce_get_qp_number(cpu_roce_transceiver_t handle);

/// `rkey` of our rx_data MR.  Peer uses this to RDMA-write into our RX ring.
uint32_t cpu_roce_get_rkey(cpu_roce_transceiver_t handle);

/// IOVA base of the externally-visible rx_data buffer.  Peer addresses
/// our slots by remote_addr = cpu_roce_get_buffer_addr() + slot*stride.
/// Currently equals the virtual address of rx_data (plain ibv_reg_mr).
uint64_t cpu_roce_get_buffer_addr(cpu_roce_transceiver_t handle);

//==============================================================================
// Ring buffer accessors (mirroring hololink_wrapper.h's surface)
//==============================================================================

void *cpu_roce_get_rx_ring_data_addr(cpu_roce_transceiver_t handle);
uint64_t *cpu_roce_get_rx_ring_flag_addr(cpu_roce_transceiver_t handle);
void *cpu_roce_get_tx_ring_data_addr(cpu_roce_transceiver_t handle);
uint64_t *cpu_roce_get_tx_ring_flag_addr(cpu_roce_transceiver_t handle);

size_t cpu_roce_get_page_size(cpu_roce_transceiver_t handle);
unsigned cpu_roce_get_num_pages(cpu_roce_transceiver_t handle);

#ifdef __cplusplus
}
#endif

#endif // CPU_ROCE_WRAPPER_H
