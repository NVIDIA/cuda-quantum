/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// CpuRoceTransceiver — pure-CPU RDMA RoCEv2 transceiver that mirrors the
// hololink::operators::GpuRoceTransceiver RX/TX ring-buffer contract but
// drives the wire from libibverbs on a CPU thread instead of from a
// DOCA-GPU kernel. Phase 1 of the cpu_transport umbrella.
//
// Ring layout per ring (RX and TX symmetric):
//   addr        : base pointer to a contiguous pinned host buffer of
//                 stride_num * stride_sz bytes.
//   stride_sz   : size of one slot (one full payload).
//   stride_num  : number of slots; must be a power of two so the indexing
//                 pattern `wqe_idx & (stride_num - 1)` matches the existing
//                 GPU transceiver's design.
//   flag        : per-slot uint64; non-zero means "fresh data" (the value
//                 is the slot's data address) and zero means "consumer has
//                 released the slot".  Identical convention to
//                 GpuRoceTransceiver's gpu_rx_ring.flag / gpu_tx_ring.flag.
//
// Wire protocol per direction:
//
//   RX path (peer -> us): peer issues RDMA_WRITE_WITH_IMMEDIATE that lands
//   in our rx_data[slot]; recv WQE we pre-posted with wr_id = encoded slot
//   completes; RX thread publishes rx_flag[slot] = (uint64_t)(rx_data +
//   slot*stride_sz).  Consumer busy-polls rx_flag, then sets it back to 0
//   when done so the RX thread can re-post the recv WQE for that slot.
//   Identical for both Phase 1 (peer is FPGA) and Phase 2 (peer is another
//   CpuRoceTransceiver).
//
//   TX path (us -> peer): producer writes payload into tx_data[slot], then
//   publishes tx_flag[slot] = (uint64_t)(tx_data + slot*stride_sz).  TX
//   thread busy-polls tx_flag, claims slot by clearing it, and issues the
//   wire verb.  The wire verb depends on tx_mode (see CpuRoceTxMode below).
//
// Memory ordering: all rx_flag/tx_flag accesses use __ATOMIC_ACQUIRE on
// loads and __ATOMIC_RELEASE on stores (the CPU equivalent of the GPU
// transceiver's doca_gpu_dev_verbs_fence_* calls).

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cudaq::realtime::bridge {

/// Which RDMA wire verb this transceiver's TX path issues.  Names the verb the
/// transmitter uses, not a peer role: whoever sets the mode is the source.  The
/// RX path is identical for both modes (an inbound Send or Write-With-Imm is
/// consumed via a pre-posted recv WQE).
enum class CpuRoceTxMode {
  /// TX issues `IBV_WR_SEND`.  Use when the peer consumes Sends via pre-posted
  /// recv WQEs -- e.g. an FPGA's HSB IP (which can only receive Sends), or a
  /// CpuRoceTransceiver acting as the responder.  Default.
  kRdmaSend,

  /// TX issues `IBV_WR_RDMA_WRITE_WITH_IMM`, targeting the peer's `rx_data`
  /// base + slot offset using the peer's rkey, with the slot index carried in
  /// the IMM so the peer can decode it.  Use when the peer's memory is the
  /// write target -- e.g. an FPGA pushing syndromes, or a CpuRoceTransceiver
  /// acting as the requester.
  kRdmaWriteWithImm,
};

/// Pure-CPU RDMA RoCEv2 transceiver.  Mirrors the public API surface of
/// `hololink::operators::GpuRoceTransceiver` (HSB) so consumers like
/// `hololink_bridge_common.h` can swap the GPU transceiver for this CPU
/// one with minimal wiring changes.
///
/// Lifecycle:
///   1. construct  : capture configuration; no kernel/network activity yet.
///   2. start()    : open ibv device, build PD/CQs/QP/MRs, pre-post recv WQEs,
///                   transition QP RST -> INIT -> RTR -> RTS.
///   3. blocking_monitor() : spawn RX/TX threads (or one unified thread when
///                   unified=true) and block until close().
///   4. close()    : signal exit_flag, join threads, release ibv resources.
class CpuRoceTransceiver {
public:
  /// Constructor.  Parameters mirror GpuRoceTransceiver's where applicable
  /// and add `unified` + `tx_mode` + `peer_rx_*` for the Phase 1 / Phase 2
  /// modes.
  ///
  /// \param ibv_name              IB device name (e.g. "mlx5_0").
  /// \param ibv_port              IB port number (1-based; typically 1).
  /// \param tx_ibv_qp             Remote QP number to connect to (the
  ///                              FPGA's hardcoded QP for Phase 1, or the
  ///                              peer transceiver's QP for Phase 2).
  /// \param cu_frame_size         Logical frame size in bytes (one RPC
  ///                              request/response payload size).
  /// \param cu_page_size          Per-slot stride in bytes (must be >=
  ///                              cu_frame_size + header overhead).
  /// \param pages                 Number of slots per ring (power of two).
  /// \param peer_ip               Peer IPv4 address for the GID exchange.
  /// \param forward               When true, RX thread immediately re-sends
  ///                              the slot back out (loopback latency
  ///                              baseline); mutually exclusive with rx_only
  ///                              / tx_only / unified.
  /// \param rx_only               Skip the TX thread (RX-only mode).
  /// \param tx_only               Skip the RX thread and recv WQE pre-post.
  /// \param unified               When true, replace separate RX + dispatcher +
  ///                              TX threads with a single loop thread that
  ///                              does poll-CQ -> function-table lookup ->
  ///                              host_fn -> post-send -> re-arm recv WQE.
  ///                              CPU analogue of GpuRoceTransceiver's
  ///                              --unified GPU kernel.  Mutually exclusive
  ///                              with forward / rx_only / tx_only.
  /// \param tx_mode               Which wire verb the TX path issues (see
  ///                              CpuRoceTxMode).  Default is kRdmaSend.
  /// \param peer_rx_base_addr     Used when tx_mode == kRdmaWriteWithImm.
  ///                              The base virtual address of the peer's
  ///                              rx_data ring; our TX issues
  ///                              IBV_WR_RDMA_WRITE_WITH_IMM with
  ///                              remote_addr = peer_rx_base_addr +
  ///                              slot * cu_page_size.  May be 0 (peer rx_data
  ///                              MR is registered with iova=0).  Ignored for
  ///                              kRdmaSend.
  /// \param peer_rx_rkey          Used when tx_mode == kRdmaWriteWithImm.
  ///                              The rkey associated with the peer's
  ///                              rx_data MR; typically supplied later via
  ///                              connect().  Ignored for kRdmaSend.
  CpuRoceTransceiver(const char *ibv_name, unsigned ibv_port,
                     unsigned tx_ibv_qp, std::size_t cu_frame_size,
                     std::size_t cu_page_size, unsigned pages,
                     const char *peer_ip, bool forward, bool rx_only,
                     bool tx_only, bool unified,
                     CpuRoceTxMode tx_mode = CpuRoceTxMode::kRdmaSend,
                     std::uint64_t peer_rx_base_addr = 0,
                     std::uint32_t peer_rx_rkey = 0);

  ~CpuRoceTransceiver();

  // Non-copyable, non-movable (owns ibv resources + threads).
  CpuRoceTransceiver(const CpuRoceTransceiver &) = delete;
  CpuRoceTransceiver &operator=(const CpuRoceTransceiver &) = delete;
  CpuRoceTransceiver(CpuRoceTransceiver &&) = delete;
  CpuRoceTransceiver &operator=(CpuRoceTransceiver &&) = delete;

  /// Open the ibv device, build PD/CQs/QP/MRs, allocate pinned ring
  /// memory, pre-post recv WQEs, transition the QP to RTR (or RTS for
  /// full-duplex).  Returns true on success.
  ///
  /// Equivalent to setup() followed by connect() using the tx_ibv_qp /
  /// peer_ip / peer_rx_rkey supplied at construction.  Use this when the
  /// peer's QP number is already known at construction time (Phase 1
  /// bridge↔FPGA, where the FPGA's QP is fixed).
  bool start();

  /// Phase 2 two-step connection bring-up (when the peer's QP number is
  /// NOT known at construction — the circular dependency of a bidirectional
  /// RDMA handshake).  Split from start() so a transceiver can learn its own
  /// QP number / rkey, exchange them with the peer out-of-band, and only
  /// then connect.
  ///
  /// setup(): open device, build PD/CQs/QP/MRs, allocate rings, pre-post
  /// recv WQEs, transition the QP to INIT.  After this returns true,
  /// get_qp_number()/get_rkey() are valid and can be sent to the peer.
  /// Does NOT transition to RTR/RTS (no peer info needed yet).
  bool setup();

  /// connect(): transition the (already setup()) QP INIT -> RTR -> RTS using
  /// the now-known peer QP number, peer IPv4 (its RoCEv2 GID is derived from
  /// this), and — for tx_mode == kRdmaWriteWithImm — the peer's rx_data
  /// rkey.  Must be called after setup() and before blocking_monitor().
  /// Returns true on success.
  bool connect(unsigned peer_qp, const char *peer_ip,
               std::uint32_t peer_rx_rkey);

  /// Spawn the RX/TX (or single unified) thread(s).  Returns when close()
  /// is called.  Idempotent: a second call returns immediately if the
  /// monitor is already running, or if start() has not been called.
  void blocking_monitor();

  /// Signal exit, join threads, release ibv resources.  Safe to call from
  /// any thread.  Idempotent.
  void close();

  /// Per-slot dispatch callback for unified mode.  The transport library
  /// invokes this on every RX CQE — the callback reads the request from
  /// `rx_slot`, writes the response into `tx_slot`, and returns the number
  /// of bytes written (or 0 if the slot should be dropped without sending
  /// a response).  The callback runs on the transceiver's unified thread,
  /// so it must be quick (otherwise consider the three-thread layout).
  ///
  /// Kept transport-agnostic so the cpu_transport library has no
  /// dependency on cudaq_realtime.h / CUDA.  The bridge tool wraps the
  /// function-table lookup into a closure and supplies it here.
  using UnifiedDispatchFn = std::size_t (*)(void *context, const void *rx_slot,
                                            void *tx_slot,
                                            std::size_t slot_size);

  /// Provide the unified-mode dispatch callback + opaque context.  Must be
  /// called between start() and blocking_monitor() when constructed with
  /// unified=true.  The caller retains ownership of `context`; it must
  /// outlive blocking_monitor().  No-op when not in unified mode.
  void set_unified_dispatch(UnifiedDispatchFn fn, void *context);

  /// Optionally pin the local source GID to a specific IPv4 address.  Must be
  /// called before setup()/start().  When unset, the transceiver selects the
  /// first IPv4-mapped RoCEv2 GID on the port (correct on single-IP ports);
  /// set this on a multi-IP port so the source GID matches the intended
  /// address.  Empty/null clears it (first-GID behaviour).
  void set_local_ip(const char *local_ip);

  // ===========================================================================
  // Local QP / rkey for the HSB control plane (Phase 1) or out-of-band
  // rendezvous (Phase 2).  Populated by start().
  // ===========================================================================

  /// Local QP number assigned by ibv_create_qp.  Pass this to the FPGA
  /// (Phase 1, via hololink_channel->authenticate) or to the peer
  /// (Phase 2, via the daemon-prints-stdout rendezvous).
  std::uint32_t get_qp_number() const;

  /// rkey for our rx_data MR.  Peer needs this to RDMA-write into our
  /// receive ring.
  std::uint32_t get_rkey() const;

  /// Logical "base address" of the externally-visible rx_data buffer.
  /// Returns 0 by convention (we register the rx_data MR with
  /// ibv_reg_mr_iova so the peer addresses slots by offset alone, with
  /// the rkey identifying our MR).
  std::uint64_t external_frame_memory() const;

  // ===========================================================================
  // Ring accessors mirroring GpuRoceTransceiver's get_*_ring_* surface
  // exactly.  Consumers in hololink_bridge_common.h populate cudaq_ringbuffer_t
  // from these.
  // ===========================================================================

  std::uint8_t *get_rx_ring_data_addr() const;
  std::size_t get_rx_ring_stride_sz() const;
  std::uint32_t get_rx_ring_stride_num() const;
  std::uint64_t *get_rx_ring_flag_addr() const;

  std::uint8_t *get_tx_ring_data_addr() const;
  std::size_t get_tx_ring_stride_sz() const;
  std::uint32_t get_tx_ring_stride_num() const;
  std::uint64_t *get_tx_ring_flag_addr() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cudaq::realtime::bridge
