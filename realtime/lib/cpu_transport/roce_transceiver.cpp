/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// CpuRoceTransceiver implementation.  See roce_transceiver.hpp for the
// design overview and ring/wire protocol description.

#include "cudaq/realtime/cpu_transport/roce_transceiver.hpp"

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cudaq::realtime::bridge {

// Architecture-portable busy-wait hint.  ARM yield / x86 pause.
static inline void cpu_relax() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  asm volatile("pause" ::: "memory");
#else
  asm volatile("" ::: "memory");
#endif
}

namespace {

// ============================================================================
// Helpers
// ============================================================================

// Pre-posted recv WQE encoding: lower 32 bits = slot index, upper 32 bits =
// generation counter (so a stale completion that bypasses our re-arm
// handshake is detectable).  Matches the §4.1 pre-post plan: "wr_id =
// (slot << 16) | generation" was a sketch; we promote both to 32 bits for
// 4M slots * 4G generations of headroom.
constexpr std::uint64_t encode_wr_id(std::uint32_t slot,
                                     std::uint32_t generation) {
  return (static_cast<std::uint64_t>(generation) << 32) | slot;
}
constexpr std::uint32_t decode_slot(std::uint64_t wr_id) {
  return static_cast<std::uint32_t>(wr_id & 0xFFFFFFFFu);
}
[[maybe_unused]] constexpr std::uint32_t
decode_generation(std::uint64_t wr_id) {
  return static_cast<std::uint32_t>(wr_id >> 32);
}

// Allocate pinned host memory aligned to the host page size and mlock it
// so the NIC's DMA target doesn't get paged out from under us.  Throws on
// failure.  The buffer is also memset to zero so flag arrays start clean
// and data slots start in a predictable state.
void *allocate_pinned(std::size_t bytes, std::size_t alignment) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, bytes) != 0)
    throw std::runtime_error("CpuRoceTransceiver: posix_memalign failed");
  std::memset(ptr, 0, bytes);
  // Best-effort mlock; not fatal if it fails (e.g. RLIMIT_MEMLOCK too low).
  // The NIC DMA still works without mlock, just with possible page-fault
  // jitter on first touch.
  (void)mlock(ptr, bytes);
  return ptr;
}

} // namespace

// ============================================================================
// Pimpl
// ============================================================================

struct CpuRoceTransceiver::Impl {
  // -- configuration captured at construction --
  const char *ibv_name = nullptr;
  unsigned ibv_port = 0;
  unsigned tx_ibv_qp = 0;
  std::size_t cu_frame_size = 0;
  std::size_t cu_page_size = 0;
  unsigned pages = 0;
  const char *peer_ip = nullptr;
  bool forward = false;
  bool rx_only = false;
  bool tx_only = false;
  bool unified = false;
  CpuRoceTxMode tx_mode = CpuRoceTxMode::kSendForFpga;
  std::uint64_t peer_rx_base_addr = 0;
  std::uint32_t peer_rx_rkey = 0;

  // -- ibv resources populated by start() --
  ibv_context *ctx = nullptr;
  ibv_pd *pd = nullptr;
  ibv_cq *rq_cq = nullptr;
  ibv_cq *sq_cq = nullptr;
  ibv_qp *qp = nullptr;
  ibv_mr *rx_data_mr = nullptr;
  ibv_mr *tx_data_mr = nullptr;
  std::uint32_t qp_number = 0;
  std::uint32_t rkey = 0;
  std::uint32_t local_tx_lkey = 0;
  std::uint32_t gid_index = 0;

  // -- ring buffers (pinned host memory) --
  std::uint8_t *rx_data = nullptr;
  std::uint8_t *tx_data = nullptr;
  std::uint64_t *rx_flags = nullptr;
  std::uint64_t *tx_flags = nullptr;
  std::size_t stride_sz = 0;
  std::uint32_t stride_num = 0;

  // -- runtime state --
  bool started = false;
  bool monitor_running = false;
  std::atomic<int> exit_flag{0};

  // -- I/O threads spawned by blocking_monitor(), joined by close() --
  std::thread rx_thread;
  std::thread tx_thread;
  std::thread forward_thread;
  std::thread unified_thread;

  // -- unified-mode dispatch hook (set by set_unified_dispatch) --
  CpuRoceTransceiver::UnifiedDispatchFn unified_fn = nullptr;
  void *unified_ctx = nullptr;

  // -- start() helpers --
  bool open_ib_device();
  bool find_roce_v2_gid();
  bool allocate_rings();
  bool register_mrs();
  bool create_qp_and_cqs();
  bool transition_qp_to_init();
  bool post_initial_recv_wqes();
  bool transition_qp_to_rtr_rts();

  // -- I/O thread loops --
  // Normal-mode RX loop: poll RQ CQ, decode slot from wr_id, wait for the
  // consumer's release of the slot, publish rx_flag[slot] = data address,
  // re-post the recv WQE with bumped generation.  Exits when exit_flag is
  // set.  Mirrors the GPU rx_only kernel.
  void rx_loop();

  // Forward-mode loop: RX thread does double duty — on every CQE, instead
  // of publishing rx_flag, immediately re-sends the slot's bytes back to
  // the peer.  Skips the consumer handshake entirely.  Useful as a wire-
  // round-trip latency baseline (no dispatch overhead).  Mirrors the GPU
  // forward kernel.
  void forward_loop();

  // Normal-mode TX loop: walk slots round-robin, busy-poll tx_flag[slot]
  // for a non-zero "ready to send" sentinel (producer publishes the slot's
  // tx_data address), claim it by clearing the flag, then issue the wire
  // verb selected by tx_mode (Send for FPGA, Write-With-Imm for peer).
  // Mirrors the GPU tx_only / tx_only_bf kernel.
  void tx_loop();

  // Unified-mode loop: collapses RX + dispatch + TX into a single thread.
  // The lowest-latency Phase 1 configuration when the dispatch callback
  // is cheap enough to run on the polling thread (e.g. the increment
  // handler).  Skips both flag arrays entirely (the consumer/producer
  // handshake collapses because the thread IS the consumer/producer).
  // Mirrors the GPU unified_dispatch_kernel concept but on the CPU.
  void unified_loop();

  void release_resources() noexcept;
};

// ----------------------------------------------------------------------------
// start() helpers
// ----------------------------------------------------------------------------

bool CpuRoceTransceiver::Impl::open_ib_device() {
  // ibverbs has reentrancy concerns when the process forks; init once per
  // process.  Matches the pattern in HSB's RoceReceiver::start.
  static std::atomic<bool> ibv_fork_init_done{false};
  bool expected = false;
  if (ibv_fork_init_done.compare_exchange_strong(expected, true)) {
    if (ibv_fork_init() != 0) {
      std::fprintf(stderr, "CpuRoceTransceiver: ibv_fork_init failed\n");
      return false;
    }
  }

  int num_devices = 0;
  ibv_device **devs = ibv_get_device_list(&num_devices);
  if (!devs) {
    std::fprintf(stderr,
                 "CpuRoceTransceiver: ibv_get_device_list failed errno=%d\n",
                 errno);
    return false;
  }
  ibv_device *picked = nullptr;
  for (int i = 0; i < num_devices; ++i) {
    if (std::strcmp(ibv_get_device_name(devs[i]), ibv_name) == 0) {
      picked = devs[i];
      break;
    }
  }
  if (!picked) {
    std::fprintf(stderr,
                 "CpuRoceTransceiver: ibv device '%s' not found among %d\n",
                 ibv_name, num_devices);
    ibv_free_device_list(devs);
    return false;
  }
  ctx = ibv_open_device(picked);
  ibv_free_device_list(devs);
  if (!ctx) {
    std::fprintf(stderr, "CpuRoceTransceiver: ibv_open_device failed\n");
    return false;
  }
  pd = ibv_alloc_pd(ctx);
  if (!pd) {
    std::fprintf(stderr, "CpuRoceTransceiver: ibv_alloc_pd failed\n");
    return false;
  }
  return true;
}

bool CpuRoceTransceiver::Impl::find_roce_v2_gid() {
  // Scan GIDs on the chosen port for an IPv4-mapped RoCEv2 entry — the
  // GID format HSB uses (and therefore what the FPGA's RoCEv2 stack
  // expects).  The IPv4-mapped IPv6 wire format is:
  //   bytes 0-9  = 0
  //   bytes 10-11 = 0xFFFF
  //   bytes 12-15 = IPv4 address (network byte order)
  //
  // We compare on `entry.gid.raw[]` to stay endianness-independent — some
  // libibverbs versions byte-swap `interface_id` to host order despite
  // the `__be64` typedef, others leave the wire bytes as-is.
  //
  // The loop terminates on `r == ENODATA` (end of GID table) or on any
  // other ibv error.  Hard cap at 256 as a belt-and-braces safety
  // (gid_tbl_len is well below this on any real NIC).
  constexpr std::uint32_t kMaxGidIndex = 256;
  for (std::uint32_t i = 0; i < kMaxGidIndex; ++i) {
    ibv_gid_entry entry{};
    int r = ibv_query_gid_ex(ctx, ibv_port, i, &entry, 0);
    if (r == ENODATA)
      break; // walked past the last entry
    if (r != 0)
      break; // some other ibv error — give up
    if (entry.gid_type != IBV_GID_TYPE_ROCE_V2)
      continue;
    const std::uint8_t *raw = entry.gid.raw;
    bool prefix_zero = true;
    for (int b = 0; b < 10; ++b)
      if (raw[b] != 0) {
        prefix_zero = false;
        break;
      }
    if (!prefix_zero)
      continue;
    if (raw[10] != 0xff || raw[11] != 0xff)
      continue;
    gid_index = i;
    return true;
  }
  std::fprintf(stderr,
               "CpuRoceTransceiver: no IPv4-mapped RoCEv2 GID found on %s "
               "port %u (did you add the IPv4 address to the interface and "
               "wait for the GID to populate?)\n",
               ibv_name, ibv_port);
  return false;
}

bool CpuRoceTransceiver::Impl::allocate_rings() {
  const std::size_t page_sz = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  const std::size_t data_bytes =
      static_cast<std::size_t>(stride_num) * stride_sz;
  const std::size_t flags_bytes =
      static_cast<std::size_t>(stride_num) * sizeof(std::uint64_t);

  // Data buffers: pinned at page granularity (NIC DMA target).
  rx_data = static_cast<std::uint8_t *>(allocate_pinned(data_bytes, page_sz));
  tx_data = static_cast<std::uint8_t *>(allocate_pinned(data_bytes, page_sz));

  // Flag arrays: cache-line-aligned to avoid false sharing between adjacent
  // slots.  Not NIC-DMA'd, so no MR registration; just CPU-shared state.
  rx_flags = static_cast<std::uint64_t *>(allocate_pinned(flags_bytes, 64));
  tx_flags = static_cast<std::uint64_t *>(allocate_pinned(flags_bytes, 64));
  return true;
}

bool CpuRoceTransceiver::Impl::register_mrs() {
  const std::size_t data_bytes =
      static_cast<std::size_t>(stride_num) * stride_sz;

  // RX ring is RDMA-write target: needs LOCAL_WRITE + REMOTE_WRITE.  rkey is
  // exported to the peer (FPGA via HSB control plane, or peer transceiver
  // via out-of-band rendezvous).
  //
  // Register with iova=0 so the peer addresses our slots by offset alone:
  //   remote_addr = 0 + slot*stride_sz
  // Why iova=0 and not iova=vaddr?  The HSB library enforces a hard limit
  // PAGES(highest_address) <= UINT32_MAX where PAGES = >> 7, so the
  // highest addressed byte must be below 2^39 = 512 GB.  Modern Linux on
  // aarch64 returns virtual addresses in the 48-bit range, well past
  // 512 GB, so vaddr-as-iova would trip the HSB limit at runtime in the
  // playback tool.
  //
  // CRITICAL consequence: on this mlx5/libibverbs stack, local SGEs that
  // reference this MR via its lkey must use IOVA-based addresses
  // (0 + slot*stride_sz), NOT virtual addresses (rx_data + slot*stride_sz).
  // The NIC validates the SGE addr against the iova range [0, data_bytes)
  // for both local reads (Send source) and local writes (Recv target).
  // post_initial_recv_wqes(), forward_loop, and any other code that
  // builds an SGE referencing rx_data_mr->lkey must follow this rule.
  rx_data_mr =
      ibv_reg_mr_iova(pd, rx_data, data_bytes, /*iova=*/0,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!rx_data_mr) {
    std::fprintf(
        stderr,
        "CpuRoceTransceiver: ibv_reg_mr_iova(rx_data) failed errno=%d\n",
        errno);
    return false;
  }
  rkey = rx_data_mr->rkey;

  // TX ring is local-only (we read from it to populate Send / Write
  // payloads); peer never addresses it.  Plain ibv_reg_mr (iova=vaddr) is
  // fine — SGEs from tx_loop reference tx_data_mr->lkey with vaddr-based
  // addresses, and there's no IOVA-vs-vaddr conflict because tx_data isn't
  // remotely addressable.
  tx_data_mr = ibv_reg_mr(pd, tx_data, data_bytes, IBV_ACCESS_LOCAL_WRITE);
  if (!tx_data_mr) {
    std::fprintf(stderr,
                 "CpuRoceTransceiver: ibv_reg_mr(tx_data) failed errno=%d\n",
                 errno);
    return false;
  }
  local_tx_lkey = tx_data_mr->lkey;
  return true;
}

bool CpuRoceTransceiver::Impl::create_qp_and_cqs() {
  // Busy-poll model: no completion channel (we never block on poll_cq).
  // CQs sized to stride_num so we never lose completions even at full
  // queue depth.
  const int cqe_count = static_cast<int>(stride_num);
  rq_cq = ibv_create_cq(ctx, cqe_count, nullptr, nullptr, 0);
  sq_cq = ibv_create_cq(ctx, cqe_count, nullptr, nullptr, 0);
  if (!rq_cq || !sq_cq) {
    std::fprintf(stderr, "CpuRoceTransceiver: ibv_create_cq failed errno=%d\n",
                 errno);
    return false;
  }

  // IBV_QPT_UC: unreliable connected.  Supports RDMA Write (incl.
  // Write-With-Imm) but not Read/Atomics.  Matches HSB's RoceReceiver and
  // GpuRoceTransceiver choice; the FPGA's HSB IP speaks the same QP type.
  ibv_qp_init_attr init{};
  init.send_cq = sq_cq;
  init.recv_cq = rq_cq;
  init.cap.max_send_wr = stride_num;
  init.cap.max_recv_wr = stride_num;
  init.cap.max_send_sge = 1;
  init.cap.max_recv_sge = 1;
  init.qp_type = IBV_QPT_UC;
  init.sq_sig_all = 0; // we'll opt slots into completions on a schedule

  qp = ibv_create_qp(pd, &init);
  if (!qp) {
    std::fprintf(stderr, "CpuRoceTransceiver: ibv_create_qp failed errno=%d\n",
                 errno);
    return false;
  }
  qp_number = qp->qp_num;
  return true;
}

bool CpuRoceTransceiver::Impl::transition_qp_to_init() {
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT;
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  a.pkey_index = 0;
  a.port_num = static_cast<std::uint8_t>(ibv_port);
  if (ibv_modify_qp(qp, &a,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_ACCESS_FLAGS) != 0) {
    std::fprintf(stderr,
                 "CpuRoceTransceiver: ibv_modify_qp(INIT) failed errno=%d\n",
                 errno);
    return false;
  }
  return true;
}

bool CpuRoceTransceiver::Impl::post_initial_recv_wqes() {
  // Pre-post stride_num receive WQEs, each pointing at one specific slot in
  // rx_data with wr_id encoding the slot.  When the peer's RDMA writes
  // complete, the CQE's wr_id tells us which slot was written into.
  //
  // SGE addr is IOVA-based (slot*stride_sz, offset from iova=0), NOT
  // vaddr-based.  Required because rx_data_mr is registered with iova=0
  // and the mlx5 NIC validates local SGE addrs against the iova range.
  // See register_mrs() for the rationale.
  for (std::uint32_t slot = 0; slot < stride_num; ++slot) {
    ibv_sge sge{};
    sge.addr = static_cast<std::uint64_t>(slot) * stride_sz;
    sge.length = static_cast<std::uint32_t>(stride_sz);
    sge.lkey = rx_data_mr->lkey;

    ibv_recv_wr wr{};
    wr.wr_id = encode_wr_id(slot, /*generation=*/0);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.next = nullptr;

    ibv_recv_wr *bad = nullptr;
    if (ibv_post_recv(qp, &wr, &bad) != 0) {
      std::fprintf(
          stderr, "CpuRoceTransceiver: ibv_post_recv slot=%u failed errno=%d\n",
          slot, errno);
      return false;
    }
  }
  return true;
}

bool CpuRoceTransceiver::Impl::transition_qp_to_rtr_rts() {
  // Convert peer IP to a RoCEv2 GID using the same scheme HSB uses (and
  // therefore the same scheme the FPGA's HSB IP expects):
  //   subnet_prefix = 0
  //   interface_id  = (peer_ip_in_network_byte_order << 32) | 0xFFFF0000
  in_addr_t peer = 0;
  if (inet_pton(AF_INET, peer_ip, &peer) != 1) {
    std::fprintf(stderr, "CpuRoceTransceiver: bad peer_ip '%s'\n", peer_ip);
    return false;
  }
  ibv_gid remote{};
  remote.global.subnet_prefix = 0;
  remote.global.interface_id =
      (static_cast<std::uint64_t>(peer) << 32) | 0xFFFF0000ull;

  // RST -> INIT -> RTR
  ibv_qp_attr rtr{};
  rtr.qp_state = IBV_QPS_RTR;
  rtr.path_mtu = IBV_MTU_4096;
  rtr.rq_psn = 0;
  rtr.dest_qp_num = tx_ibv_qp;
  rtr.ah_attr.grh.dgid = remote;
  rtr.ah_attr.grh.sgid_index = static_cast<std::uint8_t>(gid_index);
  rtr.ah_attr.grh.hop_limit = 0xFF;
  rtr.ah_attr.is_global = 1;
  rtr.ah_attr.port_num = static_cast<std::uint8_t>(ibv_port);
  // Retry a few times: HSB notes occasional ETIMEDOUT on this transition
  // that succeeds on retry.
  int r = -1;
  for (int attempt = 0; attempt < 5; ++attempt) {
    r = ibv_modify_qp(qp, &rtr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN);
    if (r == 0)
      break;
    usleep(200 * 1000);
  }
  if (r != 0) {
    std::fprintf(stderr,
                 "CpuRoceTransceiver: ibv_modify_qp(RTR) failed errno=%d\n", r);
    return false;
  }

  // RTR -> RTS so our TX path can post sends.  Even rx_only mode goes to
  // RTS for simplicity; we just won't post anything in that mode.  UC QPs
  // ignore most retry parameters.
  ibv_qp_attr rts{};
  rts.qp_state = IBV_QPS_RTS;
  rts.sq_psn = 0;
  if (ibv_modify_qp(qp, &rts, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
    std::fprintf(stderr,
                 "CpuRoceTransceiver: ibv_modify_qp(RTS) failed errno=%d\n",
                 errno);
    return false;
  }
  return true;
}

void CpuRoceTransceiver::Impl::rx_loop() {
  // Per-slot generation counter to defeat stale CQEs (paranoia: with UC QPs
  // and a single consumer, this is unlikely to matter, but keeping it
  // available is cheap).  Indexed [slot].
  std::vector<std::uint32_t> generation(stride_num, 1);

  // Polling loop: poll one CQE at a time for simplicity (we can batch in a
  // follow-up if profiling shows the CQE-read overhead is meaningful).
  ibv_wc wc{};
  while (exit_flag.load(std::memory_order_acquire) == 0) {
    int n = ibv_poll_cq(rq_cq, 1, &wc);
    if (n < 0) {
      std::fprintf(stderr,
                   "CpuRoceTransceiver: ibv_poll_cq(rq) failed errno=%d\n",
                   errno);
      break;
    }
    if (n == 0) {
      cpu_relax();
      continue;
    }
    if (wc.status != IBV_WC_SUCCESS) {
      std::fprintf(stderr,
                   "CpuRoceTransceiver: RX CQE status=%d (%s) wr_id=%lu\n",
                   wc.status, ibv_wc_status_str(wc.status),
                   static_cast<unsigned long>(wc.wr_id));
      continue;
    }
    const std::uint32_t slot = decode_slot(wc.wr_id);
    if (slot >= stride_num) {
      std::fprintf(stderr, "CpuRoceTransceiver: RX bad slot=%u (>= %u)\n", slot,
                   stride_num);
      continue;
    }

    // Slot publish handshake: wait for consumer to release the slot (set
    // rx_flag[slot] to 0).  This matches the GPU rx_only kernel's
    // back-pressure loop.  Yield to the scheduler periodically in case the
    // consumer is genuinely slow.
    auto &flag = *(volatile std::uint64_t *)&rx_flags[slot];
    while (__atomic_load_n(&flag, __ATOMIC_ACQUIRE) != 0) {
      if (exit_flag.load(std::memory_order_acquire) != 0)
        return;
      cpu_relax();
    }

    // Publish: writing the data address (non-zero) signals "fresh data".
    __atomic_store_n(
        &flag, reinterpret_cast<std::uint64_t>(rx_data + slot * stride_sz),
        __ATOMIC_RELEASE);

    // Re-post the recv WQE for this slot so future inbound writes have
    // somewhere to land.  IOVA-based addr (see post_initial_recv_wqes).
    generation[slot]++;
    ibv_sge sge{};
    sge.addr = static_cast<std::uint64_t>(slot) * stride_sz;
    sge.length = static_cast<std::uint32_t>(stride_sz);
    sge.lkey = rx_data_mr->lkey;

    ibv_recv_wr wr{};
    wr.wr_id = encode_wr_id(slot, generation[slot]);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.next = nullptr;
    ibv_recv_wr *bad = nullptr;
    if (ibv_post_recv(qp, &wr, &bad) != 0) {
      std::fprintf(
          stderr,
          "CpuRoceTransceiver: ibv_post_recv re-arm slot=%u failed errno=%d\n",
          slot, errno);
      // Continue rather than break; the next CQE may still be processable.
    }
  }
}

void CpuRoceTransceiver::Impl::forward_loop() {
  // Forward mode: every incoming write is immediately re-sent to the peer
  // from the same slot.  No rx_flag/tx_flag handshake involved (we do the
  // wire round-trip directly).  Mirrors the GPU forward kernel.
  std::vector<std::uint32_t> generation(stride_num, 1);

  // SQ drain via signal-every-N (same pattern as tx_loop).  Required
  // because the SQ slots are only freed when a signaled WQE completes
  // (and sweeps preceding unsignaled WQEs).  Without this, after
  // max_send_wr (= stride_num) sends, ibv_post_send returns ENOMEM.
  constexpr int kSignalEvery = 16;
  std::uint32_t since_signal = 0;
  ibv_wc swc[kSignalEvery]{};

  ibv_wc wc{};
  while (exit_flag.load(std::memory_order_acquire) == 0) {
    int n = ibv_poll_cq(rq_cq, 1, &wc);
    if (n < 0) {
      std::fprintf(
          stderr,
          "CpuRoceTransceiver(forward): ibv_poll_cq(rq) failed errno=%d\n",
          errno);
      break;
    }
    if (n == 0) {
      cpu_relax();
      continue;
    }
    if (wc.status != IBV_WC_SUCCESS) {
      std::fprintf(
          stderr,
          "CpuRoceTransceiver(forward): RX CQE status=%d (%s) wr_id=%lu\n",
          wc.status, ibv_wc_status_str(wc.status),
          static_cast<unsigned long>(wc.wr_id));
      continue;
    }
    const std::uint32_t slot = decode_slot(wc.wr_id);
    if (slot >= stride_num) {
      std::fprintf(stderr, "CpuRoceTransceiver(forward): bad slot=%u\n", slot);
      continue;
    }

    // Re-send the slot's bytes back to the peer.  IOVA-based addr (see
    // register_mrs / post_initial_recv_wqes).  Length uses cu_frame_size
    // for parity with tx_loop and the GPU bridge's forward kernel.
    ibv_sge send_sge{};
    send_sge.addr = static_cast<std::uint64_t>(slot) * stride_sz;
    send_sge.length = static_cast<std::uint32_t>(cu_frame_size);
    send_sge.lkey = rx_data_mr->lkey;

    ibv_send_wr swr{};
    swr.wr_id = wc.wr_id;
    swr.sg_list = &send_sge;
    swr.num_sge = 1;
    swr.next = nullptr;
    const bool signal_this = (++since_signal % kSignalEvery) == 0;
    swr.send_flags = signal_this ? IBV_SEND_SIGNALED : 0;
    if (tx_mode == CpuRoceTxMode::kWriteWithImmForPeer) {
      swr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      swr.wr.rdma.remote_addr = peer_rx_base_addr + slot * stride_sz;
      swr.wr.rdma.rkey = peer_rx_rkey;
      swr.imm_data = htonl(slot);
    } else {
      swr.opcode = IBV_WR_SEND;
    }
    ibv_send_wr *bad_send = nullptr;
    if (ibv_post_send(qp, &swr, &bad_send) != 0) {
      std::fprintf(stderr,
                   "CpuRoceTransceiver(forward): ibv_post_send slot=%u failed "
                   "errno=%d\n",
                   slot, errno);
    }
    if (signal_this) {
      int drained = ibv_poll_cq(sq_cq, kSignalEvery, swc);
      for (int k = 0; k < drained; ++k) {
        if (swc[k].status != IBV_WC_SUCCESS) {
          std::fprintf(
              stderr,
              "CpuRoceTransceiver(forward): SQ CQE wr_id=%lu status=%d (%s)\n",
              (unsigned long)swc[k].wr_id, swc[k].status,
              ibv_wc_status_str(swc[k].status));
        }
      }
    }

    // Re-post the recv WQE for this slot.  SEPARATE SGE from the send:
    // recv buffer must be sized to stride_sz to accept any future inbound
    // write (which might be larger than the original cu_frame_size).
    // IOVA-based addr (see register_mrs).
    generation[slot]++;
    ibv_sge recv_sge{};
    recv_sge.addr = static_cast<std::uint64_t>(slot) * stride_sz;
    recv_sge.length = static_cast<std::uint32_t>(stride_sz);
    recv_sge.lkey = rx_data_mr->lkey;
    ibv_recv_wr rwr{};
    rwr.wr_id = encode_wr_id(slot, generation[slot]);
    rwr.sg_list = &recv_sge;
    rwr.num_sge = 1;
    rwr.next = nullptr;
    ibv_recv_wr *bad_recv = nullptr;
    if (ibv_post_recv(qp, &rwr, &bad_recv) != 0) {
      std::fprintf(stderr,
                   "CpuRoceTransceiver(forward): ibv_post_recv re-arm slot=%u "
                   "failed errno=%d\n",
                   slot, errno);
    }
  }
}

void CpuRoceTransceiver::Impl::tx_loop() {
  // Producer publishes tx_flag[slot] = (address of tx_data + slot*stride).
  // We walk slots round-robin (matching the GPU tx_only kernel's
  // wqe_idx & mask indexing).  On non-zero flag: claim by clearing,
  // post the wire verb, advance.  Periodically drain the SQ CQ so we
  // never block on max_send_wr.
  const std::uint32_t mask = stride_num - 1;
  std::uint64_t wqe_idx = 0;

  // SQ CQ drain: signal every N WQEs (matches "signal-every-N" pattern
  // from §9 open questions; N=16 is a common starting point and amortizes
  // the CQE cost across batches without bloating the in-flight window).
  constexpr int kSignalEvery = 16;
  std::uint32_t since_signal = 0;
  ibv_wc swc[kSignalEvery]{};

  while (exit_flag.load(std::memory_order_acquire) == 0) {
    const std::uint32_t slot = static_cast<std::uint32_t>(wqe_idx & mask);
    auto &flag = *(volatile std::uint64_t *)&tx_flags[slot];
    const std::uint64_t addr = __atomic_load_n(&flag, __ATOMIC_ACQUIRE);
    if (addr == 0) {
      cpu_relax();
      continue;
    }
    // Claim: clear the flag.  Producer must wait for the flag to be 0
    // before publishing again into the same slot.
    __atomic_store_n(&flag, 0, __ATOMIC_RELEASE);

    // Build the send WQE.  Use the slot's data address as the SGE source;
    // the slot index is encoded in the IMM for Write-With-Imm mode (so the
    // peer can decode which slot the bytes belong to).  SGE length uses
    // cu_frame_size (the actual RPC frame size, RPCHeader + payload),
    // NOT stride_sz (the full slot), so we don't transmit unused slot
    // tail bytes.  Matches the GPU bridge's "frame_size" convention.
    ibv_sge sge{};
    sge.addr = addr;
    sge.length = static_cast<std::uint32_t>(cu_frame_size);
    sge.lkey = local_tx_lkey;

    ibv_send_wr swr{};
    swr.wr_id = slot;
    swr.sg_list = &sge;
    swr.num_sge = 1;
    swr.next = nullptr;
    // Signal completion every kSignalEvery WQEs so we can drain the SQ CQ
    // and free up max_send_wr capacity.  Other WQEs are unsignalled.
    const bool signal_this = (++since_signal % kSignalEvery) == 0;
    swr.send_flags = signal_this ? IBV_SEND_SIGNALED : 0;
    if (tx_mode == CpuRoceTxMode::kWriteWithImmForPeer) {
      // Phase 2 wire pattern.  remote_addr = peer_rx_base + slot*stride;
      // peer decodes the slot from imm_data (which is in network byte
      // order on the wire).
      swr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      swr.wr.rdma.remote_addr = peer_rx_base_addr + slot * stride_sz;
      swr.wr.rdma.rkey = peer_rx_rkey;
      swr.imm_data = htonl(slot);
    } else {
      // Phase 1 wire pattern (FPGA-facing).  FPGA's HSB IP receives Sends;
      // slot is identified on the peer side by which pre-posted recv WQE
      // the Send consumed (peer's wr_id encoding).
      swr.opcode = IBV_WR_SEND;
    }

    ibv_send_wr *bad = nullptr;
    if (ibv_post_send(qp, &swr, &bad) != 0) {
      std::fprintf(
          stderr, "CpuRoceTransceiver: ibv_post_send slot=%u failed errno=%d\n",
          slot, errno);
      // Restore the flag so the producer doesn't lose track of an in-flight
      // request that we failed to actually ship.
      __atomic_store_n(&flag, addr, __ATOMIC_RELEASE);
      continue;
    }

    // Drain SQ CQ opportunistically when we just signalled a WQE.  The
    // poll is non-blocking (returns 0 immediately if nothing's ready).
    if (signal_this) {
      int n = ibv_poll_cq(sq_cq, kSignalEvery, swc);
      (void)n; // we don't track per-send completion semantics; CQEs are
               // just consumed to free max_send_wr capacity.
    }
    wqe_idx++;
  }
}

void CpuRoceTransceiver::Impl::unified_loop() {
  // Single thread: RX CQE → dispatch → TX in one body, no flag handshake.
  std::vector<std::uint32_t> generation(stride_num, 1);
  // SQ-CQ drain on a signal-every-N schedule (same as tx_loop).
  constexpr int kSignalEvery = 16;
  std::uint32_t since_signal = 0;
  ibv_wc swc[kSignalEvery]{};

  ibv_wc wc{};
  while (exit_flag.load(std::memory_order_acquire) == 0) {
    int n = ibv_poll_cq(rq_cq, 1, &wc);
    if (n < 0) {
      std::fprintf(
          stderr,
          "CpuRoceTransceiver(unified): ibv_poll_cq(rq) failed errno=%d\n",
          errno);
      break;
    }
    if (n == 0) {
      cpu_relax();
      continue;
    }
    if (wc.status != IBV_WC_SUCCESS) {
      std::fprintf(stderr,
                   "CpuRoceTransceiver(unified): RX CQE status=%d (%s)\n",
                   wc.status, ibv_wc_status_str(wc.status));
      continue;
    }
    const std::uint32_t slot = decode_slot(wc.wr_id);
    if (slot >= stride_num)
      continue;

    // Dispatch: hand the slot to the user-supplied callback.  The callback
    // produces the response in tx_data[slot] and returns the response byte
    // count.  When no callback is set, treat as drop-and-rearm (lets the
    // tx_only / forward-style baseline benchmarks reuse this loop with a
    // null callback).
    const std::uint8_t *rx_slot = rx_data + slot * stride_sz;
    std::uint8_t *tx_slot = tx_data + slot * stride_sz;
    std::size_t resp_bytes = 0;
    if (unified_fn)
      resp_bytes = unified_fn(unified_ctx, rx_slot, tx_slot, stride_sz);

    // Send the response on the wire if the callback produced one.
    if (resp_bytes > 0) {
      ibv_sge sge{};
      sge.addr = reinterpret_cast<std::uintptr_t>(tx_slot);
      sge.length = static_cast<std::uint32_t>(resp_bytes);
      sge.lkey = local_tx_lkey;

      ibv_send_wr swr{};
      swr.wr_id = slot;
      swr.sg_list = &sge;
      swr.num_sge = 1;
      swr.next = nullptr;
      const bool signal_this = (++since_signal % kSignalEvery) == 0;
      swr.send_flags = signal_this ? IBV_SEND_SIGNALED : 0;
      if (tx_mode == CpuRoceTxMode::kWriteWithImmForPeer) {
        swr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        swr.wr.rdma.remote_addr = peer_rx_base_addr + slot * stride_sz;
        swr.wr.rdma.rkey = peer_rx_rkey;
        swr.imm_data = htonl(slot);
      } else {
        swr.opcode = IBV_WR_SEND;
      }
      ibv_send_wr *bad = nullptr;
      if (ibv_post_send(qp, &swr, &bad) != 0) {
        std::fprintf(stderr,
                     "CpuRoceTransceiver(unified): ibv_post_send slot=%u "
                     "failed errno=%d\n",
                     slot, errno);
      }
      if (signal_this) {
        int drained = ibv_poll_cq(sq_cq, kSignalEvery, swc);
        (void)drained;
      }
    }

    // Re-arm the recv WQE for this slot.
    generation[slot]++;
    ibv_sge rsge{};
    rsge.addr = static_cast<std::uint64_t>(slot) * stride_sz;
    rsge.length = static_cast<std::uint32_t>(stride_sz);
    rsge.lkey = rx_data_mr->lkey;

    ibv_recv_wr rwr{};
    rwr.wr_id = encode_wr_id(slot, generation[slot]);
    rwr.sg_list = &rsge;
    rwr.num_sge = 1;
    rwr.next = nullptr;
    ibv_recv_wr *rbad = nullptr;
    if (ibv_post_recv(qp, &rwr, &rbad) != 0) {
      std::fprintf(stderr,
                   "CpuRoceTransceiver(unified): ibv_post_recv re-arm slot=%u "
                   "failed errno=%d\n",
                   slot, errno);
    }
  }
}

void CpuRoceTransceiver::Impl::release_resources() noexcept {
  if (qp) {
    ibv_destroy_qp(qp);
    qp = nullptr;
  }
  if (rx_data_mr) {
    ibv_dereg_mr(rx_data_mr);
    rx_data_mr = nullptr;
  }
  if (tx_data_mr) {
    ibv_dereg_mr(tx_data_mr);
    tx_data_mr = nullptr;
  }
  if (rq_cq) {
    ibv_destroy_cq(rq_cq);
    rq_cq = nullptr;
  }
  if (sq_cq) {
    ibv_destroy_cq(sq_cq);
    sq_cq = nullptr;
  }
  if (pd) {
    ibv_dealloc_pd(pd);
    pd = nullptr;
  }
  if (ctx) {
    ibv_close_device(ctx);
    ctx = nullptr;
  }
  const std::size_t data_bytes =
      static_cast<std::size_t>(stride_num) * stride_sz;
  if (rx_data) {
    munlock(rx_data, data_bytes);
    free(rx_data);
    rx_data = nullptr;
  }
  if (tx_data) {
    munlock(tx_data, data_bytes);
    free(tx_data);
    tx_data = nullptr;
  }
  const std::size_t flags_bytes =
      static_cast<std::size_t>(stride_num) * sizeof(std::uint64_t);
  if (rx_flags) {
    munlock(rx_flags, flags_bytes);
    free(rx_flags);
    rx_flags = nullptr;
  }
  if (tx_flags) {
    munlock(tx_flags, flags_bytes);
    free(tx_flags);
    tx_flags = nullptr;
  }
  started = false;
}

// ============================================================================
// Public API
// ============================================================================

CpuRoceTransceiver::CpuRoceTransceiver(
    const char *ibv_name, unsigned ibv_port, unsigned tx_ibv_qp,
    std::size_t cu_frame_size, std::size_t cu_page_size, unsigned pages,
    const char *peer_ip, bool forward, bool rx_only, bool tx_only, bool unified,
    CpuRoceTxMode tx_mode, std::uint64_t peer_rx_base_addr,
    std::uint32_t peer_rx_rkey)
    : impl_(std::make_unique<Impl>()) {
  const int mode_count = (forward ? 1 : 0) + (rx_only ? 1 : 0) +
                         (tx_only ? 1 : 0) + (unified ? 1 : 0);
  if (mode_count > 1)
    throw std::invalid_argument(
        "CpuRoceTransceiver: forward / rx_only / tx_only / unified are "
        "mutually exclusive (at most one may be true)");
  if (tx_mode == CpuRoceTxMode::kWriteWithImmForPeer &&
      (peer_rx_base_addr == 0 || peer_rx_rkey == 0))
    throw std::invalid_argument(
        "CpuRoceTransceiver: tx_mode=kWriteWithImmForPeer requires "
        "peer_rx_base_addr and peer_rx_rkey to be non-zero");
  if (pages == 0 || (pages & (pages - 1)) != 0)
    throw std::invalid_argument(
        "CpuRoceTransceiver: pages must be a non-zero power of two");

  impl_->ibv_name = ibv_name;
  impl_->ibv_port = ibv_port;
  impl_->tx_ibv_qp = tx_ibv_qp;
  impl_->cu_frame_size = cu_frame_size;
  impl_->cu_page_size = cu_page_size;
  impl_->pages = pages;
  impl_->peer_ip = peer_ip;
  impl_->forward = forward;
  impl_->rx_only = rx_only;
  impl_->tx_only = tx_only;
  impl_->unified = unified;
  impl_->tx_mode = tx_mode;
  impl_->peer_rx_base_addr = peer_rx_base_addr;
  impl_->peer_rx_rkey = peer_rx_rkey;
  impl_->stride_sz = cu_page_size;
  impl_->stride_num = pages;
}

CpuRoceTransceiver::~CpuRoceTransceiver() {
  try {
    close();
  } catch (...) {
  }
}

bool CpuRoceTransceiver::start() {
  if (impl_->started)
    return true;
  // Sequence:  open device -> alloc + register rings -> create QP+CQs ->
  // INIT -> pre-post recv WQEs -> RTR -> RTS.  On any failure unwind
  // everything via release_resources so a second start() call won't trip
  // over half-built state.
  bool ok = impl_->open_ib_device() && impl_->find_roce_v2_gid() &&
            impl_->allocate_rings() && impl_->register_mrs() &&
            impl_->create_qp_and_cqs() && impl_->transition_qp_to_init();
  if (!ok) {
    impl_->release_resources();
    return false;
  }
  // Skip recv WQE pre-post in tx_only (no incoming traffic expected).
  if (!impl_->tx_only) {
    if (!impl_->post_initial_recv_wqes()) {
      impl_->release_resources();
      return false;
    }
  }
  if (!impl_->transition_qp_to_rtr_rts()) {
    impl_->release_resources();
    return false;
  }
  impl_->started = true;
  return true;
}

void CpuRoceTransceiver::blocking_monitor() {
  // Spawn the I/O thread(s) appropriate for the configured mode, then block
  // by joining them.  close() (called from another thread / signal handler /
  // dtor) sets exit_flag, which causes the worker loops to exit, which
  // causes our joins to return.
  if (!impl_->started)
    throw std::logic_error(
        "CpuRoceTransceiver::blocking_monitor: start() must succeed first");
  if (impl_->monitor_running)
    return; // idempotent: already running on another caller
  impl_->monitor_running = true;
  impl_->exit_flag.store(0, std::memory_order_release);

  // Mode selection (mutual exclusion was already enforced at construction).
  if (impl_->forward) {
    impl_->forward_thread =
        std::thread([impl = impl_.get()] { impl->forward_loop(); });
  } else if (impl_->unified) {
    impl_->unified_thread =
        std::thread([impl = impl_.get()] { impl->unified_loop(); });
  } else {
    // Normal three-thread layout (the §4.2 default).  tx_only / rx_only
    // skip one or the other thread; the consumer / dispatcher is on a
    // separate thread that this class doesn't own.
    if (!impl_->tx_only)
      impl_->rx_thread = std::thread([impl = impl_.get()] { impl->rx_loop(); });
    if (!impl_->rx_only)
      impl_->tx_thread = std::thread([impl = impl_.get()] { impl->tx_loop(); });
  }

  // Block until close() is called (or workers exit on their own due to a
  // fatal CQE error etc.).  Joining each present thread is enough; the
  // others are default-constructed and joinable() returns false on them.
  if (impl_->forward_thread.joinable())
    impl_->forward_thread.join();
  if (impl_->unified_thread.joinable())
    impl_->unified_thread.join();
  if (impl_->rx_thread.joinable())
    impl_->rx_thread.join();
  if (impl_->tx_thread.joinable())
    impl_->tx_thread.join();
  impl_->monitor_running = false;
}

void CpuRoceTransceiver::set_unified_dispatch(UnifiedDispatchFn fn,
                                              void *context) {
  if (!impl_)
    return;
  impl_->unified_fn = fn;
  impl_->unified_ctx = context;
}

void CpuRoceTransceiver::close() {
  if (!impl_)
    return;
  // Signal exit BEFORE joining; the worker loops poll exit_flag with
  // ACQUIRE semantics, so a RELEASE store here will be observed.
  impl_->exit_flag.store(1, std::memory_order_release);

  // Join workers if they're running and we're not the calling thread of
  // blocking_monitor (which will join them itself).  We join here as the
  // safety net for the destructor path where close() is the only thing
  // that runs cleanup.
  if (impl_->forward_thread.joinable())
    impl_->forward_thread.join();
  if (impl_->unified_thread.joinable())
    impl_->unified_thread.join();
  if (impl_->rx_thread.joinable())
    impl_->rx_thread.join();
  if (impl_->tx_thread.joinable())
    impl_->tx_thread.join();

  impl_->release_resources();
}

std::uint32_t CpuRoceTransceiver::get_qp_number() const {
  return impl_->qp_number;
}

std::uint32_t CpuRoceTransceiver::get_rkey() const { return impl_->rkey; }

std::uint64_t CpuRoceTransceiver::external_frame_memory() const {
  // rx_data_mr is registered with iova=0; peer addresses slots by offset
  // alone.  See register_mrs() for why iova=0 (HSB 32-bit page address
  // limit forces it).
  return 0;
}

std::uint8_t *CpuRoceTransceiver::get_rx_ring_data_addr() const {
  return impl_->rx_data;
}

std::size_t CpuRoceTransceiver::get_rx_ring_stride_sz() const {
  return impl_->stride_sz;
}

std::uint32_t CpuRoceTransceiver::get_rx_ring_stride_num() const {
  return impl_->stride_num;
}

std::uint64_t *CpuRoceTransceiver::get_rx_ring_flag_addr() const {
  return impl_->rx_flags;
}

std::uint8_t *CpuRoceTransceiver::get_tx_ring_data_addr() const {
  return impl_->tx_data;
}

std::size_t CpuRoceTransceiver::get_tx_ring_stride_sz() const {
  return impl_->stride_sz;
}

std::uint32_t CpuRoceTransceiver::get_tx_ring_stride_num() const {
  return impl_->stride_num;
}

std::uint64_t *CpuRoceTransceiver::get_tx_ring_flag_addr() const {
  return impl_->tx_flags;
}

} // namespace cudaq::realtime::bridge
