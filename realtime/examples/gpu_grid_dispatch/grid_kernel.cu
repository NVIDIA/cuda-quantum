/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file grid_kernel.cu
/// @brief The resident cooperative grid of the gpu_grid_dispatch example, and
/// the host wrappers that configure and launch it.
///
/// The grid is launched once, one block per SM, and stays resident until the
/// shutdown flag is raised.  Per request it waits for the mailbox (every
/// block, or block 0 alone followed by a broadcast in device memory that the
/// other blocks spin on), runs the rpc_increment body, and fans in: thread 0
/// of every block adds one to an arrival counter in device memory, and the
/// block that arrives last publishes the sequence in the done line.  There is
/// no grid-wide barrier on the request path.  The body is
///   inline  the last block to arrive increments the 8 mailbox payload bytes
///           and stores the reply into the done line just before seq; every
///           other thread stands in for a real decoder's work and does
///           nothing here;
///   ring    the whole grid, a grid-stride loop over the request's argument
///           bytes in the RX slot, one 32-bit word per thread per pass,
///           writing the reply into the TX slot.
///
/// The handoff (--notify, --poll, --request) is fixed before launch, so the
/// kernel is a template over the three choices and the host launches the one
/// instance that matches: no mode test on the request path.

#include "grid_common.h"

#include <cooperative_groups.h>
#include <cstddef>
#include <cstdio>

namespace cg = cooperative_groups;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

namespace gpu_grid_dispatch {
namespace {

//------------------------------------------------------------------------------
// Memory-order helpers.  The mailbox is polled at system scope when the CPU
// writes it in pinned host memory, and at GPU scope when it is device memory
// written through gdrcopy.  The done line is host memory: system scope.
//------------------------------------------------------------------------------
template <bool kSys>
__device__ __forceinline__ std::uint64_t load_acquire(const std::uint64_t *p) {
  std::uint64_t v;
  if constexpr (kSys)
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                 : "=l"(v)
                 : "l"(p)
                 : "memory");
  else
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
                 : "=l"(v)
                 : "l"(p)
                 : "memory");
  return v;
}

template <bool kSys>
__device__ __forceinline__ std::uint64_t load_relaxed(const std::uint64_t *p) {
  std::uint64_t v;
  if constexpr (kSys)
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                 : "=l"(v)
                 : "l"(p)
                 : "memory");
  else
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
                 : "=l"(v)
                 : "l"(p)
                 : "memory");
  return v;
}

/// Ring loads.  The RX ring is host memory the bridge writes behind the GPU's
/// back, and a slot is rewritten every `num_slots` requests, so every read of
/// it is a strong system-scope load: it is never answered by a stale L1/L2
/// copy of an earlier request.  The work loop reads 32-bit words; the byte
/// load serves a trailing partial word only.
__device__ __forceinline__ std::uint32_t load_sys_u32(const std::uint32_t *p) {
  std::uint32_t v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ std::uint8_t load_sys_u8(const std::uint8_t *p) {
  std::uint32_t v; // ld.u8 zero-extends into a 32-bit register
  asm volatile("ld.relaxed.sys.global.u8 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return static_cast<std::uint8_t>(v);
}

__device__ __forceinline__ void store_release_sys(std::uint64_t *p,
                                                  std::uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v)
               : "memory");
}

__device__ __forceinline__ void store_release_gpu(std::uint64_t *p,
                                                  std::uint64_t v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v)
               : "memory");
}

/// Fan-in arrival: add one at GPU scope with acquire-release semantics and
/// return the old count.  Release: this block's work precedes the arrival.
/// Acquire: the last arriver sees every earlier arriver's work.
__device__ __forceinline__ unsigned long long
arrive_acq_rel(unsigned long long *counter) {
  unsigned long long old;
  asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;"
               : "=l"(old)
               : "l"(counter)
               : "memory");
  return old;
}

__device__ __forceinline__ std::uint32_t current_smid() {
  std::uint32_t id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

/// One request as the grid sees it: the mailbox fields.
struct Req {
  std::uint64_t seq;      ///< 0: shut down
  std::uint64_t payload;  ///< inline mode
  std::uint64_t slot_len; ///< ring mode: pack_slot_len(slot, len)
};

/// POLL_LEADER: block 0 thread 0 copies the request here and publishes `seq`
/// with a GPU-scope release store; thread 0 of every other block spins on
/// `seq` with GPU-scope acquire loads.  The line is rewritten only once the
/// leader has seen the next request in the mailbox, which the CPU publishes
/// only after the previous request's done.seq, i.e. after every block has
/// arrived and therefore read this line.
struct alignas(128) Broadcast {
  std::uint64_t seq;
  std::uint64_t payload;
  std::uint64_t slot_len;
};
__device__ Broadcast g_broadcast;

/// Broadcast `seq` value that tells the other blocks to exit.  Larger than
/// any real sequence, so the `seq > last` spin exits on it too.
constexpr std::uint64_t kShutdown = ~std::uint64_t{0};

/// Fan-in counter: thread 0 of every block adds one once its block's work is
/// done.  It only ever counts up and is never reset: every block knows the
/// count the current request completes at (gridDim.x, 2 gridDim.x, ...), so
/// the block whose add reaches it is the last arriver.
__device__ unsigned long long g_arrived;

/// Spin until the mailbox holds a sequence newer than `last`; return it and
/// the field the request mode uses (only that one is read: in pinned mode
/// every load crosses the bus).  Returns seq 0 once the shutdown flag is
/// raised, which is read only every 1024 polls: it is host memory, and the
/// mailbox poll is the hot loop.
template <bool kSys, bool kRing>
__device__ Req wait_for_request(const GridCtx &ctx, std::uint64_t last) {
  unsigned spins = 0;
  while (true) {
    std::uint64_t seq = load_acquire<kSys>(&ctx.mailbox->seq);
    if (seq <= last && (++spins & 1023u) == 0 && *ctx.shutdown) {
      // One last look, so a request published just before the flag is still
      // served by every block rather than by some of them.
      seq = load_acquire<kSys>(&ctx.mailbox->seq);
      if (seq <= last)
        return Req{0, 0, 0};
    }
    if (seq > last) {
      Req r{seq, 0, 0};
      if constexpr (kRing)
        r.slot_len = load_relaxed<kSys>(&ctx.mailbox->slot_len);
      else
        r.payload = load_relaxed<kSys>(&ctx.mailbox->payload);
      return r;
    }
  }
}

/// POLL_LEADER, blocks other than 0: spin on the broadcast line in device
/// memory until it carries a sequence newer than `last` (or kShutdown).
template <bool kRing>
__device__ Req wait_for_broadcast(std::uint64_t last) {
  std::uint64_t seq;
  do {
    seq = load_acquire<false>(&g_broadcast.seq);
  } while (seq <= last);
  if (seq == kShutdown)
    return Req{0, 0, 0};
  Req r{seq, 0, 0};
  if constexpr (kRing)
    r.slot_len = load_relaxed<false>(&g_broadcast.slot_len);
  else
    r.payload = load_relaxed<false>(&g_broadcast.payload);
  return r;
}

/// The resident grid, one instance per handoff:
///   kSys     the mailbox is pinned host memory (NOTIFY_PINNED), polled at
///            system scope; otherwise device memory (gdrcopy), GPU scope;
///   kLeader  block 0 alone polls the mailbox and re-publishes (POLL_LEADER);
///   kRing    the request bytes live in the bridge's rings (REQUEST_RING).
/// The dynamic shared memory is never touched: it is only reserved, so that
/// two blocks cannot share an SM (see configure_grid).
template <bool kSys, bool kLeader, bool kRing>
__global__ void __launch_bounds__(1024) grid_worker(GridCtx ctx) {
  cg::grid_group grid = cg::this_grid();
  __shared__ std::uint64_t s_seq;
  __shared__ std::uint64_t s_payload;  // inline
  __shared__ std::uint64_t s_slot_len; // ring: pack_slot_len(slot, len)

  // Placement, then a ready signal once every block is resident.  This is the
  // only grid-wide barrier; the request loop below has none.
  if (threadIdx.x == 0)
    ctx.smid[blockIdx.x] = current_smid();
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0)
    store_release_sys(&ctx.done->ready, 1);

  std::uint64_t last = 0;
  // The arrival count that completes the current request (thread 0 only).
  unsigned long long target = gridDim.x;
  while (true) {
    // 1. Wait for the next request: {seq, payload | slot_len} into shared
    //    memory.  Two mailbox loads (seq, then the field); no ring header
    //    read: the CPU has already validated and clamped the request.
    if (threadIdx.x == 0) {
      Req r;
      if constexpr (!kLeader) {
        r = wait_for_request<kSys, kRing>(ctx, last);
      } else if (blockIdx.x == 0) {
        // Leader: the only poller of the mailbox.  Re-publish the request
        // in device memory for the other blocks: fields first, seq last.
        r = wait_for_request<kSys, kRing>(ctx, last);
        g_broadcast.payload = r.payload;
        g_broadcast.slot_len = r.slot_len;
        store_release_gpu(&g_broadcast.seq, r.seq == 0 ? kShutdown : r.seq);
      } else {
        r = wait_for_broadcast<kRing>(last);
      }
      s_seq = r.seq;
      s_payload = r.payload;
      s_slot_len = r.slot_len;
    }
    __syncthreads();
    const std::uint64_t seq = s_seq;
    if (seq == 0)
      return; // shutdown: uniform across the block

    // 2. The work: the inlined rpc_increment body.
    if constexpr (kRing) {
      //    Ring: the whole grid, one 32-bit argument word per thread per
      //    pass, read from the RX slot the bridge wrote, each byte plus one,
      //    and stored straight into the paired TX slot behind the response
      //    header.  Each thread i < words does one independent u32 load and
      //    one store; nothing else is loaded first.  The thread after the
      //    last full word handles a trailing partial word byte by byte.
      //    Plain stores: __syncthreads() and the block's release-arrival
      //    below order them before the last arriver's fence and release.
      //    An out-of-range slot serves nothing.
      static_assert(sizeof(RPCHeader) % 4 == 0 && sizeof(RPCResponse) % 4 == 0,
                    "payload must start on a 4-byte boundary");
      const std::uint64_t slot_len = s_slot_len;
      const std::size_t slot = slot_of(slot_len);
      const std::uint32_t len = slot < ctx.num_slots ? len_of(slot_len) : 0;
      const std::uint8_t *args =
          ctx.rx_ring + slot * ctx.rx_stride + sizeof(RPCHeader);
      std::uint8_t *result =
          ctx.tx_ring + slot * ctx.tx_stride + sizeof(RPCResponse);
      const std::uint32_t words = len >> 2, tail = len & 3u;
      const std::uint32_t stride = gridDim.x * blockDim.x;
      for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
           i < words + (tail ? 1u : 0u); i += stride) {
        if (i < words) {
          const auto *src = reinterpret_cast<const std::uint32_t *>(args) + i;
          auto *dst = reinterpret_cast<std::uint32_t *>(result) + i;
          *dst = increment_bytes(load_sys_u32(src));
        } else {
          for (std::uint32_t b = 4 * words; b < len; ++b)
            result[b] = static_cast<std::uint8_t>(load_sys_u8(args + b) + 1);
        }
      }
    }
    //    Inline: the 8-byte increment is done by the last arriver below;
    //    every thread of the grid stands in for a real decoder and does
    //    nothing here.

    // 3. Fan-in.  __syncthreads() orders the block's work before thread 0's
    //    arrival; the arrival is acq_rel at GPU scope, so the last arriver
    //    has observed every block's work (causality is transitive: each
    //    block's stores -> its release-add -> the last block's acquire of
    //    the same counter).
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned long long old = arrive_acq_rel(&g_arrived);
      if (old + 1 == target) {
        // 4. Last arriver: completion, in pinned host memory the CPU handler
        //    spins on.  The system-scope release store of done.seq publishes
        //    everything in causality order before it, the other blocks' work
        //    included, to the CPU's acquire load.
        if constexpr (kRing) {
          //    Ring: the TX-slot stores of every SM will be read by the
          //    bridge's DMA engine, not by a CPU thread that synchronizes
          //    with us, so a system-scope fence first: it makes them
          //    performed at system scope before the flag, rather than
          //    visible to an acquirer.
          __threadfence_system();
        } else {
          //    Inline: the reply travels in the done line itself and only
          //    the CPU reads it, so the release store alone orders the reply
          //    before seq.  No separate fence.
          *reinterpret_cast<volatile std::uint64_t *>(&ctx.done->reply) =
              increment_bytes(s_payload);
        }
        store_release_sys(&ctx.done->seq, seq);
      }
      target += gridDim.x;
    }
    last = seq;
  }
}

bool check(cudaError_t err, const char *what) {
  if (err == cudaSuccess)
    return true;
  std::fprintf(stderr, "ERROR: %s: %s\n", what, cudaGetErrorString(err));
  return false;
}

/// The instance that serves a handoff.
using Kernel = void (*)(GridCtx);
Kernel select_kernel(std::uint32_t notify, std::uint32_t poll,
                     std::uint32_t request) {
  static const Kernel table[8] = {
      grid_worker<false, false, false>, grid_worker<false, false, true>,
      grid_worker<false, true, false>,  grid_worker<false, true, true>,
      grid_worker<true, false, false>,  grid_worker<true, false, true>,
      grid_worker<true, true, false>,   grid_worker<true, true, true>,
  };
  const unsigned i = (notify == NOTIFY_PINNED ? 4u : 0u) |
                     (poll == POLL_LEADER ? 2u : 0u) |
                     (request == REQUEST_RING ? 1u : 0u);
  return table[i];
}

} // namespace

bool configure_grid(std::uint32_t notify, std::uint32_t poll,
                    std::uint32_t request, int device, unsigned threads,
                    unsigned &shared_bytes, int &blocks_per_sm) {
  const Kernel kernel = select_kernel(notify, poll, request);
  int per_sm = 0, per_block_optin = 0;
  if (!check(cudaDeviceGetAttribute(
                 &per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device),
             "shared memory per SM") ||
      !check(cudaDeviceGetAttribute(&per_block_optin,
                                    cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    device),
             "opt-in shared memory per block"))
    return false;
  // Just over half the SM: a second block no longer fits next to the first.
  // A real decoder is shared-memory heavy and places the same way.
  unsigned want = static_cast<unsigned>(per_sm) / 2 + 1024;
  if (want > static_cast<unsigned>(per_block_optin))
    want = static_cast<unsigned>(per_block_optin);
  if (!check(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(want)),
             "raising the grid's dynamic shared memory limit"))
    return false;
  if (!check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &blocks_per_sm, kernel, static_cast<int>(threads),
                 static_cast<std::size_t>(want)),
             "grid occupancy"))
    return false;
  shared_bytes = want;
  return true;
}

bool launch_grid(const GridCtx &ctx, unsigned blocks, unsigned threads,
                 unsigned shared_bytes, cudaStream_t stream) {
  const Kernel kernel = select_kernel(ctx.notify, ctx.poll, ctx.request);
  GridCtx arg = ctx;
  void *args[] = {&arg};
  return check(cudaLaunchCooperativeKernel(
                   reinterpret_cast<const void *>(kernel), dim3(blocks),
                   dim3(threads), args, static_cast<std::size_t>(shared_bytes),
                   stream),
               "launching the resident cooperative grid");
}

} // namespace gpu_grid_dispatch
