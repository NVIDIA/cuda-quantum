/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file grid_common.h
/// @brief What the host side (gpu_grid_dispatch.cpp) and the resident grid
/// (grid_kernel.cu) of the gpu_grid_dispatch example share: the CPU -> GPU
/// mailbox, the GPU -> CPU done line, the grid's launch context, and the
/// host-callable configure/launch wrappers.
///
/// One request travels like this (see gpu_grid_dispatch.cpp for the CPU half
/// and grid_kernel.cu for the GPU half):
///
///   CPU handler                            resident cooperative grid
///   -----------                            -------------------------
///   seq = ++counter
///   mailbox.payload | mailbox.slot_len
///   mailbox.seq     = seq    (release)  ->  poll mailbox.seq (acquire),
///                                           one relaxed load of the field
///                                           [leader: block 0 re-publishes
///                                            seq + that field in device
///                                            memory, the others spin on it]
///                                           the work (below)
///                                           fan-in: thread 0 of each block
///                                           adds 1 to a device counter
///   spin on done.seq >= seq  (acquire)  <-  last arriver: [done.reply] or
///   finish the TX slot                      [fence], done.seq (release, sys)
///
/// Where the request and the reply bytes live is `--request=`:
///
///   inline  The CPU copies the request's first 8 argument bytes into
///           `mailbox.payload`; the last block to arrive increments them and
///           stores the reply into `done.reply` just before `seq`, and the
///           CPU copies it into the TX slot.  The GPU never touches the
///           bridge's rings, which are plain host allocations registered with
///           the transport only.
///   ring    The CPU registers the bridge's RX and TX rings with CUDA at
///           startup.  Per request it passes the slot index and the clamped
///           argument length (from the header it has already validated),
///           packed in one 64-bit word.  The whole grid reads the argument
///           bytes where the bridge wrote them (RX slot N), a 32-bit word per
///           thread, and writes the reply payload straight into TX slot N;
///           the GPU reads no header, and
///           the CPU touches no payload bytes, only the 24-byte response
///           header.  This is the shape for kilobyte-sized requests.

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace gpu_grid_dispatch {

/// How the CPU writes the mailbox (`--notify=`).
enum Notify : std::uint32_t {
  NOTIFY_PINNED = 0,  ///< mapped pinned host memory, GPU polls at system scope
  NOTIFY_GDRCOPY = 1, ///< device memory written through a gdrcopy BAR mapping,
                      ///< GPU polls at GPU scope
};

/// Who polls the mailbox (`--poll=`).
enum Poll : std::uint32_t {
  POLL_ALL = 0,    ///< thread 0 of every block spins on the mailbox
  POLL_LEADER = 1, ///< block 0 thread 0 spins, then re-publishes the request
                   ///< in device memory, where the other blocks spin
};

/// Where the request and reply bytes travel (`--request=`).
enum Request : std::uint32_t {
  REQUEST_INLINE = 0, ///< 8 bytes in the mailbox, 8 bytes in the done line
  REQUEST_RING = 1,   ///< the grid reads the RX slot and writes the TX slot
};

inline const char *notify_name(std::uint32_t n) {
  return n == NOTIFY_GDRCOPY ? "gdrcopy" : "pinned";
}
inline const char *poll_name(std::uint32_t p) {
  return p == POLL_LEADER ? "leader" : "all";
}
inline const char *request_name(std::uint32_t r) {
  return r == REQUEST_RING ? "ring" : "inline";
}

/// CPU -> GPU handoff, one 128-byte line.  The mode's field (`payload` for
/// inline, `slot_len` for ring) is written first, `seq` last with release
/// semantics; the field is valid once a new `seq` has been observed.  Each
/// request gets the next `seq` (1-based, monotonic); 0 means "nothing yet".
/// The GPU reads `seq` and then that one 8-byte field, nothing else.
struct alignas(128) Mailbox {
  std::uint64_t seq;      ///< offset 0
  std::uint64_t payload;  ///< offset 8, inline: the first 8 argument bytes
  std::uint64_t slot_len; ///< offset 16, ring: pack_slot_len(slot, len)
};

/// Ring mode: the slot index (RX and TX alike) in the low 32 bits, the
/// argument bytes to serve (ring_payload_len) in the high 32 bits.
__host__ __device__ inline std::uint64_t pack_slot_len(std::uint32_t slot,
                                                       std::uint32_t len) {
  return (static_cast<std::uint64_t>(len) << 32) | slot;
}
__host__ __device__ inline std::uint32_t slot_of(std::uint64_t slot_len) {
  return static_cast<std::uint32_t>(slot_len);
}
__host__ __device__ inline std::uint32_t len_of(std::uint64_t slot_len) {
  return static_cast<std::uint32_t>(slot_len >> 32);
}

/// GPU -> CPU, in mapped pinned host memory.  The last block to arrive at the
/// fan-in writes `reply` (inline mode), then `seq` with a system-scope release
/// store.  `ready` is raised once,
/// after every block of the grid is resident.
struct alignas(128) Done {
  std::uint64_t seq;
  std::uint64_t reply; ///< inline: the 8 request bytes, each incremented by one
  std::uint64_t ready; ///< 1 once the grid is running
};

/// The resident grid's kernel argument (passed by value).  All pointers are
/// device views.
struct GridCtx {
  const Mailbox *mailbox; ///< pinned host (NOTIFY_PINNED) or device memory
  Done *done;             ///< mapped pinned host memory
  volatile int *shutdown; ///< mapped pinned host memory, polled rarely
  std::uint32_t *smid;    ///< device memory, one entry per block
  std::uint32_t notify;   ///< Notify
  std::uint32_t poll;     ///< Poll
  std::uint32_t request;  ///< Request
  // REQUEST_RING only: the bridge's rings, host memory mapped for the GPU.
  const std::uint8_t *rx_ring; ///< base of RX slot 0
  std::uint8_t *tx_ring;       ///< base of TX slot 0
  std::uint32_t rx_stride;     ///< bytes per RX slot
  std::uint32_t tx_stride;     ///< bytes per TX slot
  std::uint32_t num_slots;     ///< slots on each ring
};

/// The rpc_increment body on a whole word at once: each byte plus one, no
/// carry between bytes.  The grid computes the inline reply with the 64-bit
/// form and the ring reply with the 32-bit form; the CPU handler uses the
/// 64-bit form to check the replies it forwards.
__host__ __device__ inline std::uint64_t increment_bytes(std::uint64_t v) {
  constexpr std::uint64_t low7 = 0x7f7f7f7f7f7f7f7full;
  constexpr std::uint64_t ones = 0x0101010101010101ull;
  return ((v & low7) + ones) ^ (v & ~low7);
}
__host__ __device__ inline std::uint32_t increment_bytes(std::uint32_t v) {
  constexpr std::uint32_t low7 = 0x7f7f7f7fu;
  constexpr std::uint32_t ones = 0x01010101u;
  return ((v & low7) + ones) ^ (v & ~low7);
}

/// Ring mode: the number of argument bytes served for a request whose header
/// says `arg_len`, clamped to what follows the 24-byte header in the RX slot
/// and to the room after the 24-byte response header in the TX slot.  The
/// CPU handler computes it once and uses it both as the length it hands the
/// grid and as the `result_len` it reports.
__host__ __device__ inline std::uint32_t
ring_payload_len(std::uint32_t arg_len, std::uint32_t rx_stride,
                 std::uint32_t tx_stride) {
  constexpr std::uint32_t req = sizeof(cudaq::realtime::RPCHeader);
  constexpr std::uint32_t resp = sizeof(cudaq::realtime::RPCResponse);
  const std::uint32_t rx_room = rx_stride > req ? rx_stride - req : 0;
  const std::uint32_t tx_room = tx_stride > resp ? tx_stride - resp : 0;
  std::uint32_t n = arg_len;
  if (n > rx_room)
    n = rx_room;
  if (n > tx_room)
    n = tx_room;
  return n;
}

/// Reserve just over half an SM's shared memory per block (so the grid runs
/// one block per SM) and report the occupancy that results, for the kernel
/// instance that serves this handoff (the kernel is a template over
/// notify/poll/request; see grid_kernel.cu).  Call before launch_grid, with
/// the device selected and the same three values that go into GridCtx.
/// Returns false on a CUDA error.
bool configure_grid(std::uint32_t notify, std::uint32_t poll,
                    std::uint32_t request, int device, unsigned threads,
                    unsigned &shared_bytes, int &blocks_per_sm);

/// Launch the resident grid (the instance for ctx.notify/poll/request) with
/// cudaLaunchCooperativeKernel.  Every CUDA allocation of the process has to
/// be made before this: a resident cooperative grid blocks later allocations.
bool launch_grid(const GridCtx &ctx, unsigned blocks, unsigned threads,
                 unsigned shared_bytes, cudaStream_t stream);

} // namespace gpu_grid_dispatch
