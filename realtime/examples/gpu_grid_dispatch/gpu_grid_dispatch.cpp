/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file gpu_grid_dispatch.cpp
/// @brief Serve an RPC with a whole resident GPU grid while the CPU owns the
/// NIC.
///
/// The request is the CI workload's: a 24-byte RPCHeader plus 8 payload bytes
/// for `rpc_increment`, answered with a 24-byte RPCResponse plus the 8 bytes
/// each incremented by one (with --request=ring, any payload that fits the
/// slot, kilobytes included).  The NIC side is CUDA-Q Realtime's host dispatch
/// path: the single-thread unified loop (receive, dispatch, send on one CPU
/// thread) over the pure-CPU `cpu_roce` bridge, with one HOST_CALL entry.
/// That handler does not do the work itself: it wakes a cooperative grid,
/// launched once at startup with one block on every SM, through a mailbox,
/// and waits for the grid's completion in pinned host memory.  grid_common.h
/// has the per-request protocol.
///
///     gpu_grid_dispatch [options] [-- <cpu_roce bridge options>]
///
/// `--notify=gdrcopy|pinned` (how the CPU writes the mailbox) and
/// `--poll=all|leader` (which GPU threads poll it) pick the handoff; `auto`
/// picks the fastest one measured for the platform.  `--request=inline|ring`
/// picks where the bytes travel: inline (default) copies 8 request bytes
/// through the mailbox and 8 reply bytes back through the done line; ring
/// maps the bridge's RX/TX rings for the GPU, so the grid reads the request
/// where the NIC wrote it and writes the reply where the NIC sends it from.
///
/// Setup order matters.  Everything that allocates or registers CUDA memory
/// (mailbox, done line, gdrcopy mapping, ring registration) and the
/// dispatcher are set up before the grid is
/// launched, because a resident cooperative grid blocks later allocations in
/// the process.  The READY line goes out before cudaq_bridge_connect(), as in
/// the test server (unittests/test_server/cudaq_realtime_test_server.cpp,
/// ORDERING RULE), whose setup sequence this follows step by step.

#include "grid_common.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#ifdef GPU_GRID_HAVE_GDRCOPY
#include <cuda.h>
#include <gdrapi.h>
#endif

using namespace gpu_grid_dispatch;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

namespace {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t err_ = (call);                                           \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "ERROR: %s: %s\n", #call,                           \
                   cudaGetErrorString(err_));                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CUDAQ_CHECK(call)                                                      \
  do {                                                                         \
    const cudaq_status_t st_ = (call);                                         \
    if (st_ != CUDAQ_OK) {                                                     \
      std::fprintf(stderr, "ERROR: %s failed (status %d)\n", #call,            \
                   static_cast<int>(st_));                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

constexpr std::uint32_t RPC_INCREMENT_ID =
    cudaq::realtime::fnv1a_hash("rpc_increment");
constexpr std::size_t GDR_PAGE = 65536; // gdrcopy pins whole 64 KiB GPU pages

struct Options {
  int gpu = 0;
  std::string notify = "auto";    // auto | gdrcopy | pinned
  std::string poll = "auto";      // auto | all | leader
  std::string request = "inline"; // inline | ring
  unsigned blocks = 0;            // 0 = SM count
  unsigned threads = 256;
  int timeout = 0; // seconds, 0 = until SIGINT
  std::string bridge;
  std::vector<char *> bridge_args; // after `--`, plus --unified
};

void usage(const char *prog) {
  std::printf(
      "Usage: %s [options] [-- <cpu_roce bridge options>]\n\n"
      "Serves rpc_increment with a resident cooperative GPU grid; the CPU\n"
      "owns the NIC through the cpu_roce bridge (unified host loop).\n\n"
      "  --gpu=N                        CUDA device [0]\n"
      "  --notify=auto|gdrcopy|pinned   CPU -> GPU mailbox [auto]\n"
      "  --poll=auto|all|leader         who polls the mailbox [auto]\n"
      "  --request=inline|ring          where request/reply bytes travel:\n"
      "                                 8 bytes via mailbox/done line, or\n"
      "                                 the grid reads the RX slot and\n"
      "                                 writes the TX slot [inline]\n"
      "  --blocks=N                     grid blocks [SM count]\n"
      "  --threads=N                    threads per block [256]\n"
      "  --timeout=S                    stop after S s, 0 = never [0]\n"
      "  --bridge=PATH                  bridge provider .so [cpu_roce]\n\n"
      "Bridge options after --, e.g. --device=mlx5_0 --qp_config=hsb_fpga\n"
      "--peer-ip=A --remote-qp=N --num-slots=64 --slot-size=4096\n"
      "--frame-size=32 (--unified is added if missing).\n",
      prog);
}

bool parse(int argc, char **argv, Options &o) {
  static char unified_flag[] = "--unified";
  bool have_unified = false;
  int i = 1;
  for (; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&](const char *key) -> const char * {
      const std::size_t n = std::strlen(key);
      return a.compare(0, n, key) == 0 ? argv[i] + n : nullptr;
    };
    try {
      if (a == "--") {
        ++i;
        break;
      } else if (a == "--help" || a == "-h") {
        usage(argv[0]);
        std::exit(0);
      } else if (const char *v = val("--gpu="))
        o.gpu = std::stoi(v);
      else if (const char *v = val("--notify="))
        o.notify = v;
      else if (const char *v = val("--poll="))
        o.poll = v;
      else if (const char *v = val("--request="))
        o.request = v;
      else if (const char *v = val("--blocks="))
        o.blocks = static_cast<unsigned>(std::stoul(v));
      else if (const char *v = val("--threads="))
        o.threads = static_cast<unsigned>(std::stoul(v));
      else if (const char *v = val("--timeout="))
        o.timeout = std::stoi(v);
      else if (const char *v = val("--bridge="))
        o.bridge = v;
      else {
        std::fprintf(stderr,
                     "ERROR: unknown option '%s'; bridge options go after "
                     "'--'\n",
                     argv[i]);
        return false;
      }
    } catch (const std::exception &) {
      std::fprintf(stderr, "ERROR: bad numeric value in '%s'\n", argv[i]);
      return false;
    }
  }
  for (; i < argc; ++i) {
    have_unified = have_unified || std::strcmp(argv[i], "--unified") == 0;
    o.bridge_args.push_back(argv[i]);
  }
  // This example always runs the unified host shape.
  if (!have_unified)
    o.bridge_args.push_back(unified_flag);

  const char *bad = nullptr;
  if (o.notify != "auto" && o.notify != "gdrcopy" && o.notify != "pinned")
    bad = "--notify must be auto, gdrcopy or pinned";
  else if (o.poll != "auto" && o.poll != "all" && o.poll != "leader")
    bad = "--poll must be auto, all or leader";
  else if (o.request != "inline" && o.request != "ring")
    bad = "--request must be inline or ring";
  else if (o.threads == 0 || o.threads > 1024)
    bad = "--threads must be 1..1024";
  if (bad)
    std::fprintf(stderr, "ERROR: %s\n", bad);
  return bad == nullptr;
}

/// The provider library: --bridge=, else the cpu_roce provider next to the
/// libcudaq-realtime this was built against, else its soname for dlopen to
/// resolve.
std::string provider_library(const Options &o) {
  if (!o.bridge.empty())
    return o.bridge;
  const std::string soname = "libcudaq-realtime-bridge-cpu-roce.so";
#ifdef GPU_GRID_BRIDGE_DIR
  const std::string path = std::string(GPU_GRID_BRIDGE_DIR) + "/" + soname;
  if (std::ifstream(path).good())
    return path;
#endif
  return soname;
}

// The handoff.  The HOST_CALL signature has no user-data pointer, so the
// handler's state is file scope: filled before the dispatcher starts, then
// touched only by the dispatcher's thread (`abort` aside).
struct Handoff {
  Mailbox *mailbox = nullptr; // CPU's view: pinned host, or the gdrcopy map
  bool gdr = false;
#ifdef GPU_GRID_HAVE_GDRCOPY
  gdr_t gdr_handle = nullptr;
  gdr_mh_t gdr_mh{};
  void *gdr_map = nullptr;
#endif
  Done *done = nullptr; // pinned host memory
  std::uint32_t request = REQUEST_INLINE;
  // REQUEST_RING: the bridge's rings (host view) and their geometry.
  const std::uint8_t *rx_ring = nullptr;
  const std::uint8_t *tx_ring = nullptr;
  std::uint32_t rx_stride = 0, tx_stride = 0, num_slots = 0;
  std::uint64_t seq = 0;
  std::uint64_t abandoned = 0;   // requests answered with an error at shutdown
  std::uint64_t slot_errors = 0; // ring: RX/TX slots that did not pair up
  std::uint64_t bad_replies = 0; // replies whose first bytes were not arg+1
  std::atomic<bool> abort{false};
};
Handoff g_handoff;

/// Ring mode: the slot index of a request, from where the dispatch loop put
/// its RX and TX slots.  The unified loop pairs slot N of both rings; a
/// mismatch is counted, reported once, and answered with an error.
bool ring_slot(const void *rx_slot, const void *tx_slot, std::uint32_t &slot) {
  Handoff &h = g_handoff;
  const auto rx = reinterpret_cast<std::uintptr_t>(rx_slot) -
                  reinterpret_cast<std::uintptr_t>(h.rx_ring);
  const auto tx = reinterpret_cast<std::uintptr_t>(tx_slot) -
                  reinterpret_cast<std::uintptr_t>(h.tx_ring);
  const std::uintptr_t rx_index = rx / h.rx_stride, tx_index = tx / h.tx_stride;
  // Unsigned wrap makes a pointer below the base huge, so one range test each.
  if (rx % h.rx_stride == 0 && tx % h.tx_stride == 0 &&
      rx_index < h.num_slots && rx_index == tx_index) {
    slot = static_cast<std::uint32_t>(rx_index);
    return true;
  }
  if (h.slot_errors++ == 0)
    std::fprintf(stderr,
                 "ERROR: request slots do not pair up: RX offset %zu (slot "
                 "%zu), TX offset %zu (slot %zu), %u slots; answering with an "
                 "error (reported once)\n",
                 static_cast<std::size_t>(rx),
                 static_cast<std::size_t>(rx_index),
                 static_cast<std::size_t>(tx),
                 static_cast<std::size_t>(tx_index), h.num_slots);
  return false;
}

/// Hand request `seq` to the grid: the mode's field first, then the sequence
/// with release semantics.
void publish_request(std::uint64_t seq, std::uint64_t payload,
                     std::uint64_t slot_len) {
  Handoff &h = g_handoff;
  const bool ring = h.request == REQUEST_RING;
  if (h.gdr) {
#ifdef GPU_GRID_HAVE_GDRCOPY
    // Two copies, each ending in a store fence: payload or slot_len, then
    // seq.
    if (ring)
      gdr_copy_to_mapping(h.gdr_mh, &h.mailbox->slot_len, &slot_len,
                          sizeof(slot_len));
    else
      gdr_copy_to_mapping(h.gdr_mh, &h.mailbox->payload, &payload,
                          sizeof(payload));
    gdr_copy_to_mapping(h.gdr_mh, &h.mailbox->seq, &seq, sizeof(seq));
#endif
  } else {
    if (ring)
      __atomic_store_n(&h.mailbox->slot_len, slot_len, __ATOMIC_RELAXED);
    else
      __atomic_store_n(&h.mailbox->payload, payload, __ATOMIC_RELAXED);
    __atomic_store_n(&h.mailbox->seq, seq, __ATOMIC_RELEASE);
  }
}

/// Spin until the grid has published `seq` (the TX slot's payload is written
/// by every block before the fan-in, the inline reply by the last block to
/// arrive, which then writes seq with a system-scope release; this acquire
/// makes those writes visible here).  False, and counted as
/// abandoned, when shutting down without an answer.
bool wait_for_grid(std::uint64_t seq) {
  unsigned spins = 0;
  while (__atomic_load_n(&g_handoff.done->seq, __ATOMIC_ACQUIRE) < seq) {
    if ((++spins & 1023u) == 0 &&
        g_handoff.abort.load(std::memory_order_relaxed)) {
      ++g_handoff.abandoned;
      return false;
    }
  }
  return true;
}

/// A cheap check of what the grid produced: the first (up to 8) result bytes
/// must be the argument bytes plus one.  One 8-byte read of each slot, so it
/// costs nothing measurable; the count is reported at shutdown.  It is what
/// ties the fan-in's release/acquire chain to an observable result.
void check_reply(const void *args, const void *result, std::uint32_t n) {
  if (n == 0)
    return;
  const std::uint32_t m = n < 8 ? n : 8;
  std::uint64_t a = 0, r = 0;
  std::memcpy(&a, args, m);
  std::memcpy(&r, result, m);
  const std::uint64_t mask = m == 8 ? ~std::uint64_t{0} : (1ull << (8 * m)) - 1;
  if (((increment_bytes(a) ^ r) & mask) != 0)
    ++g_handoff.bad_replies;
}

/// The CPU half of one request, run inline on the dispatcher's thread.  That
/// thread has no CUDA device selected and must not call the CUDA runtime;
/// nothing here does (gdr_copy_to_mapping is a plain CPU copy to a BAR
/// mapping).
void grid_handler(const void *rx_slot, void *tx_slot, std::size_t slot_size) {
  const auto *req = static_cast<const RPCHeader *>(rx_slot);
  auto *resp = static_cast<RPCResponse *>(tx_slot);
  Handoff &h = g_handoff;
  bool ok = true;
  std::uint32_t n = 0; // result bytes

  if (h.request == REQUEST_RING) {
    // The grid reads the arguments in the RX slot and writes the result
    // behind the response header in the TX slot; the CPU passes the slot
    // index and the clamped length `n` (the header was validated by the
    // dispatch loop) and touches no payload byte.
    std::uint32_t slot = 0;
    ok = ring_slot(rx_slot, tx_slot, slot);
    if (ok) {
      n = ring_payload_len(req->arg_len, h.rx_stride, h.tx_stride);
      const std::uint64_t seq = ++h.seq;
      publish_request(seq, 0, pack_slot_len(slot, n));
      ok = wait_for_grid(seq);
      if (ok)
        check_reply(req + 1, resp + 1, n);
    }
  } else {
    // The grid serves up to 8 argument bytes, the CI frame's payload.
    const std::size_t room =
        slot_size > sizeof(RPCResponse) ? slot_size - sizeof(RPCResponse) : 0;
    n = static_cast<std::uint32_t>(
        std::min<std::size_t>({req->arg_len, 8, room}));
    std::uint64_t payload = 0;
    std::memcpy(&payload, req + 1, n); // the GPU never reads the NIC ring
    const std::uint64_t seq = ++h.seq;
    publish_request(seq, payload, 0);
    ok = wait_for_grid(seq);
    if (ok) {
      const std::uint64_t reply =
          __atomic_load_n(&h.done->reply, __ATOMIC_RELAXED);
      std::memcpy(resp + 1, &reply, n);
      check_reply(req + 1, resp + 1, n);
    }
  }

  // The response header, echoing the request's identifiers.
  resp->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
  resp->status = ok ? 0 : -1;
  resp->result_len = ok ? n : 0;
  resp->request_id = req->request_id;
  resp->ptp_timestamp = req->ptp_timestamp;
}

#ifdef GPU_GRID_HAVE_GDRCOPY
/// Pin and map the device mailbox page for CPU writes.  On failure `why`
/// says what failed and nothing stays open.
bool map_with_gdrcopy(void *d_page, std::string &why) {
  Handoff &h = g_handoff;
  h.gdr_handle = gdr_open();
  if (!h.gdr_handle) {
    why = "gdr_open failed: is gdrdrv loaded?";
    return false;
  }
  // Keep the GPU's view of the page coherent with the BAR writes.
  const auto address = reinterpret_cast<CUdeviceptr>(d_page);
  int sync = 1;
  cuPointerSetAttribute(&sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, address);
  gdr_info_t info{};
  const int pin =
      gdr_pin_buffer(h.gdr_handle, address, GDR_PAGE, 0, 0, &h.gdr_mh);
  const int map =
      pin ? -1 : gdr_map(h.gdr_handle, h.gdr_mh, &h.gdr_map, GDR_PAGE);
  const int inf = map ? -1 : gdr_get_info(h.gdr_handle, h.gdr_mh, &info);
  if (pin || map || inf || address < info.va) {
    why = "gdrcopy pin/map of the mailbox failed (pin " + std::to_string(pin) +
          ", map " + std::to_string(map) + ", info " + std::to_string(inf) +
          ")";
    if (!map)
      gdr_unmap(h.gdr_handle, h.gdr_mh, h.gdr_map, GDR_PAGE);
    if (!pin)
      gdr_unpin_buffer(h.gdr_handle, h.gdr_mh);
    gdr_close(h.gdr_handle);
    h.gdr_handle = nullptr;
    return false;
  }
  h.mailbox = reinterpret_cast<Mailbox *>(
      static_cast<std::uint8_t *>(h.gdr_map) + (address - info.va));
  h.gdr = true;
  return true;
}

void unmap_gdrcopy() {
  Handoff &h = g_handoff;
  if (!h.gdr)
    return;
  gdr_unmap(h.gdr_handle, h.gdr_mh, h.gdr_map, GDR_PAGE);
  gdr_unpin_buffer(h.gdr_handle, h.gdr_mh);
  gdr_close(h.gdr_handle);
  h.gdr = false;
}
#endif

/// Ring mode: register one of the bridge's host rings with CUDA and return
/// the GPU's view of it.  The cpu_roce rings are page-aligned posix_memalign
/// allocations registered with the NIC (ibv_reg_mr) but not with CUDA;
/// cudaHostRegister maps those already-pinned pages into the GPU's address
/// space.  On GH200 and the integrated platforms the GPU reaches that memory
/// coherently at memory speed; on a PCIe GPU every ring access crosses PCIe,
/// which is correct but slow (the inline mode exists for that case).
bool map_ring_for_gpu(std::uint8_t *host, std::size_t bytes, const char *name,
                      int device, std::uint8_t **dev) {
  cudaError_t err = cudaHostRegister(
      host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (err == cudaSuccess) {
    void *d = nullptr;
    err = cudaHostGetDevicePointer(&d, host, 0);
    if (err == cudaSuccess) {
      *dev = static_cast<std::uint8_t *>(d);
      return true;
    }
    cudaHostUnregister(host);
  }
  int supported = 0;
  cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterSupported, device);
  std::fprintf(stderr,
               "ERROR: --request=ring: cannot map the bridge's %s ring "
               "(%zu B at %p) for GPU %d: %s.  The GPU must be able to map "
               "memory the NIC driver allocated (cudaHostRegister; "
               "cudaDevAttrHostRegisterSupported=%d on this platform).  Use "
               "--request=inline here.\n",
               name, bytes, static_cast<void *>(host), device,
               cudaGetErrorString(err), supported);
  return false;
}

volatile std::sig_atomic_t g_interrupted = 0;
void on_signal(int) { g_interrupted = 1; }

} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0); // stdout is a pipe under scripts
  Options opt;
  if (!parse(argc, argv, opt))
    return 1;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // [1] Device and platform.
  CUDA_CHECK(cudaSetDevice(opt.gpu));
  int sms = 0, integrated = 0, cooperative = 0;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, opt.gpu));
  CUDA_CHECK(
      cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, opt.gpu));
  CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                                    opt.gpu));
  if (!cooperative) {
    std::fprintf(stderr, "ERROR: GPU %d does not support cooperative launch\n",
                 opt.gpu);
    return 1;
  }
  const unsigned blocks = opt.blocks ? opt.blocks : static_cast<unsigned>(sms);

  // [2] The mailbox, and the handoff decision.  gdrcopy maps whole 64 KiB GPU
  //     pages, so the device mailbox gets a page of its own, aligned by hand
  //     (cudaMalloc guarantees only 256-byte alignment).
  void *d_mailbox_alloc = nullptr; // device memory, gdrcopy only
  void *h_mailbox_alloc = nullptr; // pinned host memory, pinned only
  const Mailbox *d_mailbox = nullptr;
  std::string why = "built without gdrcopy";
#ifdef GPU_GRID_HAVE_GDRCOPY
  if (opt.notify == "gdrcopy" || (opt.notify == "auto" && !integrated)) {
    CUDA_CHECK(cudaMalloc(&d_mailbox_alloc, 2 * GDR_PAGE));
    CUDA_CHECK(cudaMemset(d_mailbox_alloc, 0, 2 * GDR_PAGE));
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto raw = reinterpret_cast<std::uintptr_t>(d_mailbox_alloc);
    void *page = reinterpret_cast<void *>((raw + GDR_PAGE - 1) &
                                          ~std::uintptr_t(GDR_PAGE - 1));
    if (map_with_gdrcopy(page, why)) {
      d_mailbox = static_cast<const Mailbox *>(page);
    } else if (opt.notify == "gdrcopy") {
      std::fprintf(stderr, "ERROR: --notify=gdrcopy: %s\n", why.c_str());
      return 1;
    } else {
      CUDA_CHECK(cudaFree(d_mailbox_alloc));
      d_mailbox_alloc = nullptr;
    }
  }
#else
  if (opt.notify == "gdrcopy") {
    std::fprintf(stderr, "ERROR: --notify=gdrcopy: %s\n", why.c_str());
    return 1;
  }
#endif
  if (!g_handoff.gdr) {
    void *d = nullptr;
    CUDA_CHECK(
        cudaHostAlloc(&h_mailbox_alloc, sizeof(Mailbox), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d, h_mailbox_alloc, 0));
    std::memset(h_mailbox_alloc, 0, sizeof(Mailbox));
    g_handoff.mailbox = static_cast<Mailbox *>(h_mailbox_alloc);
    d_mailbox = static_cast<const Mailbox *>(d);
  }
  const std::uint32_t notify = g_handoff.gdr ? NOTIFY_GDRCOPY : NOTIFY_PINNED;
  const std::uint32_t request =
      opt.request == "ring" ? REQUEST_RING : REQUEST_INLINE;
  g_handoff.request = request;
  // Polling host memory from every SM is cheap only where host memory is the
  // GPU's own DRAM (integrated); a gdrcopy mailbox is device memory.
  std::uint32_t poll =
      (integrated || notify == NOTIFY_GDRCOPY) ? POLL_ALL : POLL_LEADER;
  if (opt.poll != "auto")
    poll = opt.poll == "leader" ? POLL_LEADER : POLL_ALL;
  std::printf("handoff: notify=%s (%s) poll=%s (%s) request=%s; %s GPU, %d "
              "SMs\n",
              notify_name(notify), opt.notify == "auto" ? "auto" : "flag",
              poll_name(poll), opt.poll == "auto" ? "auto" : "flag",
              request_name(request), integrated ? "integrated" : "discrete",
              sms);
  if (opt.notify == "auto" && !integrated && !g_handoff.gdr)
    std::printf("  gdrcopy not used: %s\n", why.c_str());

  // [3] The rest of the grid's memory: done line and shutdown flag (mapped
  //     pinned host memory) and the per-block SM ids (device memory).
  void *h_done = nullptr, *d_done = nullptr;
  void *h_shutdown = nullptr, *d_shutdown = nullptr;
  std::uint32_t *d_smid = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_done, sizeof(Done), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_done, h_done, 0));
  std::memset(h_done, 0, sizeof(Done));
  CUDA_CHECK(cudaHostAlloc(&h_shutdown, sizeof(int), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_shutdown, h_shutdown, 0));
  *static_cast<volatile int *>(h_shutdown) = 0;
  CUDA_CHECK(cudaMalloc(&d_smid, blocks * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMemset(d_smid, 0xff, blocks * sizeof(std::uint32_t)));
  g_handoff.done = static_cast<Done *>(h_done);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  unsigned shared_bytes = 0;
  int per_sm = 0;
  if (!configure_grid(notify, poll, request, opt.gpu, opt.threads, shared_bytes,
                      per_sm))
    return 1;
  if (per_sm < 1 || blocks > static_cast<unsigned>(per_sm * sms)) {
    std::fprintf(stderr,
                 "ERROR: %u blocks of %u threads (%u B shared) do not fit "
                 "co-resident: %d per SM x %d SMs\n",
                 blocks, opt.threads, shared_bytes, per_sm, sms);
    return 1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // [4] Bridge and dispatcher, as in the test server: load the provider with
  //     its own options, read its ring geometry and endpoint, create a HOST +
  //     UNIFIED dispatcher, wire the bridge, register the one HOST_CALL entry,
  //     hand it a host shutdown flag, start the loop.  No CUDA in any of it
  //     but the ring registration of --request=ring.
  const std::string library = provider_library(opt);
  cudaq_realtime_bridge_handle_t bridge = nullptr;
  if (cudaq_bridge_create_from_library(&bridge, library.c_str(),
                                       static_cast<int>(opt.bridge_args.size()),
                                       opt.bridge_args.data()) != CUDAQ_OK) {
    std::fprintf(stderr, "ERROR: cannot create a bridge from '%s'\n",
                 library.c_str());
    return 1;
  }
  std::uint32_t num_slots = 0, slot_size = 0;
  CUDAQ_CHECK(cudaq_bridge_get_ring_geometry(bridge, &num_slots, &slot_size));
  char endpoint[512] = {0};
  CUDAQ_CHECK(
      cudaq_bridge_get_endpoint_info(bridge, endpoint, sizeof(endpoint)));

  // --request=ring: map the bridge's RX and TX rings for the grid.  Reading
  // the CPU data plane is a plain accessor (it describes the provider's
  // rings and hooks, nothing more); the dispatcher still gets the bridge
  // itself below and asks for the same data plane on its own.  This
  // registers memory with CUDA, so it too comes before the grid launch.
  std::uint8_t *d_rx_ring = nullptr, *d_tx_ring = nullptr;
  std::uint8_t *h_rx_ring = nullptr, *h_tx_ring = nullptr;
  std::size_t rx_bytes = 0, tx_bytes = 0;
  if (request == REQUEST_RING) {
    cudaq_cpu_dataplane_t dp{};
    CUDAQ_CHECK(cudaq_bridge_get_cpu_dataplane(bridge, &dp));
    const cudaq_ringbuffer_t &r = dp.ring;
    if (!r.rx_data_host || !r.tx_data_host ||
        r.rx_stride_sz <= sizeof(RPCHeader) ||
        r.tx_stride_sz <= sizeof(RPCResponse) || num_slots == 0 ||
        r.rx_stride_sz % 4 != 0 || r.tx_stride_sz % 4 != 0 ||
        reinterpret_cast<std::uintptr_t>(r.rx_data_host) % 4 != 0 ||
        reinterpret_cast<std::uintptr_t>(r.tx_data_host) % 4 != 0) {
      std::fprintf(stderr,
                   "ERROR: --request=ring: unusable rings (%u slots, strides "
                   "%zu / %zu B; each must hold a 24-byte header and be "
                   "4-byte aligned: the grid moves 32-bit words)\n",
                   num_slots, r.rx_stride_sz, r.tx_stride_sz);
      return 1;
    }
    h_rx_ring = r.rx_data_host;
    h_tx_ring = r.tx_data_host;
    rx_bytes = static_cast<std::size_t>(num_slots) * r.rx_stride_sz;
    tx_bytes = static_cast<std::size_t>(num_slots) * r.tx_stride_sz;
    if (!map_ring_for_gpu(h_rx_ring, rx_bytes, "RX", opt.gpu, &d_rx_ring))
      return 1;
    if (!map_ring_for_gpu(h_tx_ring, tx_bytes, "TX", opt.gpu, &d_tx_ring)) {
      cudaHostUnregister(h_rx_ring);
      return 1;
    }
    g_handoff.rx_ring = h_rx_ring;
    g_handoff.tx_ring = h_tx_ring;
    g_handoff.rx_stride = static_cast<std::uint32_t>(r.rx_stride_sz);
    g_handoff.tx_stride = static_cast<std::uint32_t>(r.tx_stride_sz);
    g_handoff.num_slots = num_slots;
  }

  cudaq_dispatch_manager_t *manager = nullptr;
  cudaq_dispatcher_t *dispatcher = nullptr;
  CUDAQ_CHECK(cudaq_dispatch_manager_create(&manager));
  cudaq_dispatcher_config_t dcfg{};
  dcfg.kernel_type = CUDAQ_KERNEL_UNIFIED;
  dcfg.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  dcfg.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  CUDAQ_CHECK(cudaq_dispatcher_create(manager, &dcfg, &dispatcher));
  CUDAQ_CHECK(cudaq_dispatcher_set_bridge(dispatcher, bridge));

  static cudaq_function_entry_t entry; // read by the loop for its lifetime
  std::memset(&entry, 0, sizeof(entry));
  entry.handler.host_fn = &grid_handler;
  entry.function_id = RPC_INCREMENT_ID;
  entry.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  entry.schema.num_args = 1;
  entry.schema.num_results = 1;
  entry.schema.args[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
  entry.schema.results[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
  cudaq_function_table_t table{&entry, 1};
  CUDAQ_CHECK(cudaq_dispatcher_set_function_table(dispatcher, &table));

  static volatile int host_shutdown = 0;
  static std::uint64_t host_stats = 0;
  CUDAQ_CHECK(
      cudaq_dispatcher_set_control(dispatcher, &host_shutdown, &host_stats));
  CUDAQ_CHECK(cudaq_dispatcher_start(dispatcher));

  // [5] Only now the resident grid, then wait until every block is running.
  GridCtx ctx{};
  ctx.mailbox = d_mailbox;
  ctx.done = static_cast<Done *>(d_done);
  ctx.shutdown = static_cast<volatile int *>(d_shutdown);
  ctx.smid = d_smid;
  ctx.notify = notify;
  ctx.poll = poll;
  ctx.request = request;
  ctx.rx_ring = d_rx_ring;
  ctx.tx_ring = d_tx_ring;
  ctx.rx_stride = g_handoff.rx_stride;
  ctx.tx_stride = g_handoff.tx_stride;
  ctx.num_slots = g_handoff.num_slots;
  if (!launch_grid(ctx, blocks, opt.threads, shared_bytes, stream))
    return 1;
  const auto t0 = std::chrono::steady_clock::now();
  while (__atomic_load_n(&g_handoff.done->ready, __ATOMIC_ACQUIRE) == 0) {
    const cudaError_t q = cudaStreamQuery(stream);
    if (q != cudaErrorNotReady) {
      std::fprintf(stderr, "ERROR: the grid exited before it was ready: %s\n",
                   cudaGetErrorString(q));
      return 1;
    }
    if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
      std::fprintf(stderr, "ERROR: the grid did not become ready in 10 s\n");
      return 1;
    }
    std::this_thread::yield();
  }
  std::printf("grid: %u blocks x %u threads, %u B dynamic shared memory per "
              "block; bridge: %u slots of %u B",
              blocks, opt.threads, shared_bytes, num_slots, slot_size);
  if (request == REQUEST_RING)
    std::printf("; rings: rx %u x %u B, tx %u x %u B, mapped for the GPU",
                num_slots, g_handoff.rx_stride, num_slots, g_handoff.tx_stride);
  std::printf("\n");

  // [6] Announce BEFORE connecting (ORDERING RULE), then connect and launch.
  std::printf("CUDAQ_REALTIME_SERVER_READY %s notify=%s poll=%s request=%s\n",
              endpoint, notify_name(notify), poll_name(poll),
              request_name(request));
  CUDAQ_CHECK(cudaq_bridge_connect(bridge));
  CUDAQ_CHECK(cudaq_bridge_launch(bridge));

  // [7] Serve until SIGINT/SIGTERM or the timeout.
  const auto started = std::chrono::steady_clock::now();
  while (!g_interrupted) {
    if (opt.timeout > 0 && std::chrono::steady_clock::now() - started >=
                               std::chrono::seconds(opt.timeout)) {
      std::printf("timeout reached (%d s)\n", opt.timeout);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // [8] Shutdown: the dispatcher first (its loop calls into the bridge and
  //     into our handler), then the grid, then the transport and memory.
  g_handoff.abort.store(true, std::memory_order_relaxed);
  host_shutdown = 1;
  cudaq_dispatcher_stop(dispatcher);
  std::uint64_t served = 0;
  cudaq_dispatcher_get_processed(dispatcher, &served);

  __atomic_store_n(static_cast<int *>(h_shutdown), 1, __ATOMIC_RELEASE);
  const auto t1 = std::chrono::steady_clock::now();
  while (cudaStreamQuery(stream) == cudaErrorNotReady) {
    if (std::chrono::steady_clock::now() - t1 > std::chrono::seconds(10)) {
      std::fprintf(stderr, "ERROR: the grid did not exit in 10 s\n");
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::uint32_t> smid(blocks);
  CUDA_CHECK(cudaMemcpy(smid.data(), d_smid, blocks * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost));
  const std::set<std::uint32_t> distinct(smid.begin(), smid.end());
  std::printf("requests served: %llu (abandoned at shutdown: %llu",
              static_cast<unsigned long long>(served),
              static_cast<unsigned long long>(g_handoff.abandoned));
  if (request == REQUEST_RING)
    std::printf("; unpaired slots: %llu",
                static_cast<unsigned long long>(g_handoff.slot_errors));
  std::printf("; bad replies: %llu)\n",
              static_cast<unsigned long long>(g_handoff.bad_replies));
  std::printf("placement: %u blocks on %zu distinct SMs (of %d)\n", blocks,
              distinct.size(), sms);

  cudaq_dispatcher_destroy(dispatcher);
  cudaq_dispatch_manager_destroy(manager);
  // The rings belong to the transceiver: unmap them before it releases them.
  if (h_rx_ring)
    cudaHostUnregister(h_rx_ring);
  if (h_tx_ring)
    cudaHostUnregister(h_tx_ring);
  cudaq_bridge_disconnect(bridge);
  cudaq_bridge_destroy(bridge);
#ifdef GPU_GRID_HAVE_GDRCOPY
  unmap_gdrcopy();
#endif
  cudaStreamDestroy(stream);
  cudaFree(d_smid);
  cudaFreeHost(h_shutdown);
  cudaFreeHost(h_done);
  if (h_mailbox_alloc)
    cudaFreeHost(h_mailbox_alloc);
  if (d_mailbox_alloc)
    cudaFree(d_mailbox_alloc);
  return 0;
}
