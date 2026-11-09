/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file cudaq_realtime_test_server.cpp
/// @brief Service half of the realtime two-process tests: a dispatcher serving
///        the `rpc_increment` handler over any bridge provider, on the host
///        (HOST_CALL) or on the GPU (DEVICE_CALL).
///
/// Also meant to be read.  This is the shortest complete path from "I have a
/// bridge provider" to "I am serving RPCs", so the numbered steps in `main` are
/// the reference: they are the whole API sequence, in order, with nothing
/// hidden behind a helper.  Copy them.
///
/// Transport-agnostic by construction.  Every transport is reached through the
/// same bridge vtable (bridge_interface.h), and each provider owns its own
/// bring-up -- udp binds a socket in create(), cpu_roce does the full TCP
/// rendezvous plus QP/rkey swap across create()/connect() -- so there is no
/// per-transport code here.
///
/// TWO COMMAND LINES, ONE ARGV
///
///     cudaq-realtime-test-server [server options] [-- bridge options]
///
/// `--` is the boundary: before it are this server's options, after it are the
/// bridge's, forwarded verbatim and never read here.  That is how `--port=`,
/// `--device=`, `--local-ip=`, `--qp_config=`, `--peer-ip=`, `--remote-qp=`,
/// `--num-slots=`, `--slot-size=`, `--unified` and `--pinned-rings` reach the
/// provider that understands them, without this file knowing any of them exist.
/// Sharing one namespace was tried: everything unrecognized was forwarded and
/// providers ignore what they do not know, so a misspelled option of OURS was
/// swallowed and served the default.  The split is what lets this server reject
/// its own typos.
///
/// THE SHAPE TAKES A TOKEN ON EACH SIDE OF THE `--`
///
///     --dispatch=ring                          (and no --unified)
///     --dispatch=unified  --  ... --unified
///
/// `--dispatch=` wires THIS SERVER: `ring` means the dispatcher polls the
/// provider's ring buffer while the provider moves bytes to the wire on its own
/// threads; `unified` means one loop does RX, dispatch and TX, driving the
/// transport through the provider's rx_poll/tx_publish hooks.  A provider
/// parses its own arguments in create() and cannot see this flag, so it needs
/// its own for the same shape.
///
/// THE PATH IS WHERE THE DISPATCH LOOP RUNS
///
///     --path=host     a host thread, HOST_CALL handler     (udp, cpu_roce)
///     --path=device   a GPU kernel, DEVICE_CALL handler    (gpu_roce)
///
/// The two shapes exist on both paths.  On the device, `ring` is the 3-kernel
/// arrangement -- the provider's RX and TX kernels around the dispatch kernel
/// -- and `unified` is the single kernel that drives the provider's device
/// hooks.  The device path is compiled in only when the build has the
/// gpu_roce tooling (CUDAQ_REALTIME_ENABLE_HSB_TOOLS); otherwise
/// `--path=device` is refused.  It differs from the host path in three
/// places, each a branch of a numbered step below: the dispatcher config [3],
/// the function table [5], and the control variables [6], which the GPU must
/// be able to read.  All device function-table construction is in
/// build_device_table().  The 3-kernel dispatch kernel comes from
/// libcudaq-realtime-dispatch.a, device-linked into this binary with the
/// handler, and the dispatcher finds its launcher there by itself.  The
/// unified kernel is linked at run time from libcudaq-realtime's own image,
/// the bridge's hooks and this binary's device code.
///
/// AN APPLICATION'S OWN DISPATCH KERNEL
///
///     --path=device --dispatch=unified --dispatcher=single_thread_dispatcher
///
/// replaces the dispatcher's unified kernel with one of this binary's own:
/// single_thread_dispatcher.cu (unittests/dispatchers/), which does RX,
/// dispatch and TX on one thread.  Two things make that work, and they are
/// all an application adds: the build embeds the kernel's LTO-IR in this
/// binary's device code (one cudaq_realtime_add_dispatch_kernel call in this
/// directory's CMakeLists.txt), so the run-time link that the handler lookup
/// triggers also takes the kernel; and step [5k] looks the kernel up and
/// hands the dispatcher a launcher, launch_own_dispatcher(), which
/// cudaq_dispatcher_start calls in place of launching its own kernel.
///
/// ORDERING RULE: the READY line is printed BEFORE cudaq_bridge_connect().
/// A rendezvous transport blocks in connect() until the caller dials in, while
/// the caller waits for READY before dialing -- announcing after connecting
/// would deadlock the pair.  Providers are built for this order: endpoint info
/// is valid as soon as create() returns.
///
/// Handshake, both parsed by cudaq::realtime::testing::ServerProcess:
///   CUDAQ_REALTIME_SERVER_READY <provider key=value...> path=<path>
///       dispatch=<shape> [dispatcher=<name>] slots=<N> slot_size=<M>
///       (one line; wrapped here; dispatcher= only with --dispatcher=)
///   CUDAQ_REALTIME_SERVER_PROCESSED count=<K>   (printed after shutdown)
///
/// The processed count is only final once the dispatch loop has exited, so it
/// is printed on the way out rather than served on request.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#ifdef CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
#include "driver_launch.h"
#include <cuda_runtime.h>
#endif

// Provided by init_rpc_increment_function_table_host.cpp.
extern "C" void
setup_rpc_increment_function_table_host(cudaq_function_entry_t *h_entries);

#ifdef CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
// Provided by init_rpc_increment_function_table.cu (rpc_increment_ft): an init
// kernel writes the address of the __device__ increment handler into the
// entry, which must be taken on the device.
extern "C" void
setup_rpc_increment_function_table(cudaq_function_entry_t *d_entries);
// The same entry, holding a handler address from a run-time linked module.
extern "C" cudaq_status_t
setup_rpc_increment_function_table_linked(cudaq_function_entry_t *d_entries,
                                          cudaq_dispatcher_t *dispatcher);
#endif

namespace {

struct ServerConfig {
  std::string transport = "udp";
  bool device = false;
  bool unified = false;
  // --dispatcher=: this binary's own dispatch kernel, run in place of the
  // dispatcher's (DEVICE unified only).  Empty for the dispatcher's own.
  std::string dispatcher;
  int timeout_sec = 60;
  // The bridge's own command line: the tokens after `--`, forwarded verbatim
  // and never interpreted here.  Points into main's argv, which outlives
  // create().
  int bridge_argc = 0;
  char **bridge_argv = nullptr;
};

std::atomic<int> g_shutdown{0};
void on_signal(int) { g_shutdown.store(1, std::memory_order_release); }

bool starts_with(const std::string &s, const char *prefix) {
  const std::size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [server options] [-- bridge options]\n\n"
      << "Serves the rpc_increment handler over any bridge provider.\n\n"
      << "Server options (before --):\n"
      << "  --transport=NAME    provider to load: a bare name resolved to\n"
      << "                      libcudaq-realtime-bridge-<name>.so (udp,\n"
      << "                      cpu_roce, gpu_roce, ...) or a path to a\n"
      << "                      provider .so               [default udp]\n"
      << "  --path=PATH         host | device             [default host]\n"
      << "                      host: HOST_CALL handler on a host thread;\n"
      << "                      device: DEVICE_CALL handler in a GPU kernel\n"
#ifndef CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
      << "                      (device: not built into this binary)\n"
#endif
      << "  --dispatch=SHAPE    ring | unified            [default ring]\n"
      << "                      wires THIS SERVER only; the bridge needs its\n"
      << "                      own flag for the same shape (udp: --unified)\n"
      << "  --dispatcher=NAME   run this binary's own dispatch kernel NAME\n"
      << "                      in place of the dispatcher's; device\n"
      << "                      unified only (single_thread_dispatcher)\n"
      << "  --timeout=N         run timeout in seconds    [default 60]\n\n"
      << "Bridge options (after --) are forwarded verbatim and never read\n"
      << "here, e.g. --port=N, --num-slots=N, --slot-size=N, --unified,\n"
      << "--pinned-rings, --device=NAME, --local-ip=ADDR, --qp_config=MODE,\n"
      << "--peer-ip=ADDR, --remote-qp=N, --num-pages=N, --page-size=N,\n"
      << "--payload-size=N.\n\n"
      << "Examples:\n"
      << "  " << program << " --transport=udp --dispatch=ring -- --port=0\n"
      << "  " << program
      << " --transport=udp --dispatch=unified -- --port=0 --unified\n"
      << "  " << program
      << " --transport=gpu_roce --path=device --dispatch=unified --\n"
      << "      --device=mlx5_0 --peer-ip=ADDR --remote-qp=N --unified\n";
}

// Ring geometry is queried from the provider rather than parsed here, so the
// dispatcher and the transport cannot disagree about slot count or stride.
bool parse_args(int argc, char **argv, ServerConfig &cfg, bool &help) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i] ? argv[i] : "";
    if (a == "--help" || a == "-h") {
      help = true;
      return false;
    }
    // `--` ends our options; the rest is the bridge's, untouched, including
    // tokens that look like ours.
    if (a == "--") {
      cfg.bridge_argv = argv + i + 1;
      cfg.bridge_argc = argc - i - 1;
      return true;
    }
    try {
      if (starts_with(a, "--transport="))
        cfg.transport = a.substr(12);
      else if (a == "--path=host")
        cfg.device = false;
      else if (a == "--path=device") {
#ifndef CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
        std::cerr << "ERROR: --path=device: this server was not built with "
                     "device support (it needs the gpu_roce tooling, "
                     "CUDAQ_REALTIME_ENABLE_HSB_TOOLS=ON)"
                  << std::endl;
        return false;
#endif
        cfg.device = true;
      } else if (a == "--dispatch=ring")
        cfg.unified = false;
      else if (a == "--dispatch=unified")
        cfg.unified = true;
      else if (starts_with(a, "--dispatcher="))
        cfg.dispatcher = a.substr(13);
      else if (starts_with(a, "--timeout="))
        cfg.timeout_sec = std::stoi(a.substr(10));
      else {
        std::cerr << "ERROR: unknown server option '" << a
                  << "'; bridge options go after '--'" << std::endl;
        return false;
      }
    } catch (const std::exception &) {
      std::cerr << "ERROR: bad numeric value in '" << a << "'" << std::endl;
      return false;
    }
  }
  return true;
}

// Resolve a --transport value to the provider library to dlopen: anything with
// a '/' is a caller-supplied path, a bare name maps to the shipped soname.
// Provider names are snake_case (udp, cpu_roce) while the sonames are
// hyphenated, so '_' maps to '-'; the literal spelling is probed too so an
// out-of-tree provider whose soname really contains an underscore resolves.
// The build directory is probed first so the tests run without an install.
std::string resolve_provider_lib(const std::string &transport) {
  if (transport.find('/') != std::string::npos)
    return transport;
  std::string hyphenated = transport;
  std::replace(hyphenated.begin(), hyphenated.end(), '_', '-');
  const std::string soname = "libcudaq-realtime-bridge-" + hyphenated + ".so";
#ifdef CUDAQ_REALTIME_BRIDGE_PROVIDER_DIR
  const std::string literal = "libcudaq-realtime-bridge-" + transport + ".so";
  for (const auto &name : {soname, literal}) {
    const std::string candidate =
        std::string(CUDAQ_REALTIME_BRIDGE_PROVIDER_DIR) + "/" + name;
    if (std::ifstream(candidate).good())
      return candidate;
  }
#endif
  return soname; // fall back to the dynamic loader's search path
}

// Reverse-order teardown, so any early `return 1` releases the transport too.
// The dispatcher goes first: its loop reads the provider's rings (and under
// the unified shape calls straight into the provider), so it must be stopped
// before the bridge is disconnected.  Every call is idempotent, which lets the
// normal shutdown path stop the dispatcher, read its final count, and then
// leave the rest to this destructor.
struct ServerResources {
  cudaq_realtime_bridge_handle_t bridge = nullptr;
  cudaq_dispatch_manager_t *manager = nullptr;
  cudaq_dispatcher_t *dispatcher = nullptr;
  // The host view of the dispatcher's shutdown flag, raised before stop():
  // on the device path, stop()'s own write is a cudaMemcpy that would queue
  // behind the kernel it is meant to end.
  volatile int *shutdown_flag = nullptr;
#ifdef CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
  // Device path only, freed last: the dispatch kernel reads all three.
  void *h_shutdown_flag = nullptr; // cudaHostAllocMapped
  std::uint64_t *d_stats = nullptr;
  cudaq_function_entry_t *d_entries = nullptr;
#endif

  ~ServerResources() {
    if (dispatcher) {
      if (shutdown_flag)
        *shutdown_flag = 1;
      cudaq_dispatcher_stop(dispatcher);
      cudaq_dispatcher_destroy(dispatcher);
    }
    if (manager)
      cudaq_dispatch_manager_destroy(manager);
    if (bridge) {
      cudaq_bridge_disconnect(bridge);
      cudaq_bridge_destroy(bridge);
    }
#ifdef CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
    if (d_entries)
      cudaFree(d_entries);
    if (d_stats)
      cudaFree(d_stats);
    if (h_shutdown_flag)
      cudaFreeHost(h_shutdown_flag);
#endif
  }
};

#ifdef CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
// Fill the device function table `d_entries` (one entry) for `dispatcher`.
// Every DEVICE_CALL entry holds a device-function address, which is only valid
// inside the CUDA module the dispatch kernel runs in, so how the table is
// built depends on where that kernel comes from.  The 3-kernel shape
// device-links it into this binary, together with the handler, so it takes
// the address from an init kernel in that same module.  The unified kernel is
// linked at run time with the bridge's hooks and this binary's device code
// (rpc_increment_handler.cu exports the handler), so the address comes
// from that module.
bool build_device_table(cudaq_dispatcher_t *dispatcher, bool unified,
                        cudaq_function_entry_t *d_entries) {
  if (unified) {
    if (setup_rpc_increment_function_table_linked(d_entries, dispatcher) !=
        CUDAQ_OK)
      return false;
  } else {
    setup_rpc_increment_function_table(d_entries);
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "ERROR: building the device function table: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }
  return true;
}

// The --dispatcher= kernel, looked up in the dispatcher's linked module in
// [5k].  cudaq_unified_launch_fn_t carries no user data, hence file scope.
cudaKernel_t g_own_dispatcher = nullptr;

// What cudaq_dispatcher_start calls in place of launching its own kernel when
// --dispatcher= is given: the application picks the grid and launch form, and
// launches through the driver API, the same on CUDA 12 and 13.
// single_thread_dispatcher works on one thread, so one block of one thread,
// where the block-partitioned increment handler does all of its bytes.  The
// arguments are the kernel's parameters, in order.  Launched on `stream`,
// which cudaq_dispatcher_stop waits on; a failure fails start().
cudaq_status_t
launch_own_dispatcher(void *transport_ctx, size_t tx_stride_sz,
                      cudaq_function_entry_t *function_table, size_t func_count,
                      volatile int *shutdown_flag, uint64_t *stats,
                      uint32_t /*threads_per_block*/, cudaStream_t stream) {
  void *args[] = {&transport_ctx,  &tx_stride_sz, &shutdown_flag,
                  &function_table, &func_count,   &stats};
  CUlaunchConfig launch{};
  launch.gridDimX = launch.gridDimY = launch.gridDimZ = 1;
  launch.blockDimX = launch.blockDimY = launch.blockDimZ = 1;
  launch.hStream = stream;
  return cudaq::realtime::testing::launch_kernel_ex(&launch, g_own_dispatcher,
                                                    args);
}
#endif

} // namespace

int main(int argc, char **argv) {
  ServerConfig cfg;
  bool help = false;
  if (!parse_args(argc, argv, cfg, help)) {
    if (help) {
      print_usage(argv[0]);
      return 0;
    }
    return 1;
  }
  const char *path = cfg.device ? "device" : "host";
  const char *shape = cfg.unified ? "unified" : "ring";
  if (!cfg.dispatcher.empty() && !(cfg.device && cfg.unified)) {
    std::cerr << "ERROR: --dispatcher= needs --path=device --dispatch=unified"
              << std::endl;
    return 1;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  ServerResources res;

  // [1] Load the provider and hand it its own command line -- the tokens after
  //     `--`, verbatim, and nothing of ours.
  const std::string library = resolve_provider_lib(cfg.transport);
  if (cudaq_bridge_create_from_library(&res.bridge, library.c_str(),
                                       cfg.bridge_argc,
                                       cfg.bridge_argv) != CUDAQ_OK) {
    std::cerr << "ERROR: cannot create a bridge from '" << library
              << "' (--transport=" << cfg.transport << ")" << std::endl;
    return 1;
  }

  // [2] Take the dispatcher's ring geometry and the endpoint to publish from
  //     the provider.  Both are interface-version-2 queries; a provider
  //     without them cannot be served or advertised.
  std::uint32_t num_slots = 0, slot_size = 0;
  if (cudaq_bridge_get_ring_geometry(res.bridge, &num_slots, &slot_size) !=
          CUDAQ_OK ||
      num_slots == 0 || slot_size == 0) {
    std::cerr << "ERROR: provider '" << library
              << "' did not report a usable ring geometry" << std::endl;
    return 1;
  }
  char endpoint[512] = {0};
  if (cudaq_bridge_get_endpoint_info(res.bridge, endpoint, sizeof(endpoint)) !=
      CUDAQ_OK) {
    std::cerr << "ERROR: provider '" << library
              << "' does not report endpoint info; the caller would have "
                 "nothing to dial"
              << std::endl;
    return 1;
  }

  // [3] Dispatcher: the path, its handler mode, and the shape -- the one place
  //     it appears.  Geometry is left at 0 for the bridge to supply.
  if (cudaq_dispatch_manager_create(&res.manager) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_dispatch_manager_create failed" << std::endl;
    return 1;
  }
  cudaq_dispatcher_config_t dcfg{};
  dcfg.kernel_type = cfg.unified ? CUDAQ_KERNEL_UNIFIED : CUDAQ_KERNEL_REGULAR;
  if (!cfg.device) {
    dcfg.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
    dcfg.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  } else {
    // [3d] DEVICE path, DEVICE_CALL mode.  Both dispatch kernels run one
    //      block of threads_per_block threads, which every DEVICE_CALL
    //      handler runs on.
    dcfg.dispatch_path = CUDAQ_DISPATCH_PATH_DEVICE;
    dcfg.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    dcfg.num_blocks = 1;
    dcfg.threads_per_block = 32;
  }
  if (cudaq_dispatcher_create(res.manager, &dcfg, &res.dispatcher) !=
      CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_dispatcher_create failed" << std::endl;
    return 1;
  }

  // [4] Wire the transport to the dispatcher.  The dispatcher asks the bridge
  //     for the data plane the shape needs -- a ring it polls, or the two hooks
  //     it drives itself -- and fails here if the provider serves the other.
  if (cudaq_dispatcher_set_bridge(res.dispatcher, res.bridge) != CUDAQ_OK) {
    std::cerr << "ERROR: provider '" << library << "' cannot serve the "
              << shape << " shape (start it with matching options)"
              << std::endl;
    return 1;
  }

  // [5] Function table: the single increment handler.  The table's entries
  //     are read by the dispatch loop, so on the device path they live in
  //     device memory.
  cudaq_function_entry_t entries[1];
  cudaq_function_table_t table{};
  table.entries = entries;
  table.count = 1;
  if (!cfg.device) {
    setup_rpc_increment_function_table_host(entries);
  } else {
#ifdef CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
    // [5d] The DEVICE_CALL handler, in a device table.
    if (cudaMalloc(&res.d_entries, sizeof(cudaq_function_entry_t)) !=
            cudaSuccess ||
        !build_device_table(res.dispatcher, cfg.unified, res.d_entries)) {
      std::cerr << "ERROR: cannot build the device function table" << std::endl;
      return 1;
    }
    table.entries = res.d_entries;
#endif
  }
  if (cudaq_dispatcher_set_function_table(res.dispatcher, &table) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_dispatcher_set_function_table failed"
              << std::endl;
    return 1;
  }

#ifdef CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
  // [5k] --dispatcher= only: this binary's own dispatch kernel in place of
  //      the dispatcher's.  The handler lookup in [5d] linked the module from
  //      this binary's device code, which holds the kernel too (embedded by
  //      the build), so the kernel is looked up the same way.  The launcher
  //      takes the bridge's transport context, hence the NULL here.
  if (!cfg.dispatcher.empty() &&
      (cudaq_dispatcher_get_device_kernel(res.dispatcher,
                                          cfg.dispatcher.c_str(),
                                          &g_own_dispatcher) != CUDAQ_OK ||
       cudaq_dispatcher_set_unified_launch(
           res.dispatcher, launch_own_dispatcher, nullptr) != CUDAQ_OK)) {
    std::cerr << "ERROR: cannot run dispatch kernel '" << cfg.dispatcher << "'"
              << std::endl;
    return 1;
  }
#endif

  // [6] Control variables, then start polling.  A HOST_CALL-only table needs
  //     no graph engine and therefore no GPU.
  volatile int host_shutdown_flag = 0;
  std::uint64_t host_stats = 0;
  volatile int *shutdown_flag = &host_shutdown_flag;
  std::uint64_t *stats = &host_stats;
#ifdef CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
  if (cfg.device) {
    // [6d] The kernel polls the flag, so it lives in mapped pinned memory and
    //      the dispatcher gets its device view; the host keeps the other for
    //      the shutdown in [11].  The kernel adds its count to a device word.
    void *d_shutdown_flag = nullptr;
    if (cudaHostAlloc(&res.h_shutdown_flag, sizeof(int), cudaHostAllocMapped) !=
            cudaSuccess ||
        cudaHostGetDevicePointer(&d_shutdown_flag, res.h_shutdown_flag, 0) !=
            cudaSuccess ||
        cudaMalloc(&res.d_stats, sizeof(std::uint64_t)) != cudaSuccess ||
        cudaMemset(res.d_stats, 0, sizeof(std::uint64_t)) != cudaSuccess) {
      std::cerr << "ERROR: cannot allocate the device control variables"
                << std::endl;
      return 1;
    }
    *static_cast<volatile int *>(res.h_shutdown_flag) = 0;
    res.shutdown_flag = static_cast<volatile int *>(res.h_shutdown_flag);
    shutdown_flag = static_cast<volatile int *>(d_shutdown_flag);
    stats = res.d_stats;
  }
#endif
  if (cudaq_dispatcher_set_control(res.dispatcher, shutdown_flag, stats) !=
      CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_dispatcher_set_control failed" << std::endl;
    return 1;
  }
  if (cudaq_dispatcher_start(res.dispatcher) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_dispatcher_start failed" << std::endl;
    return 1;
  }

  // [7] Announce BEFORE connecting: see the ORDERING RULE at the top.  The
  //     endpoint string is the provider's own key=value description, passed
  //     through so the harness needs no per-transport parsing.  std::endl
  //     flushes, which matters because stdout is a pipe here.
  std::cout << "CUDAQ_REALTIME_SERVER_READY " << endpoint << " path=" << path
            << " dispatch=" << shape;
  if (!cfg.dispatcher.empty())
    std::cout << " dispatcher=" << cfg.dispatcher;
  std::cout << " slots=" << num_slots << " slot_size=" << slot_size
            << std::endl;

  // [8] Connect: a no-op for connectionless transports, a blocking wait for
  //     the caller on a rendezvous transport.
  if (cudaq_bridge_connect(res.bridge) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_bridge_connect failed" << std::endl;
    return 1;
  }

  // [9] Hand the provider its go-ahead, last, once the dispatcher is already
  //     polling.  Unconditional: a transport with nothing to start under this
  //     shape (udp under --unified, whose loop is ours) does nothing here,
  //     which is its business rather than ours to predict.
  if (cudaq_bridge_launch(res.bridge) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_bridge_launch failed" << std::endl;
    return 1;
  }

  // [10] Serve until SIGTERM/SIGINT or the timeout backstop.
  const auto started = std::chrono::steady_clock::now();
  while (g_shutdown.load(std::memory_order_acquire) == 0) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    if (elapsed >= cfg.timeout_sec) {
      std::cerr << "timeout reached (" << cfg.timeout_sec << "s)" << std::endl;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // [11] Stop the dispatch loop, then report the count it accumulated (only
  //      final once the loop has exited).  ServerResources unwinds the rest.
  //      A device kernel is ended through the flag's host view first, see
  //      ServerResources::shutdown_flag.
  if (res.shutdown_flag)
    *res.shutdown_flag = 1;
  cudaq_dispatcher_stop(res.dispatcher);
  std::uint64_t processed = 0;
  cudaq_dispatcher_get_processed(res.dispatcher, &processed);
  std::cout << "CUDAQ_REALTIME_SERVER_PROCESSED count=" << processed
            << std::endl;
  return 0;
}
