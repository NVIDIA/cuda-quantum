/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file cudaq_realtime_test_server.cpp
/// @brief Service half of the realtime two-process tests: a dispatcher serving
///        the `rpc_increment` HOST_CALL handler over any bridge provider.
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
/// ORDERING RULE: the READY line is printed BEFORE cudaq_bridge_connect().
/// A rendezvous transport blocks in connect() until the caller dials in, while
/// the caller waits for READY before dialing -- announcing after connecting
/// would deadlock the pair.  Providers are built for this order: endpoint info
/// is valid as soon as create() returns.
///
/// Handshake, both parsed by cudaq::realtime::testing::ServerProcess:
///   CUDAQ_REALTIME_SERVER_READY <provider key=value...> dispatch=<shape>
///       slots=<N> slot_size=<M>            (one line; wrapped here)
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

// Provided by init_rpc_increment_function_table_host.cpp.
extern "C" void
setup_rpc_increment_function_table_host(cudaq_function_entry_t *h_entries);

namespace {

struct ServerConfig {
  std::string transport = "udp";
  bool unified = false;
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
      << "Serves the rpc_increment HOST_CALL handler over any bridge "
         "provider.\n\n"
      << "Server options (before --):\n"
      << "  --transport=NAME    provider to load: a bare name resolved to\n"
      << "                      libcudaq-realtime-bridge-<name>.so (udp,\n"
      << "                      cpu_roce, ...) or a path to a provider .so\n"
      << "                      [default udp]\n"
      << "  --dispatch=SHAPE    ring | unified            [default ring]\n"
      << "                      wires THIS SERVER only; the bridge needs its\n"
      << "                      own flag for the same shape (udp: --unified)\n"
      << "  --timeout=N         run timeout in seconds    [default 60]\n\n"
      << "Bridge options (after --) are forwarded verbatim and never read\n"
      << "here, e.g. --port=N, --num-slots=N, --slot-size=N, --unified,\n"
      << "--pinned-rings, --device=NAME, --local-ip=ADDR, --qp_config=MODE,\n"
      << "--peer-ip=ADDR, --remote-qp=N.\n\n"
      << "Examples:\n"
      << "  " << program << " --transport=udp --dispatch=ring -- --port=0\n"
      << "  " << program
      << " --transport=udp --dispatch=unified -- --port=0 --unified\n";
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
      else if (a == "--dispatch=ring")
        cfg.unified = false;
      else if (a == "--dispatch=unified")
        cfg.unified = true;
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

  ~ServerResources() {
    if (dispatcher) {
      cudaq_dispatcher_stop(dispatcher);
      cudaq_dispatcher_destroy(dispatcher);
    }
    if (manager)
      cudaq_dispatch_manager_destroy(manager);
    if (bridge) {
      cudaq_bridge_disconnect(bridge);
      cudaq_bridge_destroy(bridge);
    }
  }
};

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
  const char *shape = cfg.unified ? "unified" : "ring";

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // Declared up here, and not down at [4] where it is filled, because
  // cudaq_dispatcher_set_cpu_dataplane RETAINS THIS POINTER -- the loop
  // dereferences it on every iteration.  Before `res` for the same reason: its
  // destructor stops the dispatcher, which has to happen while the struct the
  // dispatcher is reading is still alive.  The ring buffer needs none of this
  // and stays local to its branch: set_ringbuffer copies.
  cudaq_cpu_dataplane_t dataplane{};
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

  // [3] Dispatcher: HOST path, HOST_CALL mode, geometry from the transport.
  //     One of the two places the shape appears.
  if (cudaq_dispatch_manager_create(&res.manager) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_dispatch_manager_create failed" << std::endl;
    return 1;
  }
  cudaq_dispatcher_config_t dcfg{};
  dcfg.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  dcfg.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  dcfg.num_slots = num_slots;
  dcfg.slot_size = slot_size;
  dcfg.kernel_type = cfg.unified ? CUDAQ_KERNEL_UNIFIED : CUDAQ_KERNEL_REGULAR;
  // Ring shape only: the provider's TX pump owns the wire and treats any
  // non-zero flag as a slot address, so the in-flight sentinel must not be
  // written.  The unified loop needs that sentinel (its publish hook reads it
  // to tell a running graph from a finished one) and forces this to 0 anyway.
  dcfg.skip_tx_markers = cfg.unified ? 0 : 1;
  if (cudaq_dispatcher_create(res.manager, &dcfg, &res.dispatcher) !=
      CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_dispatcher_create failed" << std::endl;
    return 1;
  }

  // [4] Wire the transport to the dispatcher: a ring the dispatcher polls, or
  //     the two hooks it drives itself.  The other place the shape appears, and
  //     the one that catches a shape the provider is not serving.
  if (cfg.unified) {
    cudaq_unified_dispatch_ctx_t unified_dispatch{};
    if (cudaq_bridge_get_transport_context(res.bridge, UNIFIED,
                                           &unified_dispatch) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to get unified dispatch context" << std::endl;
      return 1;
    }
    if (unified_dispatch.launch_fn != nullptr &&
        unified_dispatch.transport_ctx != nullptr &&
        dcfg.dispatch_path == CUDAQ_DISPATCH_PATH_DEVICE) {
      // Legacy path: the bridge provides the dispatch call.
      if (cudaq_dispatcher_set_unified_launch(
              res.dispatcher, unified_dispatch.launch_fn,
              unified_dispatch.transport_ctx) != CUDAQ_OK) {
        std::cerr << "ERROR: Failed to set unified launch function"
                  << std::endl;
        return 1;
      }
    } else {
      if (cudaq_bridge_get_cpu_dataplane(res.bridge, &dataplane) != CUDAQ_OK) {
        std::cerr << "ERROR: Failed to get CPU dataplane" << std::endl;
        return 1;
      }
      if (cudaq_dispatcher_set_cpu_dataplane(res.dispatcher, &dataplane) !=
          CUDAQ_OK) {
        std::cerr << "ERROR: Failed to set CPU dataplane" << std::endl;
        return 1;
      }
    }
  } else {
    cudaq_ringbuffer_t ring{};
    if (cudaq_bridge_get_transport_context(res.bridge, RING_BUFFER, &ring) !=
        CUDAQ_OK) {
      std::cerr << "ERROR: Failed to get ring buffer context" << std::endl;
      return 1;
    }
    if (cudaq_dispatcher_set_ringbuffer(res.dispatcher, &ring) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to set ring buffer" << std::endl;
      return 1;
    }

    // Legacy path: the dispatching call is provided externally.
    if (dcfg.dispatch_path == CUDAQ_DISPATCH_PATH_DEVICE) {
      if (cudaq_dispatcher_set_launch_fn(
              res.dispatcher, &cudaq_launch_dispatch_kernel_regular) !=
          CUDAQ_OK) {
        std::cerr << "ERROR: Failed to set launch function" << std::endl;
        return 1;
      }
    }
  }

  // [5] Function table: the single host-side increment handler.
  cudaq_function_entry_t entries[1];
  setup_rpc_increment_function_table_host(entries);
  cudaq_function_table_t table{};
  table.entries = entries;
  table.count = 1;
  if (cudaq_dispatcher_set_function_table(res.dispatcher, &table) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_dispatcher_set_function_table failed"
              << std::endl;
    return 1;
  }

  // [6] Control variables, then start polling.  A HOST_CALL-only table needs
  //     no graph engine and therefore no GPU.
  volatile int shutdown_flag = 0;
  std::uint64_t stats = 0;
  if (cudaq_dispatcher_set_control(res.dispatcher, &shutdown_flag, &stats) !=
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
  std::cout << "CUDAQ_REALTIME_SERVER_READY " << endpoint
            << " dispatch=" << shape << " slots=" << num_slots
            << " slot_size=" << slot_size << std::endl;

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
  cudaq_dispatcher_stop(res.dispatcher);
  std::uint64_t processed = 0;
  cudaq_dispatcher_get_processed(res.dispatcher, &processed);
  std::cout << "CUDAQ_REALTIME_SERVER_PROCESSED count=" << processed
            << std::endl;
  return 0;
}
