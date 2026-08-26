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
/// Transport-agnostic by construction.  Every transport is reached through the
/// same bridge vtable (bridge_interface.h), and each provider owns its own
/// bring-up -- udp binds a socket in create(), cpu_roce does the full TCP
/// rendezvous plus QP/rkey swap across create()/connect() -- so there is no
/// per-transport code here.  `--transport=` names the provider library and
/// every unrecognized argument is forwarded to it verbatim, which is how
/// `--port=`, `--device=`, `--local-ip=`, `--qp_config=`, `--peer-ip=`,
/// `--remote-qp=`, `--num-slots=` and `--slot-size=` reach the provider that
/// understands them.  Providers ignore arguments they do not recognize, by
/// contract, so forwarding the whole command line is safe.
///
/// ORDERING RULE: the READY line is printed BEFORE cudaq_bridge_connect().
/// A rendezvous transport blocks in connect() until the caller dials in, while
/// the caller waits for READY before dialing -- announcing after connecting
/// would deadlock the pair.  Providers are built for this order: endpoint info
/// is valid as soon as create() returns.
///
/// Two dispatch shapes are wired through the three functions below, so a shape
/// is one command-line token rather than a second binary:
///   --dispatch=ring     dispatcher polls the provider's ring buffer while the
///                       provider's own pump threads move bytes to the wire.
///   --dispatch=unified  dispatcher drives the transport itself through the
///                       provider's rx_poll/tx_publish hooks and the provider
///                       starts no threads.  Fails cleanly with UNSUPPORTED
///                       against a provider whose get_cpu_dataplane is NULL.
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

enum class Shape { Ring, Unified };

const char *shape_name(Shape shape) {
  return shape == Shape::Unified ? "unified" : "ring";
}

struct ServerConfig {
  std::string transport = "udp";
  Shape shape = Shape::Ring;
  int timeout_sec = 60;
};

std::atomic<int> g_shutdown{0};
void on_signal(int) { g_shutdown.store(1, std::memory_order_release); }

bool starts_with(const std::string &s, const char *prefix) {
  const std::size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options] [provider options]\n\n"
      << "Serves the rpc_increment HOST_CALL handler over any bridge "
         "provider.\n\n"
      << "Options:\n"
      << "  --transport=NAME    provider to load: a bare name resolved to\n"
      << "                      libcudaq-realtime-bridge-<name>.so (udp,\n"
      << "                      cpu_roce, ...) or a path to a provider .so\n"
      << "                      [default udp]\n"
      << "  --dispatch=SHAPE    ring | unified            [default ring]\n"
      << "  --timeout=N         run timeout in seconds    [default 60]\n\n"
      << "All other options are forwarded to the provider, e.g. --port=N,\n"
      << "--num-slots=N, --slot-size=N, --device=NAME, --local-ip=ADDR,\n"
      << "--qp_config=MODE, --peer-ip=ADDR, --remote-qp=N.\n";
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
    try {
      if (starts_with(a, "--transport="))
        cfg.transport = a.substr(12);
      else if (starts_with(a, "--dispatch=")) {
        const std::string shape = a.substr(11);
        if (shape == "ring")
          cfg.shape = Shape::Ring;
        else if (shape == "unified")
          cfg.shape = Shape::Unified;
        else {
          std::cerr << "ERROR: unknown --dispatch=" << shape
                    << " (expected ring or unified)" << std::endl;
          return false;
        }
      } else if (starts_with(a, "--timeout="))
        cfg.timeout_sec = std::stoi(a.substr(10));
      // Everything else belongs to the provider; forwarded untouched.
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

//===----------------------------------------------------------------------===//
// The shape seam: the only three shape-aware functions in this file.
//===----------------------------------------------------------------------===//

// Transport context the dispatcher is wired to.  The unified data-plane is
// held by pointer inside the dispatcher, so this must outlive it (declared
// before the dispatcher in main, destroyed after).
struct ShapeState {
  cudaq_ringbuffer_t ring{};
  cudaq_cpu_dataplane_t dataplane{};
};

// (1) Read at cudaq_dispatcher_create time.
void configure_shape(Shape shape, cudaq_dispatcher_config_t &dcfg) {
  if (shape == Shape::Unified) {
    // The unified loop keeps the per-slot TX markers (its publish hook reads
    // them to tell a running graph from a finished one), so skip_tx_markers
    // stays 0; the dispatcher overrides it regardless.
    dcfg.kernel_type = CUDAQ_KERNEL_UNIFIED;
    return;
  }
  dcfg.kernel_type = CUDAQ_KERNEL_REGULAR;
  // The provider's TX pump owns the wire and treats any non-zero flag as a
  // slot address, so the in-flight sentinel must not be written.
  dcfg.skip_tx_markers = 1;
}

// (2) Called after create, before start.
cudaq_status_t wire_shape(Shape shape, cudaq_realtime_bridge_handle_t bridge,
                          cudaq_dispatcher_t *dispatcher, ShapeState &state) {
  if (shape == Shape::Unified) {
    const cudaq_status_t status =
        cudaq_bridge_get_cpu_dataplane(bridge, &state.dataplane);
    if (status != CUDAQ_OK)
      return status;
    return cudaq_dispatcher_set_cpu_dataplane(dispatcher, &state.dataplane);
  }
  const cudaq_status_t status =
      cudaq_bridge_get_transport_context(bridge, RING_BUFFER, &state.ring);
  if (status != CUDAQ_OK)
    return status;
  return cudaq_dispatcher_set_ringbuffer(dispatcher, &state.ring);
}

// (3) Unified drives the transport from the dispatcher's own thread; starting
// the provider's pump threads as well would race its hooks for the rings.
bool needs_bridge_launch(Shape shape) { return shape != Shape::Unified; }

//===----------------------------------------------------------------------===//

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

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // Declared before the resources so it outlives the dispatcher that points
  // into it (see ShapeState).
  ShapeState state;
  ServerResources res;

  // [1] Load the provider and let it parse our whole command line.
  const std::string library = resolve_provider_lib(cfg.transport);
  if (cudaq_bridge_create_from_library(&res.bridge, library.c_str(), argc,
                                       argv) != CUDAQ_OK) {
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
  if (cudaq_dispatch_manager_create(&res.manager) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_dispatch_manager_create failed" << std::endl;
    return 1;
  }
  cudaq_dispatcher_config_t dcfg{};
  dcfg.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  dcfg.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  dcfg.num_slots = num_slots;
  dcfg.slot_size = slot_size;
  configure_shape(cfg.shape, dcfg);
  if (cudaq_dispatcher_create(res.manager, &dcfg, &res.dispatcher) !=
      CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_dispatcher_create failed" << std::endl;
    return 1;
  }

  // [4] Wire the transport to the dispatcher (shape-dependent).
  const cudaq_status_t wired =
      wire_shape(cfg.shape, res.bridge, res.dispatcher, state);
  if (wired != CUDAQ_OK) {
    std::cerr << "ERROR: cannot wire --dispatch=" << shape_name(cfg.shape)
              << " to provider '" << library << "': "
              << (wired == CUDAQ_ERR_UNSUPPORTED
                      ? "the provider does not support this shape"
                      : "wiring failed")
              << std::endl;
    return 1;
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
            << " dispatch=" << shape_name(cfg.shape) << " slots=" << num_slots
            << " slot_size=" << slot_size << std::endl;

  // [8] Connect: a no-op for connectionless transports, a blocking wait for
  //     the caller on a rendezvous transport.
  if (cudaq_bridge_connect(res.bridge) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_bridge_connect failed" << std::endl;
    return 1;
  }

  // [9] Start the provider's pump threads last, once the dispatcher is
  //     already polling.
  if (needs_bridge_launch(cfg.shape) &&
      cudaq_bridge_launch(res.bridge) != CUDAQ_OK) {
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
