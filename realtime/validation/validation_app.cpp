/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file validation_app.cpp
/// @brief Serve validation requests through a runtime-loaded RTH bridge.
///
/// The app selects CPU/GPU dispatch and registers the shared increment handler.
/// The bridge owns networking setup and interprets every option after --.
/// Native forward mode uses only the bridge's echo path. The runner coordinates
/// this receiver with playback_app, which verifies replies and reports timing.
#include "error_handling.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

extern "C" void
setup_rpc_increment_function_table_host(cudaq_function_entry_t *entries);
extern "C" void
setup_rpc_increment_function_table(cudaq_function_entry_t *entries);
extern "C" cudaq_status_t
setup_rpc_increment_function_table_linked(cudaq_function_entry_t *entries,
                                          cudaq_dispatcher_t *dispatcher);

namespace {
// Execution settings belong to the app; networking settings remain opaque.
struct Config {
  bool cpu = false; ///< Use host dispatch instead of CUDA device dispatch.
  bool unified = false; ///< Combine receive, dispatch and transmit execution.
  bool forward = false; ///< Echo through the bridge without an RPC handler.
  int gpu = 0; ///< CUDA device ordinal when using GPU execution.
  int timeout = 60; ///< Maximum service time in seconds after App Ready.
  unsigned threads = 32; ///< Threads per GPU dispatch block.
  std::string bridge; ///< Shared library passed to the bridge loader.
  int bridge_argument_count = 0; ///< Number of arguments after --.
  char **bridge_arguments = nullptr; ///< Borrowed argv tail; valid through main.
};

bool parse_args(int argument_count, char **arguments, Config &config) {
  bool cpu_selected = false, gpu_selected = false;
  bool dispatch_selected = false;
  for (int argument_index = 1; argument_index < argument_count;
       ++argument_index) {
    const std::string argument = arguments[argument_index];
    // Preserve the bridge's argument spelling and boundaries exactly. In
    // particular, the app must not reinterpret provider-specific options.
    if (argument == "--") {
      config.bridge_argument_count = argument_count - argument_index - 1;
      config.bridge_arguments = arguments + argument_index + 1;
      break;
    }
    const auto equals_position = argument.find('=');
    const auto key = argument.substr(0, equals_position);
    const auto value = equals_position == std::string::npos
                           ? ""
                           : argument.substr(equals_position + 1);
    if (argument == "--help" || argument == "-h") {
      std::cout
          << "Usage: validation_app --bridge=LIB [options] -- [bridge "
             "options]\n"
          << "  --bridge=LIB         Provider soname or path (required)\n"
          << "  --cpu / --gpu        Host or device dispatch (default: GPU)\n"
          << "  --gpu-id=N           GPU device ID (default: 0; --gpu=N "
             "accepted)\n"
          << "  --timeout=N          Run timeout in seconds (default: 60)\n"
          << "  --threads=N          GPU dispatch block size (default: 32)\n"
          << "  --dispatch=MODE      separate (default), unified, or forward\n"
          << "  --forward            Bypass dispatch; configure native "
             "forwarding\n"
          << "                       separately in the provider's options\n"
          << "  --unified            Unified dispatch (default: separate "
             "RX/dispatch/TX)\n"
          << "                       Configure the provider for the same "
             "shape\n"
          << "Options after -- are forwarded verbatim, without "
             "interpretation.\n"
          << "The workload is rpc_increment; forward mode runs no RPC "
             "handler.\n";
      return false;
    } else if (argument == "--cpu") {
      cpu_selected = config.cpu = true;
    } else if (key == "--gpu") {
      gpu_selected = true;
      if (equals_position != std::string::npos)
        config.gpu = std::stoi(value);
    } else if (key == "--gpu-id")
      config.gpu = std::stoi(value);
    else if (key == "--bridge")
      config.bridge = value;
    else if (key == "--timeout")
      config.timeout = std::stoi(value);
    else if (key == "--threads")
      config.threads = std::stoul(value);
    else if (key == "--dispatch" || argument == "--unified" ||
             argument == "--forward") {
      if (dispatch_selected)
        throw std::runtime_error("Specify dispatch mode only once");
      dispatch_selected = true;
      const auto mode = key == "--dispatch" ? value : argument.substr(2);
      if (mode != "separate" && mode != "unified" && mode != "forward")
        throw std::runtime_error(
            "Dispatch mode must be separate, unified, or forward");
      config.unified = mode == "unified";
      config.forward = mode == "forward";
    } else
      throw std::runtime_error("Unknown app argument: " + argument +
                               "; bridge options belong after --");
  }
  if (cpu_selected && gpu_selected)
    throw std::runtime_error("--cpu and --gpu are mutually exclusive");
  if (config.forward && config.unified)
    throw std::runtime_error("--forward and --unified are mutually exclusive");
  if (config.bridge.empty())
    throw std::runtime_error("--bridge=LIB is required");
  if (config.timeout <= 0 || config.gpu < 0 || config.threads == 0 ||
      config.threads > 1024)
    throw std::runtime_error("Invalid timeout, GPU ID, or thread count");
  return true;
}

// Signal handlers only request shutdown; the main thread releases resources.
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

// Tables and control storage must outlive the dispatcher and bridge, including
// partial initialization. CPU mode never allocates/frees CUDA memory.
struct Resources {
  cudaq_realtime_bridge_handle_t bridge = nullptr; ///< Owns networking resources.
  cudaq_dispatch_manager_t *manager = nullptr; ///< Owns dispatcher management.
  cudaq_dispatcher_t *dispatcher = nullptr; ///< Absent in native forward mode.
  volatile int host_shutdown = 0; ///< Stop flag read by CPU dispatch.
  uint64_t host_stats = 0; ///< CPU dispatch's processed-request counter.
  cudaq_function_entry_t host_entries[1]{}; ///< Host increment handler entry.
  int *mapped_shutdown = nullptr; ///< Host-written, GPU-readable stop flag.
  uint64_t *device_stats = nullptr; ///< GPU processed-request counter.
  cudaq_function_entry_t *device_entries = nullptr; ///< GPU handler table.

  void stop() {
    // Set mapped memory before stop()'s CUDA copy: the default stream may
    // otherwise wait for the persistent kernels that this flag terminates.
    if (mapped_shutdown) {
      *static_cast<volatile int *>(mapped_shutdown) = 1;
      __sync_synchronize();
    }
    if (dispatcher)
      HANDLE_REALTIME_ERROR(cudaq_dispatcher_stop(dispatcher));
  }

  // Normal shutdown is checked before main returns success. The destructor
  // also cleans up partial setup, but cannot turn a failed operation into PASS.
  void close() {
    // Dispatch must stop using bridge buffers before the bridge is destroyed.
    // Handler/control storage is released last; null guards cover partial setup.
    stop();
    if (dispatcher) {
      HANDLE_REALTIME_ERROR(cudaq_dispatcher_destroy(dispatcher));
      dispatcher = nullptr;
    }
    if (manager) {
      HANDLE_REALTIME_ERROR(cudaq_dispatch_manager_destroy(manager));
      manager = nullptr;
    }
    if (bridge) {
      HANDLE_REALTIME_ERROR(cudaq_bridge_disconnect(bridge));
      HANDLE_REALTIME_ERROR(cudaq_bridge_destroy(bridge));
      bridge = nullptr;
    }
    if (device_entries) {
      HANDLE_CUDA_ERROR(cudaFree(device_entries));
      device_entries = nullptr;
    }
    if (device_stats) {
      HANDLE_CUDA_ERROR(cudaFree(device_stats));
      device_stats = nullptr;
    }
    if (mapped_shutdown) {
      HANDLE_CUDA_ERROR(cudaFreeHost(mapped_shutdown));
      mapped_shutdown = nullptr;
    }
  }
  ~Resources() {
    try {
      close();
    } catch (const std::exception &error) {
      std::cerr << "Shutdown error: " << error.what() << std::endl;
    }
  }
};
} // namespace

int main(int argument_count, char **arguments) {
  try {
    Config config;
    if (!parse_args(argument_count, arguments, config))
      return 0;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    Resources resources;

    std::cout << "CUDAQ_VALIDATION_CASE execution="
              << (config.cpu ? "cpu" : "gpu") << " dispatch="
              << (config.forward   ? "forward"
                  : config.unified ? "unified"
                                   : "separate")
              << " workload="
              << (config.forward ? "native_forward" : "rpc_increment")
              << " bridge=" << config.bridge << std::endl;
    if (!config.cpu)
      HANDLE_CUDA_ERROR(cudaSetDevice(config.gpu));
    std::cout << "=== Validation App (" << (config.cpu ? "CPU" : "GPU")
              << ") ===\nLoading " << config.bridge << std::endl;
    // 1. Load the bridge and pass its options through unchanged. The bridge
    // chooses its transport resources; the app does not assume ring geometry.
    HANDLE_REALTIME_ERROR(cudaq_bridge_create_from_library(
        &resources.bridge, config.bridge.c_str(), config.bridge_argument_count,
        config.bridge_arguments));

    // Publish the endpoint before connect(): rendezvous providers can block
    // there while waiting for a producer to use the advertised endpoint.
    char endpoint[4096]{};
    const auto endpoint_status = cudaq_bridge_get_endpoint_info(
        resources.bridge, endpoint, sizeof(endpoint));
    if (endpoint_status == CUDAQ_OK)
      std::cout << "CUDAQ_VALIDATION_ENDPOINT " << endpoint << std::endl;
    else if (endpoint_status != CUDAQ_ERR_UNSUPPORTED)
      HANDLE_REALTIME_ERROR(endpoint_status);
    else
      // Some bridges need no published endpoint. Still emit the marker so
      // the runner can start provider preparation with an empty endpoint.
      std::cout << "CUDAQ_VALIDATION_ENDPOINT " << std::endl;
    HANDLE_REALTIME_ERROR(cudaq_bridge_connect(resources.bridge));

    // 2. Attach a dispatcher unless the bridge itself is forwarding packets.
    // The selected bridge must support the requested execution/dispatch mode.
    if (!config.forward) {
      HANDLE_REALTIME_ERROR(cudaq_dispatch_manager_create(&resources.manager));
      cudaq_dispatcher_config_t dispatcher_config{};
      dispatcher_config.device_id = config.gpu;
      dispatcher_config.dispatch_path =
          config.cpu ? CUDAQ_DISPATCH_PATH_HOST : CUDAQ_DISPATCH_PATH_DEVICE;
      dispatcher_config.dispatch_mode =
          config.cpu ? CUDAQ_DISPATCH_HOST_CALL : CUDAQ_DISPATCH_DEVICE_CALL;
      dispatcher_config.kernel_type =
          config.unified ? CUDAQ_KERNEL_UNIFIED : CUDAQ_KERNEL_REGULAR;
      dispatcher_config.num_blocks = 1;
      dispatcher_config.threads_per_block = config.threads;
      HANDLE_REALTIME_ERROR(cudaq_dispatcher_create(
          resources.manager, &dispatcher_config, &resources.dispatcher));
      // Let the bridge supply ring buffers or integrated transport hooks.
      // The dispatcher validates that they support the selected execution mode.
      HANDLE_REALTIME_ERROR(
          cudaq_dispatcher_set_bridge(resources.dispatcher, resources.bridge));

      // 3. Register the one validation RPC in memory accessible to dispatch.
      // CPU dispatch uses ordinary host storage and makes no CUDA allocations.
      cudaq_function_table_t table{};
      table.count = 1;
      volatile int *shutdown = &resources.host_shutdown;
      uint64_t *stats = &resources.host_stats;
      if (config.cpu) {
        setup_rpc_increment_function_table_host(resources.host_entries);
        table.entries = resources.host_entries;
      } else {
        // Mapped host memory lets shutdown signal persistent GPU work without
        // first scheduling a CUDA copy behind that work.
        HANDLE_CUDA_ERROR(cudaHostAlloc(&resources.mapped_shutdown, sizeof(int),
                                        cudaHostAllocMapped));
        *resources.mapped_shutdown = 0;
        int *device_shutdown = nullptr;
        // The device alias need not equal the host pointer. Pass that alias
        // to the kernel while retaining the host pointer for shutdown writes.
        HANDLE_CUDA_ERROR(cudaHostGetDevicePointer(
            &device_shutdown, resources.mapped_shutdown, 0));
        shutdown = device_shutdown;
        HANDLE_CUDA_ERROR(
            cudaMalloc(&resources.device_stats, sizeof(uint64_t)));
        HANDLE_CUDA_ERROR(
            cudaMemset(resources.device_stats, 0, sizeof(uint64_t)));
        stats = resources.device_stats;
        HANDLE_CUDA_ERROR(cudaMalloc(&resources.device_entries,
                                     sizeof(cudaq_function_entry_t)));
        // Unified dispatch resolves the handler in the runtime-linked module;
        // separate dispatch initializes the executable's device handler entry.
        if (config.unified)
          HANDLE_REALTIME_ERROR(setup_rpc_increment_function_table_linked(
              resources.device_entries, resources.dispatcher));
        else {
          setup_rpc_increment_function_table(resources.device_entries);
          HANDLE_CUDA_ERROR(cudaGetLastError());
        }
        table.entries = resources.device_entries;
      }
      HANDLE_REALTIME_ERROR(
          cudaq_dispatcher_set_function_table(resources.dispatcher, &table));
      // Bridges with an integrated dispatch path also need the handler table.
      HANDLE_REALTIME_ERROR(
          cudaq_bridge_set_function_table(resources.bridge, &table));
      // Control storage and handler entries stay owned by Resources for the
      // entire run; the dispatcher receives pointers, not independent copies.
      HANDLE_REALTIME_ERROR(
          cudaq_dispatcher_set_control(resources.dispatcher, shutdown, stats));
      HANDLE_REALTIME_ERROR(cudaq_dispatcher_start(resources.dispatcher));
    }
    // 4. Start transport work after dispatch is ready. The runner waits for
    // this exact readiness marker before releasing the producer's start gate.
    HANDLE_REALTIME_ERROR(cudaq_bridge_launch(resources.bridge));
    std::cout << "=== App Ready ===" << std::endl;
    // This clock bounds receiver lifetime; playback measures request latency.
    const auto start = std::chrono::steady_clock::now();
    while (!interrupted && std::chrono::steady_clock::now() - start <
                               std::chrono::seconds(config.timeout))
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "\n=== Shutting down ===" << std::endl;
    // 5. Stop before reading the final dispatch count. Explicit close reports
    // cleanup errors to main; the destructor also handles exceptional exits.
    resources.stop();
    if (resources.dispatcher) {
      uint64_t processed = 0;
      HANDLE_REALTIME_ERROR(
          cudaq_dispatcher_get_processed(resources.dispatcher, &processed));
      std::cout << "Packets dispatched: " << processed << std::endl;
    }
    resources.close();
    return 0;
  } catch (const cudaq::realtime::validation::UnsupportedConfiguration &error) {
    std::cerr << "UNSUPPORTED: " << error.what() << std::endl;
    return 4;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 1;
  }
}
