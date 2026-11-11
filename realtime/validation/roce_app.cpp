/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file roce_app.cpp
/// @brief CPU/GPU RoCE receiver for increment and native-forward validation.
///
/// Translate RoCE command-line settings into CPU or GPU bridge options, load
/// that bridge, and attach the increment handler to the selected dispatcher.
/// Forward mode uses the bridge's echo path without a dispatcher or handler.
/// The orchestration script and playback tool configure the producer and check
/// replies; this process serves requests until signaled or its timeout expires.
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
#include <vector>

extern "C" void
setup_rpc_increment_function_table_host(cudaq_function_entry_t *entries);
extern "C" void
setup_rpc_increment_function_table(cudaq_function_entry_t *entries);
extern "C" cudaq_status_t
setup_rpc_increment_function_table_linked(cudaq_function_entry_t *entries,
                                          cudaq_dispatcher_t *dispatcher);

namespace {
// Receiver and RoCE settings. parse_args() validates the common geometry;
// the selected bridge validates its own transport-specific requirements.
struct Config {
  bool cpu = false;     ///< Select host dispatch and the CPU RoCE bridge.
  bool unified = false; ///< Combine receive, dispatch and transmit execution.
  bool forward = false; ///< Echo through the bridge without invoking an RPC.
  int gpu = 0;          ///< CUDA device ordinal for GPU execution.
  int timeout = 60;     ///< Seconds to serve requests after App Ready.
  unsigned threads = 32; ///< Threads per GPU dispatch block.
  std::string device = "mlx5_0"; ///< RDMA device connected to the producer.
  std::string peer_ip = "192.168.0.2"; ///< Producer's data-plane IP address.
  unsigned remote_qp = 2;             ///< Producer's RoCE queue-pair number.
  unsigned num_pages = 128;           ///< Number of receive/transmit ring slots.
  std::size_t page_size = 384; ///< Slot stride in bytes, including header space.
  unsigned payload_size = 8;  ///< RPC argument bytes, excluding the RPC header.
  std::vector<std::string> gpu_options; ///< GPU QP-exchange options to forward.
};

// Return false after printing help; invalid options throw before bridge setup.
bool parse_args(int argument_count, char **arguments, Config &config) {
  bool cpu_selected = false, gpu_selected = false;
  for (int argument_index = 1; argument_index < argument_count;
       ++argument_index) {
    const std::string argument = arguments[argument_index];
    // Value options use --name=value; the shell runner accepts either spelling
    // and supplies this form to the executable.
    const auto equals_position = argument.find('=');
    const auto key = argument.substr(0, equals_position);
    const auto value = equals_position == std::string::npos
                           ? ""
                           : argument.substr(equals_position + 1);
    if (argument == "--help" || argument == "-h") {
      std::cout
          << "Usage: roce_app [--cpu | --gpu] [options]\n"
          << "  --cpu / --gpu        Host or device dispatch (default: GPU)\n"
          << "  --gpu-id=N           GPU device ID (default: 0; --gpu=N also "
             "accepted)\n"
          << "  --device=NAME        RDMA device (default: mlx5_0)\n"
          << "  --peer-ip=ADDR       Producer IP (default: 192.168.0.2)\n"
          << "  --remote-qp=N        Producer QP (default: 2)\n"
          << "  --num-pages=N        Ring slots, power of two (default: 128)\n"
          << "  --page-size=N        Slot stride in bytes (default: 384)\n"
          << "  --payload-size=N     RPC payload bytes (default: 8)\n"
          << "  --timeout=N          Run timeout in seconds (default: 60)\n"
          << "  --threads=N          GPU dispatch block size (default: 32)\n"
          << "  --forward            Native bridge echo, no RPC dispatch\n"
          << "  --unified            Combined RX/dispatch/TX thread or kernel\n"
          << "                       Default: separate RX/dispatch/TX\n"
          << "  --exchange-qp        GPU provider TCP QP exchange\n"
          << "  --exchange-port=N    GPU QP exchange port (default: 12345)\n";
      return false;
    } else if (argument == "--cpu") {
      cpu_selected = config.cpu = true;
    } else if (key == "--gpu") {
      gpu_selected = true;
      if (equals_position != std::string::npos)
        config.gpu = std::stoi(value);
    } else if (key == "--gpu-id")
      config.gpu = std::stoi(value);
    else if (key == "--device")
      config.device = value;
    else if (key == "--peer-ip")
      config.peer_ip = value;
    else if (key == "--remote-qp")
      config.remote_qp = std::stoul(value, nullptr, 0);
    else if (key == "--num-pages")
      config.num_pages = std::stoul(value);
    else if (key == "--page-size")
      config.page_size = std::stoull(value);
    else if (key == "--payload-size")
      config.payload_size = std::stoul(value);
    else if (key == "--timeout")
      config.timeout = std::stoi(value);
    else if (key == "--threads")
      config.threads = std::stoul(value);
    else if (argument == "--unified")
      config.unified = true;
    else if (argument == "--forward")
      config.forward = true;
    else if (argument == "--exchange-qp" || key == "--exchange-port")
      config.gpu_options.push_back(argument);
    else
      throw std::runtime_error("Unknown argument: " + argument);
  }
  // Reject conflicting execution modes and frames that cannot fit the ring
  // before creating CUDA or RDMA resources.
  if (cpu_selected && gpu_selected)
    throw std::runtime_error("--cpu and --gpu are mutually exclusive");
  if (config.forward && config.unified)
    throw std::runtime_error("--forward and --unified are mutually exclusive");
  if (config.cpu && !config.gpu_options.empty())
    throw std::runtime_error("TCP QP exchange options require --gpu");
  if (config.timeout <= 0 || config.gpu < 0 || config.threads == 0 ||
      config.threads > 1024)
    throw std::runtime_error("Invalid timeout, GPU ID, or thread count");
  if (!config.num_pages || (config.num_pages & (config.num_pages - 1)) ||
      config.num_pages > 4096 ||
      config.page_size <
          sizeof(cudaq::realtime::RPCHeader) + config.payload_size)
    throw std::runtime_error(
        "Invalid ring geometry or payload exceeds slot size");
  return true;
}

// Signal handlers only request a stop; cleanup stays on the main thread.
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

// Tables and control storage must outlive the dispatcher and bridge, including
// partial initialization. CPU mode never allocates/frees CUDA memory.
struct Resources {
  cudaq_realtime_bridge_handle_t bridge = nullptr; ///< Owns transport resources.
  cudaq_dispatch_manager_t *manager = nullptr; ///< Creates/manages the dispatcher.
  cudaq_dispatcher_t *dispatcher = nullptr;    ///< Absent in native forward mode.
  volatile int host_shutdown = 0; ///< Stop flag read by the CPU dispatcher.
  uint64_t host_stats = 0;        ///< CPU dispatcher's processed-request count.
  cudaq_function_entry_t host_entries[1]{}; ///< Host increment handler entry.
  int *mapped_shutdown = nullptr; ///< Host allocation also readable by the GPU.
  uint64_t *device_stats = nullptr; ///< GPU allocation for the processed count.
  cudaq_function_entry_t *device_entries = nullptr; ///< GPU handler table storage.

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

  ~Resources() {
    // Stop dispatch before releasing the bridge's rings, then free the table
    // and control storage. Null members also allow cleanup after partial setup.
    try {
      stop();
    } catch (const std::exception &error) {
      std::cerr << "Shutdown error: " << error.what() << std::endl;
    }
    if (dispatcher)
      cudaq_dispatcher_destroy(dispatcher);
    if (manager)
      cudaq_dispatch_manager_destroy(manager);
    if (bridge) {
      cudaq_bridge_disconnect(bridge);
      cudaq_bridge_destroy(bridge);
    }
    if (device_entries)
      cudaFree(device_entries);
    if (device_stats)
      cudaFree(device_stats);
    if (mapped_shutdown)
      cudaFreeHost(mapped_shutdown);
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

    // 1. Translate the shared RoCE settings into the selected bridge's CLI.
    // CPU frame size includes the RPC header; the GPU bridge takes payload size.
    std::vector<std::string> options = {
        "--device=" + config.device, "--peer-ip=" + config.peer_ip,
        "--remote-qp=" + std::to_string(config.remote_qp)};
    if (config.cpu) {
      options.emplace_back("--qp_config=hsb_fpga");
      options.emplace_back("--num-slots=" + std::to_string(config.num_pages));
      options.emplace_back("--slot-size=" + std::to_string(config.page_size));
      options.emplace_back("--frame-size=" +
                           std::to_string(sizeof(cudaq::realtime::RPCHeader) +
                                          config.payload_size));
    } else {
      HANDLE_CUDA_ERROR(cudaSetDevice(config.gpu));
      options.emplace_back("--gpu=" + std::to_string(config.gpu));
      options.emplace_back("--timeout=" + std::to_string(config.timeout));
      options.emplace_back("--num-pages=" + std::to_string(config.num_pages));
      options.emplace_back("--page-size=" + std::to_string(config.page_size));
      options.emplace_back("--payload-size=" +
                           std::to_string(config.payload_size));
      options.emplace_back("--threads=" + std::to_string(config.threads));
      options.insert(options.end(), config.gpu_options.begin(),
                     config.gpu_options.end());
    }
    if (config.unified)
      options.emplace_back("--unified");
    if (config.forward)
      options.emplace_back("--forward");
    // Keep the strings alive and unchanged while create() reads these pointers.
    std::vector<char *> bridge_args;
    for (auto &option : options)
      bridge_args.push_back(option.data());

    // 2. Load the bridge and establish the receiver side of the connection.
    // The HSB runner obtains QP/rkey/address from the bridge's startup output
    // and passes them to the playback tool for producer configuration.
    const char *library = config.cpu ? "libcudaq-realtime-bridge-cpu-roce.so"
                                     : "libcudaq-realtime-bridge-gpu-roce.so";
    std::cout << "=== RoCE Validation App (" << (config.cpu ? "CPU" : "GPU")
              << ") ===\nLoading " << library << std::endl;
    HANDLE_REALTIME_ERROR(cudaq_bridge_create_from_library(
        &resources.bridge, library, static_cast<int>(bridge_args.size()),
        bridge_args.data()));
    HANDLE_REALTIME_ERROR(cudaq_bridge_connect(resources.bridge));

    // 3. Attach a dispatcher and function table for increment requests.
    // Native forwarding needs only the bridge and skips this entire setup.
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

      // The single entry maps rpc_increment to a host or device handler.
      // Control pointers must be accessible from the same execution path.
      cudaq_function_table_t table{};
      table.count = 1;
      volatile int *shutdown = &resources.host_shutdown;
      uint64_t *stats = &resources.host_stats;
      if (config.cpu) {
        // CPU dispatch uses ordinary host memory and makes no CUDA allocations.
        setup_rpc_increment_function_table_host(resources.host_entries);
        table.entries = resources.host_entries;
      } else {
        // A mapped stop flag lets the host stop persistent GPU kernels without
        // first waiting for a CUDA copy. Counters and handlers live on device.
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
        // Unified GPU dispatch links its module at runtime, so resolve the
        // handler address from that module. Separate dispatch uses this
        // executable's device-linked handler via the setup kernel.
        if (config.unified)
          HANDLE_REALTIME_ERROR(setup_rpc_increment_function_table_linked(
              resources.device_entries, resources.dispatcher));
        else {
          setup_rpc_increment_function_table(resources.device_entries);
          HANDLE_CUDA_ERROR(cudaGetLastError());
        }
        table.entries = resources.device_entries;
      }
      // Register the table with dispatch and any bridge that needs it; the
      // bridge API treats an absent registration callback as a successful no-op.
      HANDLE_REALTIME_ERROR(
          cudaq_dispatcher_set_function_table(resources.dispatcher, &table));
      HANDLE_REALTIME_ERROR(
          cudaq_bridge_set_function_table(resources.bridge, &table));
      // Control storage and handler entries stay owned by Resources for the
      // entire run; the dispatcher receives pointers, not independent copies.
      HANDLE_REALTIME_ERROR(
          cudaq_dispatcher_set_control(resources.dispatcher, shutdown, stats));
      HANDLE_REALTIME_ERROR(cudaq_dispatcher_start(resources.dispatcher));
    }
    // 4. Launch transport work after dispatch is ready. The shell runner waits
    // for this exact marker before it starts producer playback.
    HANDLE_REALTIME_ERROR(cudaq_bridge_launch(resources.bridge));
    std::cout << "=== App Ready ===" << std::endl;
    // The runner normally signals us when playback finishes. The timeout also
    // bounds service if no signal arrives; this loop does not measure latency.
    const auto start = std::chrono::steady_clock::now();
    while (!interrupted && std::chrono::steady_clock::now() - start <
                               std::chrono::seconds(config.timeout))
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 5. Stop dispatch before reading its final count. Resources then releases
    // the bridge and memory as the scope exits. Forward mode has no RPC count.
    std::cout << "\n=== Shutting down ===" << std::endl;
    resources.stop();
    if (resources.dispatcher) {
      uint64_t processed = 0;
      HANDLE_REALTIME_ERROR(
          cudaq_dispatcher_get_processed(resources.dispatcher, &processed));
      std::cout << "Packets dispatched: " << processed << std::endl;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 1;
  }
}
