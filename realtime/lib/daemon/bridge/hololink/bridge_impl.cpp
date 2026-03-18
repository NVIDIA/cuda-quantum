/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file bridge_impl.cpp
/// @brief Hololink bridge interface implementation for libcudaq-realtime
/// dispatch.

#include <sstream>

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/bridge/hololink/hololink_wrapper.h"
#include "cudaq/realtime/hololink_bridge_common.h"

namespace {
#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      std::stringstream ss;                                                    \
      ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "            \
         << cudaGetErrorString(err) << std::endl;                              \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  }

struct HololinkBridgeContext {
  cudaq::realtime::BridgeConfig config;
  hololink_transceiver_t transceiver = nullptr;
  std::unique_ptr<std::thread> hololink_thread;
  bool is_igpu = false;
  HololinkBridgeContext(const cudaq::realtime::BridgeConfig &cfg)
      : config(cfg) {
    //============================================================================
    // [1] Initialize CUDA
    //============================================================================
    HANDLE_CUDA_ERROR(cudaSetDevice(config.gpu_id));
    cudaDeviceProp prop;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, config.gpu_id));

    //============================================================================
    // [2] Create Hololink transceiver
    //============================================================================
    // Ensure page_size >= frame_size
    if (config.page_size < config.frame_size) {
      config.page_size = config.frame_size;
    }

    // On iGPU (e.g. DGX Spark GB10), DOCA NIC doorbells require a CPU proxy
    // thread.  Hololink's blocking_monitor() starts this thread alongside its
    // kernels.  For unified mode on iGPU we need the CPU proxy but NOT the
    // hololink kernels, so we pass (false, false, false) -- blocking_monitor
    // will start only the CPU proxy thread.  On dGPU, no CPU proxy is needed
    // and we use forward=true to get 64-deep receive pre-posting from start().
    is_igpu = (prop.integrated != 0);
    const bool unified_igpu = config.unified && is_igpu;

    bool use_forward = config.forward || (config.unified && !is_igpu);
    bool use_3kernel = !config.forward && !config.unified;

    transceiver = hololink_create_transceiver(
        config.device.c_str(), 1, // ib_port (FIXME: make configurable?)
        config.remote_qp,         // remote QP number
        config.gpu_id,            // GPU device ID
        config.frame_size, config.page_size, config.num_pages,
        config.peer_ip.c_str(), // immediate connection
        use_forward ? 1 : 0,    // forward
        use_3kernel ? 1 : 0,    // rx_only
        use_3kernel ? 1 : 0     // tx_only
    );
  }
};
} // namespace

extern "C" {
static cudaq_status_t
hololink_bridge_create(cudaq_realtime_bridge_handle_t *handle, int argc,
                       char **argv) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  cudaq::realtime::BridgeConfig config;

  // Parse common bridge args
  cudaq::realtime::parse_bridge_args(argc, argv, config);

  // Frame size: RPCHeader + payload size
  config.frame_size = sizeof(cudaq::realtime::RPCHeader) + config.payload_size;

  // Create and initialize the bridge context (including the Hololink
  // transceiver)
  HololinkBridgeContext *ctx = new HololinkBridgeContext(config);
  if (!ctx) {
    std::cerr << "ERROR: Failed to create HololinkBridgeContext" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  // Set the output handle to the created context (opaque to the caller)
  *handle = ctx;

  if (!ctx->transceiver) {
    std::cerr << "ERROR: Failed to create Hololink transceiver" << std::endl;
    delete ctx;
    return CUDAQ_ERR_INTERNAL;
  }

  if (!hololink_start(ctx->transceiver)) {
    std::cerr << "ERROR: Failed to start Hololink transceiver" << std::endl;
    hololink_destroy_transceiver(ctx->transceiver);
    delete ctx;
    return CUDAQ_ERR_INTERNAL;
  }

  // Hololink start() pops the CUDA context via cuCtxPopCurrent; restore it.
  HANDLE_CUDA_ERROR(cudaSetDevice(config.gpu_id));

  // On iGPU unified mode, start() didn't pre-post receive WQEs (transceiver
  // was created with forward=false, rx_only=false, tx_only=false).  Call the
  // prepare kernel here so the NIC has receive buffers before any packets
  // arrive.
  const bool unified_igpu = config.unified && ctx->is_igpu;
  if (unified_igpu) {
    if (!hololink_prepare_receive_send(ctx->transceiver, config.frame_size)) {
      std::cerr << "ERROR: Failed to pre-post receive WQEs" << std::endl;
      hololink_destroy_transceiver(ctx->transceiver);
      delete ctx;
      return CUDAQ_ERR_INTERNAL;
    }
  }
  return CUDAQ_OK;
}

static cudaq_status_t
hololink_bridge_destroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  HololinkBridgeContext *ctx =
      reinterpret_cast<HololinkBridgeContext *>(handle);
  if (ctx->transceiver) {
    hololink_destroy_transceiver(ctx->transceiver);
  }
  delete ctx;
  return CUDAQ_OK;
}

static cudaq_status_t hololink_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t handle,
    cudaq_realtime_transport_context_t context_type, void *out_context) {

  if (!handle || !out_context)
    return CUDAQ_ERR_INVALID_ARG;
  HololinkBridgeContext *ctx =
      reinterpret_cast<HololinkBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;

  auto &transceiver = ctx->transceiver;
  if (context_type == RING_BUFFER) {
    cudaq_ringbuffer_t *ringbuffer =
        reinterpret_cast<cudaq_ringbuffer_t *>(out_context);

    // Ring buffer pointers
    uint8_t *rx_ring_data = reinterpret_cast<uint8_t *>(
        hololink_get_rx_ring_data_addr(transceiver));
    uint64_t *rx_ring_flag = hololink_get_rx_ring_flag_addr(transceiver);
    uint8_t *tx_ring_data = reinterpret_cast<uint8_t *>(
        hololink_get_tx_ring_data_addr(transceiver));
    uint64_t *tx_ring_flag = hololink_get_tx_ring_flag_addr(transceiver);

    if (!rx_ring_data || !rx_ring_flag || !tx_ring_data || !tx_ring_flag) {
      std::cerr << "ERROR: Failed to get ring buffer pointers" << std::endl;
      return CUDAQ_ERR_INTERNAL;
    }

    ringbuffer->rx_flags = reinterpret_cast<volatile uint64_t *>(rx_ring_flag);
    ringbuffer->tx_flags = reinterpret_cast<volatile uint64_t *>(tx_ring_flag);
    ringbuffer->rx_data = rx_ring_data;
    ringbuffer->tx_data = tx_ring_data;
    ringbuffer->rx_stride_sz = ctx->config.page_size;
    ringbuffer->tx_stride_sz = ctx->config.page_size;
  } else if (context_type == UNIFIED) {
    cudaq_unified_dispatch_ctx_t *dispatch_ctx =
        reinterpret_cast<cudaq_unified_dispatch_ctx_t *>(out_context);

    static hololink_doca_transport_ctx doca_ctx{};
    doca_ctx.gpu_dev_qp = hololink_get_gpu_dev_qp(transceiver);
    doca_ctx.rx_ring_data = reinterpret_cast<uint8_t *>(
        hololink_get_rx_ring_data_addr(transceiver));
    doca_ctx.rx_ring_stride_sz = hololink_get_page_size(transceiver);
    doca_ctx.rx_ring_mkey = htonl(hololink_get_rkey(transceiver));
    doca_ctx.rx_ring_stride_num = hololink_get_num_pages(transceiver);
    doca_ctx.frame_size = ctx->config.frame_size;
    doca_ctx.use_bf = ctx->is_igpu ? 0 : 1;

    dispatch_ctx->launch_fn = &hololink_launch_unified_dispatch;
    dispatch_ctx->transport_ctx = &doca_ctx;
  } else {
    std::cerr << "ERROR: Invalid transport context type" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }

  return CUDAQ_OK;
}

static cudaq_status_t
hololink_bridge_connect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  HololinkBridgeContext *ctx =
      reinterpret_cast<HololinkBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;
  if (ctx->hololink_thread && ctx->hololink_thread->joinable()) {
    std::cerr << "ERROR: Hololink bridge already connected" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  auto &transceiver = ctx->transceiver;

  uint32_t our_qp = hololink_get_qp_number(transceiver);
  uint32_t our_rkey = hololink_get_rkey(transceiver);
  uint64_t our_buffer = hololink_get_buffer_addr(transceiver);

  // FIXME: Figure out a better way to share this info with the caller (e.g. via
  // output params or context struct) rather than printing to stdout. Print QP
  // info for FPGA stimulus tool
  std::cout << "\n=== Bridge Ready ===" << std::endl;
  std::cout << "  QP Number: 0x" << std::hex << our_qp << std::dec << std::endl;
  std::cout << "  RKey: " << our_rkey << std::endl;
  std::cout << "  Buffer Addr: 0x" << std::hex << our_buffer << std::dec
            << std::endl;
  std::cout << "\nWaiting for data (Ctrl+C to stop, timeout="
            << ctx->config.timeout_sec << "s)..." << std::endl;

  return CUDAQ_OK;
}

static cudaq_status_t
hololink_bridge_launch(cudaq_realtime_bridge_handle_t handle) {
  auto *ctx = reinterpret_cast<HololinkBridgeContext *>(handle);
  if (!ctx || !ctx->transceiver)
    return CUDAQ_ERR_INVALID_ARG;
  auto &transceiver = ctx->transceiver;
  const bool unified_igpu = ctx->config.unified && ctx->is_igpu;
  if (ctx->config.unified && !unified_igpu) {
    std::cout << "\n Unified mode -- no hololink monitor thread needed"
              << std::endl;
  } else if (unified_igpu) {
    std::cout << "\n Unified mode (iGPU) -- starting Hololink monitor "
              << "(CPU proxy only, no hololink kernels)" << std::endl;
    ctx->hololink_thread = std::make_unique<std::thread>(
        [transceiver]() { hololink_blocking_monitor(transceiver); });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  } else {
    //============================================================================
    // Launch Hololink kernels and run
    //============================================================================
    ctx->hololink_thread = std::make_unique<std::thread>(
        [transceiver]() { hololink_blocking_monitor(transceiver); });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  return CUDAQ_OK;
}

static cudaq_status_t
hololink_bridge_disconnect(cudaq_realtime_bridge_handle_t handle) {
  auto *ctx = reinterpret_cast<HololinkBridgeContext *>(handle);
  if (!ctx || !ctx->transceiver)
    return CUDAQ_ERR_INVALID_ARG;
  auto &transceiver = ctx->transceiver;
  hololink_close(transceiver);
  if (ctx->hololink_thread && ctx->hololink_thread->joinable())
    ctx->hololink_thread->join();
  return CUDAQ_OK;
}

cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface() {
  static cudaq_realtime_bridge_interface_t cudaq_hololink_bridge_interface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      hololink_bridge_create,
      hololink_bridge_destroy,
      hololink_bridge_get_transport_context,
      hololink_bridge_connect,
      hololink_bridge_launch,
      hololink_bridge_disconnect,
  };
  return &cudaq_hololink_bridge_interface;
}
}
