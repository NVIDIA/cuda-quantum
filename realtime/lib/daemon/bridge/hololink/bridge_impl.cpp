/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "hololink_wrapper.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
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
  HololinkBridgeContext(const cudaq::realtime::BridgeConfig &cfg)
      : config(cfg) {
    //============================================================================
    // [1] Initialize CUDA
    //============================================================================
    std::cout << "\n[1/5] Initializing CUDA..." << std::endl;
    HANDLE_CUDA_ERROR(cudaSetDevice(config.gpu_id));

    cudaDeviceProp prop;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, config.gpu_id));
    std::cout << "  GPU: " << prop.name << std::endl;

    //============================================================================
    // [2] Create Hololink transceiver
    //============================================================================
    std::cout << "\n[2/5] Creating Hololink transceiver..." << std::endl;

    // Ensure page_size >= frame_size
    if (config.page_size < config.frame_size) {
      std::cout << "  Adjusting page_size from " << config.page_size << " to "
                << config.frame_size << " to fit frame" << std::endl;
      config.page_size = config.frame_size;
    }

    std::cout << "  Frame size: " << config.frame_size << " bytes" << std::endl;
    std::cout << "  Page size: " << config.page_size << " bytes" << std::endl;
    std::cout << "  Num pages: " << config.num_pages << std::endl;

    transceiver = hololink_create_transceiver(
        config.device.c_str(), 1, // ib_port
        config.frame_size, config.page_size, config.num_pages,
        "0.0.0.0", // deferred connection
        0,         // forward = false
        1,         // rx_only = true
        1          // tx_only = true
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

  // Frame size: RPCHeader + 256 bytes payload
  config.frame_size = sizeof(cudaq::realtime::RPCHeader) + 256;

  std::cout << "Device: " << config.device << std::endl;
  std::cout << "Peer IP: " << config.peer_ip << std::endl;
  std::cout << "Remote QP: 0x" << std::hex << config.remote_qp << std::dec
            << std::endl;
  std::cout << "GPU: " << config.gpu_id << std::endl;

  HololinkBridgeContext *ctx = new HololinkBridgeContext(config);
  if (!ctx) {
    std::cerr << "ERROR: Failed to create HololinkBridgeContext" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
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

static cudaq_status_t
hololink_bridge_get_ringbuffer(cudaq_realtime_bridge_handle_t handle,
                               cudaq_ringbuffer_t *out_ringbuffer) {

  if (!handle || !out_ringbuffer)
    return CUDAQ_ERR_INVALID_ARG;
  HololinkBridgeContext *ctx =
      reinterpret_cast<HololinkBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;

  auto &transceiver = ctx->transceiver;

  // Ring buffer pointers
  uint8_t *rx_ring_data =
      reinterpret_cast<uint8_t *>(hololink_get_rx_ring_data_addr(transceiver));
  uint64_t *rx_ring_flag = hololink_get_rx_ring_flag_addr(transceiver);
  uint8_t *tx_ring_data =
      reinterpret_cast<uint8_t *>(hololink_get_tx_ring_data_addr(transceiver));
  uint64_t *tx_ring_flag = hololink_get_tx_ring_flag_addr(transceiver);

  if (!rx_ring_data || !rx_ring_flag || !tx_ring_data || !tx_ring_flag) {
    std::cerr << "ERROR: Failed to get ring buffer pointers" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  out_ringbuffer->rx_flags =
      reinterpret_cast<volatile uint64_t *>(rx_ring_flag);
  out_ringbuffer->tx_flags =
      reinterpret_cast<volatile uint64_t *>(tx_ring_flag);
  out_ringbuffer->rx_data = rx_ring_data;
  out_ringbuffer->tx_data = tx_ring_data;
  out_ringbuffer->rx_stride_sz = ctx->config.page_size;
  out_ringbuffer->tx_stride_sz = ctx->config.page_size;

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
  // Connect QP to remote peer
  {
    uint8_t remote_gid[16] = {};
    remote_gid[10] = 0xff;
    remote_gid[11] = 0xff;
    inet_pton(AF_INET, ctx->config.peer_ip.c_str(), &remote_gid[12]);

    std::cout << "  Connecting QP to remote QP 0x" << std::hex
              << ctx->config.remote_qp << std::dec << " at "
              << ctx->config.peer_ip << "..." << std::endl;

    if (!hololink_reconnect_qp(transceiver, remote_gid,
                               ctx->config.remote_qp)) {
      std::cerr << "ERROR: Failed to connect QP to remote peer" << std::endl;
      return CUDAQ_ERR_INTERNAL;
    }
    std::cout << "  QP connected to remote peer" << std::endl;
  }

  uint32_t our_qp = hololink_get_qp_number(transceiver);
  uint32_t our_rkey = hololink_get_rkey(transceiver);
  uint64_t our_buffer = hololink_get_buffer_addr(transceiver);

  std::cout << "  QP Number: 0x" << std::hex << our_qp << std::dec << std::endl;
  std::cout << "  RKey: " << our_rkey << std::endl;
  std::cout << "  Buffer Addr: 0x" << std::hex << our_buffer << std::dec
            << std::endl;

  if (!hololink_query_kernel_occupancy()) {
    std::cerr << "ERROR: Hololink kernel occupancy query failed" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  

  // Print QP info for FPGA stimulus tool
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

  //============================================================================
  // Launch Hololink kernels and run
  //============================================================================
  std::cout << "\n[5/5] Launching Hololink kernels..." << std::endl;

  ctx->hololink_thread = std::make_unique<std::thread>(
      [transceiver]() { hololink_blocking_monitor(transceiver); });

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  std::cout << "  Hololink RX+TX kernels started" << std::endl;

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
      hololink_bridge_get_ringbuffer,
      hololink_bridge_connect,
      hololink_bridge_launch,
      hololink_bridge_disconnect,
  };
  return &cudaq_hololink_bridge_interface;
}
}
