/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// C ABI shim over CpuRoceTransceiver.  Translates the typed handle to a
// `void *` and bridges the C bool-as-int / enum-as-int convention to the
// underlying C++ types.  All entry points are noexcept at the C boundary;
// std::exceptions from the C++ side are caught and reported to stderr.

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"

#include "cudaq/realtime/cpu_transport/roce_transceiver.hpp"

#include <cstdio>
#include <exception>

using cudaq::realtime::bridge::CpuRoceTransceiver;
using cudaq::realtime::bridge::CpuRoceTxMode;

namespace {
CpuRoceTransceiver *as_cpp(cpu_roce_transceiver_t h) {
  return static_cast<CpuRoceTransceiver *>(h);
}
} // namespace

extern "C" {

cpu_roce_transceiver_t cpu_roce_create_transceiver(
    const char *device_name, int ib_port, unsigned tx_ibv_qp,
    size_t frame_size, size_t page_size, unsigned num_pages,
    const char *peer_ip, int forward, int rx_only, int tx_only, int unified,
    cpu_roce_tx_mode_t tx_mode, uint64_t peer_rx_base_addr,
    uint32_t peer_rx_rkey) {
  try {
    // Translate the C enum to the C++ enum.  An out-of-range value falls
    // through to the FPGA default — the constructor will validate further.
    const CpuRoceTxMode cpp_mode =
        (tx_mode == CPU_ROCE_TX_MODE_WRITE_WITH_IMM_FOR_PEER)
            ? CpuRoceTxMode::kWriteWithImmForPeer
            : CpuRoceTxMode::kSendForFpga;
    auto *t = new CpuRoceTransceiver(
        device_name, static_cast<unsigned>(ib_port), tx_ibv_qp, frame_size,
        page_size, num_pages, peer_ip, forward != 0, rx_only != 0,
        tx_only != 0, unified != 0, cpp_mode, peer_rx_base_addr, peer_rx_rkey);
    return t;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "cpu_roce_create_transceiver: %s\n", e.what());
    return nullptr;
  }
}

void cpu_roce_destroy_transceiver(cpu_roce_transceiver_t handle) {
  delete as_cpp(handle);
}

int cpu_roce_start(cpu_roce_transceiver_t handle) {
  if (!handle)
    return 0;
  try {
    return as_cpp(handle)->start() ? 1 : 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "cpu_roce_start: %s\n", e.what());
    return 0;
  }
}

void cpu_roce_close(cpu_roce_transceiver_t handle) {
  if (handle)
    as_cpp(handle)->close();
}

void cpu_roce_blocking_monitor(cpu_roce_transceiver_t handle) {
  if (!handle)
    return;
  try {
    as_cpp(handle)->blocking_monitor();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "cpu_roce_blocking_monitor: %s\n", e.what());
  }
}

void cpu_roce_set_unified_dispatch(cpu_roce_transceiver_t handle,
                                   cpu_roce_unified_dispatch_fn_t fn,
                                   void *context) {
  if (!handle)
    return;
  // The C and C++ function pointer types have identical signatures
  // (size_t/std::size_t are the same; void* are the same) so a direct
  // reinterpret is safe.
  as_cpp(handle)->set_unified_dispatch(
      reinterpret_cast<CpuRoceTransceiver::UnifiedDispatchFn>(fn), context);
}

uint32_t cpu_roce_get_qp_number(cpu_roce_transceiver_t handle) {
  return handle ? as_cpp(handle)->get_qp_number() : 0;
}

uint32_t cpu_roce_get_rkey(cpu_roce_transceiver_t handle) {
  return handle ? as_cpp(handle)->get_rkey() : 0;
}

uint64_t cpu_roce_get_buffer_addr(cpu_roce_transceiver_t handle) {
  return handle ? as_cpp(handle)->external_frame_memory() : 0;
}

void *cpu_roce_get_rx_ring_data_addr(cpu_roce_transceiver_t handle) {
  return handle ? static_cast<void *>(as_cpp(handle)->get_rx_ring_data_addr())
                : nullptr;
}

uint64_t *cpu_roce_get_rx_ring_flag_addr(cpu_roce_transceiver_t handle) {
  return handle ? as_cpp(handle)->get_rx_ring_flag_addr() : nullptr;
}

void *cpu_roce_get_tx_ring_data_addr(cpu_roce_transceiver_t handle) {
  return handle ? static_cast<void *>(as_cpp(handle)->get_tx_ring_data_addr())
                : nullptr;
}

uint64_t *cpu_roce_get_tx_ring_flag_addr(cpu_roce_transceiver_t handle) {
  return handle ? as_cpp(handle)->get_tx_ring_flag_addr() : nullptr;
}

size_t cpu_roce_get_page_size(cpu_roce_transceiver_t handle) {
  return handle ? as_cpp(handle)->get_rx_ring_stride_sz() : 0;
}

unsigned cpu_roce_get_num_pages(cpu_roce_transceiver_t handle) {
  return handle ? as_cpp(handle)->get_rx_ring_stride_num() : 0;
}

} // extern "C"
