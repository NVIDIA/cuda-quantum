/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// C ABI shim over the installed Hololink CPU RoCE transport, mirroring
// roce_wrapper.cpp.  The legacy CUDA-Q implementation remains selectable at
// configure time; exactly one implementation of the cpu_roce_* symbols is
// linked into a build.

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"

#include <hololink/transport/roce/cpu_roce_transceiver.hpp>

#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

namespace roce = hololink::transport::roce;
using Transceiver = roce::CpuRoceTransceiver;

// Hololink's ring accessors report 0 until setup(), so the adapter keeps the
// create() geometry that the C accessors must return immediately.
struct HololinkCpuRoceAdapter {
  Transceiver::Mode mode = Transceiver::Mode::Duplex;
  std::size_t page_size = 0;
  std::uint32_t num_pages = 0;
  std::uint64_t peer_rx_base = 0;
  // Set before blocking_monitor(); concurrent update and callback execution
  // are unsupported.
  cpu_roce_unified_dispatch_fn_t unified_dispatch = nullptr;
  void *unified_context = nullptr;
  // Declared last so ~CpuRoceTransceiver() waits for the monitor while the
  // fields above are still alive.
  std::unique_ptr<Transceiver> transceiver;
};

HololinkCpuRoceAdapter *as_adapter(cpu_roce_transceiver_t handle) {
  return static_cast<HololinkCpuRoceAdapter *>(handle);
}

// CUDA-Q receives Hololink's tx_capacity as slot_size; zero means Drop.
Transceiver::UnifiedDispatchResult
unified_dispatch_thunk(void *opaque, const void *rx_slot, std::size_t,
                       void *tx_slot, std::size_t tx_capacity) {
  auto *adapter = static_cast<HololinkCpuRoceAdapter *>(opaque);
  if (!adapter->unified_dispatch)
    return {Transceiver::UnifiedDispatchDisposition::Drop, 0};
  const std::size_t bytes = adapter->unified_dispatch(
      adapter->unified_context, rx_slot, tx_slot, tx_capacity);
  return {bytes == 0 ? Transceiver::UnifiedDispatchDisposition::Drop
                     : Transceiver::UnifiedDispatchDisposition::Send,
          bytes};
}

Transceiver::Mode select_mode(int forward, int rx_only, int tx_only,
                              int unified) {
  if ((forward != 0) + (rx_only != 0) + (tx_only != 0) + (unified != 0) > 1)
    throw std::invalid_argument(
        "forward / rx_only / tx_only / unified are mutually exclusive");
  if (forward)
    return Transceiver::Mode::Forward;
  if (rx_only)
    return Transceiver::Mode::Rx;
  if (tx_only)
    return Transceiver::Mode::Tx;
  if (unified)
    return Transceiver::Mode::Unified;
  return Transceiver::Mode::Duplex;
}

void report_failure(const char *operation, const std::exception &error) {
  std::fprintf(stderr, "%s: %s\n", operation, error.what());
}

void report_monitor_stall(const char *detail) {
  std::fprintf(stderr,
               "cpu_roce_blocking_monitor: rings will no longer advance: %s\n",
               detail);
}

} // namespace

extern "C" {

cpu_roce_transceiver_t cpu_roce_create_transceiver(
    const char *device_name, int ib_port, unsigned tx_ibv_qp,
    std::size_t frame_size, std::size_t page_size, unsigned num_pages,
    const char *peer_ip, int forward, int rx_only, int tx_only, int unified,
    cpu_roce_tx_mode_t tx_mode, std::uint64_t peer_rx_base_addr,
    std::uint32_t peer_rx_rkey) {
  try {
    // Hololink's constructor defers validation to setup(), so repeat the
    // legacy constructor's checks to keep create() failing on the same
    // arguments.  The null checks also guard std::string assignment.
    if (!device_name || !peer_ip)
      throw std::invalid_argument("device_name and peer_ip must be non-null");
    if (frame_size == 0 || frame_size > page_size)
      throw std::invalid_argument(
          "frame_size must be non-zero and <= page_size");
    if (num_pages == 0 || (num_pages & (num_pages - 1)) != 0)
      throw std::invalid_argument("num_pages must be a non-zero power of two");

    Transceiver::Config config;
    config.ibv_name = device_name;
    config.ibv_port = static_cast<std::uint32_t>(ib_port);
    config.mode = select_mode(forward, rx_only, tx_only, unified);
    // As in the legacy wrapper, any value other than the exact WriteWithImm
    // enumerator selects Send.
    config.tx_operation = tx_mode == CPU_ROCE_TX_MODE_RDMA_WRITE_WITH_IMM
                              ? Transceiver::TxOperation::WriteWithImm
                              : Transceiver::TxOperation::Send;

    config.rx.stride = page_size;
    config.rx.slot_count = num_pages;
    config.tx.stride = page_size;
    config.tx.slot_count = num_pages;
    // A Unified callback owns the whole slot and may return that many bytes;
    // the other modes keep the legacy cu_frame_size SGE length.
    const std::size_t payload_size =
        config.mode == Transceiver::Mode::Unified ? page_size : frame_size;
    config.rx.frame_layout.payload_size = payload_size;
    config.tx.frame_layout.payload_size = payload_size;

    // CUDA-Q already requires matching peer geometry, so the peer's slot
    // pitch and depth are the local ones until connect() learns otherwise.
    config.peer.qp = tx_ibv_qp;
    config.peer.ip = peer_ip;
    config.peer.rx_base = peer_rx_base_addr;
    config.peer.rx_rkey = peer_rx_rkey;
    config.peer.rx_stride = page_size;
    config.peer.rx_slot_count = num_pages;

    auto adapter = std::make_unique<HololinkCpuRoceAdapter>();
    adapter->mode = config.mode;
    adapter->page_size = page_size;
    adapter->num_pages = num_pages;
    adapter->peer_rx_base = peer_rx_base_addr;
    adapter->transceiver = std::make_unique<Transceiver>(std::move(config));
    // Unified requires a callback before setup(), but the C caller may
    // install the real one later; the thunk reads it at dispatch time.
    if (adapter->mode == Transceiver::Mode::Unified)
      adapter->transceiver->set_unified_dispatch(&unified_dispatch_thunk,
                                                 adapter.get());
    return adapter.release();
  } catch (const std::exception &error) {
    report_failure("cpu_roce_create_transceiver", error);
    return nullptr;
  }
}

void cpu_roce_destroy_transceiver(cpu_roce_transceiver_t handle) {
  // The transceiver is the last adapter member, so it is destroyed first and
  // ~CpuRoceTransceiver() waits for the monitor to leave blocking_monitor()
  // before the Unified thunk's fields go away.
  delete as_adapter(handle);
}

int cpu_roce_start(cpu_roce_transceiver_t handle) {
  if (!handle)
    return 0;
  try {
    as_adapter(handle)->transceiver->start();
    return 1;
  } catch (const std::exception &error) {
    report_failure("cpu_roce_start", error);
    return 0;
  }
}

int cpu_roce_setup(cpu_roce_transceiver_t handle) {
  if (!handle)
    return 0;
  try {
    as_adapter(handle)->transceiver->setup();
    return 1;
  } catch (const std::exception &error) {
    report_failure("cpu_roce_setup", error);
    return 0;
  }
}

int cpu_roce_connect(cpu_roce_transceiver_t handle, unsigned peer_qp,
                     const char *peer_ip, std::uint32_t peer_rx_rkey) {
  if (!handle)
    return 0;
  auto *adapter = as_adapter(handle);
  try {
    if (!peer_ip)
      throw std::invalid_argument("peer_ip is null");
    Transceiver::Peer peer;
    peer.qp = peer_qp;
    peer.ip = peer_ip;
    peer.rx_base = adapter->peer_rx_base;
    peer.rx_rkey = peer_rx_rkey;
    // The C ABI carries no peer geometry; preserve legacy's assumption that
    // it matches local page_size and num_pages.
    peer.rx_stride = adapter->page_size;
    peer.rx_slot_count = adapter->num_pages;
    adapter->transceiver->connect(std::move(peer));
    return 1;
  } catch (const std::exception &error) {
    report_failure("cpu_roce_connect", error);
    return 0;
  }
}

void cpu_roce_close(cpu_roce_transceiver_t handle) {
  if (!handle)
    return;
  try {
    as_adapter(handle)->transceiver->close();
  } catch (const std::exception &error) {
    // Prevent Hololink's self-close error from crossing the C boundary.
    report_failure("cpu_roce_close", error);
  }
}

void cpu_roce_blocking_monitor(cpu_roce_transceiver_t handle) {
  if (!handle)
    return;
  // A Hololink terminal failure unwinds the monitor instead of returning, and
  // the void C API cannot report it, so callers still detect the stall via
  // their session timeout.
  try {
    as_adapter(handle)->transceiver->blocking_monitor();
  } catch (const std::exception &error) {
    report_monitor_stall(error.what());
  } catch (...) {
    // A Unified callback throwing a non-std exception would otherwise unwind
    // through this extern "C" frame into CUDA-Q's monitor thread.
    report_monitor_stall("unknown monitor failure");
  }
}

void cpu_roce_set_unified_dispatch(cpu_roce_transceiver_t handle,
                                   cpu_roce_unified_dispatch_fn_t fn,
                                   void *context) {
  if (!handle)
    return;
  auto *adapter = as_adapter(handle);
  // The thunk is registered with Hololink only in Unified mode.
  if (adapter->mode != Transceiver::Mode::Unified)
    return;
  adapter->unified_dispatch = fn;
  adapter->unified_context = context;
}

void cpu_roce_set_local_ip(cpu_roce_transceiver_t handle,
                           const char *local_ip) {
  if (!handle)
    return;
  try {
    as_adapter(handle)->transceiver->set_local_ip(local_ip ? local_ip : "");
  } catch (const std::exception &error) {
    // Hololink rejects a malformed address before storing it, so Config keeps
    // the default empty local_ip and later GID selection matches legacy.
    report_failure("cpu_roce_set_local_ip", error);
  }
}

std::uint32_t cpu_roce_get_qp_number(cpu_roce_transceiver_t handle) {
  return handle ? as_adapter(handle)->transceiver->get_qp_number() : 0;
}

std::uint32_t cpu_roce_get_rkey(cpu_roce_transceiver_t handle) {
  return handle ? as_adapter(handle)->transceiver->get_rkey() : 0;
}

std::uint64_t cpu_roce_get_buffer_addr(cpu_roce_transceiver_t handle) {
  return handle ? as_adapter(handle)->transceiver->external_frame_memory() : 0;
}

void *cpu_roce_get_rx_ring_data_addr(cpu_roce_transceiver_t handle) {
  return handle ? static_cast<void *>(
                      as_adapter(handle)->transceiver->get_rx_ring_data_addr())
                : nullptr;
}

std::uint64_t *cpu_roce_get_rx_ring_flag_addr(cpu_roce_transceiver_t handle) {
  return handle ? as_adapter(handle)->transceiver->get_rx_ring_flag_addr()
                : nullptr;
}

void *cpu_roce_get_tx_ring_data_addr(cpu_roce_transceiver_t handle) {
  return handle ? static_cast<void *>(
                      as_adapter(handle)->transceiver->get_tx_ring_data_addr())
                : nullptr;
}

std::uint64_t *cpu_roce_get_tx_ring_flag_addr(cpu_roce_transceiver_t handle) {
  return handle ? as_adapter(handle)->transceiver->get_tx_ring_flag_addr()
                : nullptr;
}

std::size_t cpu_roce_get_page_size(cpu_roce_transceiver_t handle) {
  return handle ? as_adapter(handle)->page_size : 0;
}

unsigned cpu_roce_get_num_pages(cpu_roce_transceiver_t handle) {
  return handle ? as_adapter(handle)->num_pages : 0;
}

} // extern "C"
