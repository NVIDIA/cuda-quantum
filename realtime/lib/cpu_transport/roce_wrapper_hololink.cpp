/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// C ABI shim over Hololink's CpuRoceTransceiver (find_package(HololinkRoce)),
// mirroring roce_wrapper.cpp. The legacy CUDA-Q implementation remains
// selectable at configure time; exactly one implementation of the cpu_roce_*
// symbols is linked into a build.

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
  std::size_t page_size = 0;
  std::uint32_t num_pages = 0;
  std::uint64_t peer_rx_base = 0;
  std::unique_ptr<Transceiver> transceiver;
};

HololinkCpuRoceAdapter *as_adapter(cpu_roce_transceiver_t handle) {
  return static_cast<HololinkCpuRoceAdapter *>(handle);
}

// unified=1 is CUDA-Q's thread-free mode (cpu_roce_rx_poll /
// cpu_roce_tx_publish).  Hololink's Mode::Unified is a different thing -- its
// own loop around a per-slot callback -- and Hololink exposes no thread-free
// poll/post, so the mode is refused rather than approximated.
Transceiver::Mode select_mode(int forward, int rx_only, int tx_only,
                              int unified) {
  if ((forward != 0) + (rx_only != 0) + (tx_only != 0) + (unified != 0) > 1)
    throw std::invalid_argument(
        "forward / rx_only / tx_only / unified are mutually exclusive");
  if (unified)
    throw std::invalid_argument("unified (thread-free) mode is not supported "
                                "by the Hololink CPU RoCE backend");
  if (forward)
    return Transceiver::Mode::Forward;
  if (rx_only)
    return Transceiver::Mode::Rx;
  if (tx_only)
    return Transceiver::Mode::Tx;
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
    config.rx.frame_layout.payload_size = frame_size;
    config.tx.frame_layout.payload_size = frame_size;

    // CUDA-Q already requires matching peer geometry, so the peer's slot
    // pitch and depth are the local ones until connect() learns otherwise.
    config.peer.qp = tx_ibv_qp;
    config.peer.ip = peer_ip;
    config.peer.rx_base = peer_rx_base_addr;
    config.peer.rx_rkey = peer_rx_rkey;
    config.peer.rx_stride = page_size;
    config.peer.rx_slot_count = num_pages;

    auto adapter = std::make_unique<HololinkCpuRoceAdapter>();
    adapter->page_size = page_size;
    adapter->num_pages = num_pages;
    adapter->peer_rx_base = peer_rx_base_addr;
    adapter->transceiver = std::make_unique<Transceiver>(std::move(config));
    return adapter.release();
  } catch (const std::exception &error) {
    report_failure("cpu_roce_create_transceiver", error);
    return nullptr;
  }
}

void cpu_roce_destroy_transceiver(cpu_roce_transceiver_t handle) {
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
    // A non-std exception would otherwise unwind through this extern "C"
    // frame into CUDA-Q's monitor thread.
    report_monitor_stall("unknown monitor failure");
  }
}

// Unified mode is refused at create (see select_mode), so no handle ever
// reaches these in that mode.
int cpu_roce_rx_poll(cpu_roce_transceiver_t, std::uint32_t *) { return 0; }

int cpu_roce_tx_publish(cpu_roce_transceiver_t, std::uint32_t) { return 0; }

void cpu_roce_set_local_ip(cpu_roce_transceiver_t handle,
                           const char *local_ip) {
  if (!handle)
    return;
  try {
    as_adapter(handle)->transceiver->set_local_ip(local_ip ? local_ip : "");
  } catch (const std::exception &error) {
    // Hololink rejects a malformed address before storing it, so Config keeps
    // the previously configured local_ip (empty until a successful setter).
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
