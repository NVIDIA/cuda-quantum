/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hololink_wrapper.cpp
/// @brief C wrapper implementation for Hololink GpuRoceTransceiver.
///
/// This file is compiled by g++ (not nvcc) to isolate Hololink's fmt
/// dependency from CUDA translation units.

#include "hololink_wrapper.h"

// Include Hololink headers here (with Holoscan's fmt)
#include <hololink/operators/gpu_roce_transceiver/gpu_roce_transceiver.hpp>

#include <iostream>

using namespace hololink::operators;

//==============================================================================
// Internal implementation struct
//==============================================================================

struct HololinkTransceiverImpl {
  std::unique_ptr<GpuRoceTransceiver> transceiver;
  size_t page_size;
  unsigned num_pages;
};

//==============================================================================
// Lifecycle
//==============================================================================

hololink_transceiver_t hololink_create_transceiver(const char *device_name,
                                                   int ib_port,
                                                   size_t frame_size,
                                                   size_t page_size,
                                                   unsigned num_pages,
                                                   const char *peer_ip,
                                                   int forward, int rx_only,
                                                   int tx_only) {
  try {
    auto *impl = new HololinkTransceiverImpl();
    impl->page_size = page_size;
    impl->num_pages = num_pages;
    impl->transceiver = std::make_unique<GpuRoceTransceiver>(
        device_name, static_cast<unsigned>(ib_port),
        frame_size, page_size, num_pages,
        peer_ip, forward != 0, rx_only != 0, tx_only != 0);
    return reinterpret_cast<hololink_transceiver_t>(impl);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Failed to create GpuRoceTransceiver: " << e.what()
              << std::endl;
    return nullptr;
  } catch (...) {
    std::cerr
        << "ERROR: Failed to create GpuRoceTransceiver: unknown exception"
        << std::endl;
    return nullptr;
  }
}

void hololink_destroy_transceiver(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    delete impl;
  }
}

int hololink_start(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->start() ? 1 : 0;
  }
  return 0;
}

void hololink_close(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    impl->transceiver->close();
  }
}

void hololink_blocking_monitor(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    impl->transceiver->blocking_monitor();
  }
}

//==============================================================================
// QP information
//==============================================================================

uint32_t hololink_get_qp_number(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->get_qp_number();
  }
  return 0;
}

uint32_t hololink_get_rkey(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->get_rkey();
  }
  return 0;
}

uint64_t hololink_get_buffer_addr(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->external_frame_memory();
  }
  return 0;
}

int hololink_get_gid(hololink_transceiver_t handle, uint8_t *gid_out) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->get_gid(gid_out);
  }
  return 0;
}

//==============================================================================
// Deferred QP connection
//==============================================================================

int hololink_reconnect_qp(hololink_transceiver_t handle,
                           const uint8_t *remote_gid, uint32_t remote_qpn) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->reconnect_qp(remote_gid, remote_qpn) ? 1 : 0;
  }
  return 0;
}

//==============================================================================
// Ring buffer access
//==============================================================================

void *hololink_get_rx_ring_data_addr(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->get_rx_ring_data_addr();
  }
  return nullptr;
}

uint64_t *hololink_get_rx_ring_flag_addr(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->get_rx_ring_flag_addr();
  }
  return nullptr;
}

void *hololink_get_tx_ring_data_addr(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->get_tx_ring_data_addr();
  }
  return nullptr;
}

uint64_t *hololink_get_tx_ring_flag_addr(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->get_tx_ring_flag_addr();
  }
  return nullptr;
}

uint64_t *hololink_get_tx_ring_flag_host_addr(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->get_tx_ring_flag_host_addr();
  }
  return nullptr;
}

uint64_t *hololink_get_rx_ring_flag_host_addr(hololink_transceiver_t handle) {
  // Note: GpuRoceTransceiver does not currently expose host RX flag addr.
  (void)handle;
  return nullptr;
}

bool hololink_query_kernel_occupancy(void) {
  int prep = 0, rx = 0, tx = 0;
  cudaError_t err = GpuRoceTransceiverQueryOccupancy(&prep, &rx, &tx);
  if (err != cudaSuccess) {
    fprintf(stderr, "ERROR: Hololink kernel occupancy query failed: %s\n",
            cudaGetErrorString(err));
    return false;
  }
  printf("  Hololink kernel occupancy: prepare=%d rx=%d tx=%d\n", prep, rx, tx);
  return true;
}

size_t hololink_get_page_size(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->page_size;
  }
  return 0;
}

unsigned hololink_get_num_pages(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->num_pages;
  }
  return 0;
}
