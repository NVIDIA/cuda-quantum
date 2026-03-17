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

#include "cudaq/realtime/daemon/bridge/hololink/hololink_wrapper.h"

// Include Hololink headers here (with Holoscan's fmt)
// Disable deprecation warnings for Hololink headers, which may use deprecated
// APIs
#if (defined(__GNUC__) && !defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <hololink/operators/gpu_roce_transceiver/gpu_roce_transceiver.hpp>
#if (defined(__GNUC__) && !defined(__clang__))
#pragma GCC diagnostic pop
#endif

#include <doca_error.h>
#include <endian.h>
#include <iostream>

extern "C" cudaError_t GpuRoceTransceiverQueryOccupancy(int *, int *, int *);
extern "C" doca_error_t GpuRoceTransceiverPrepareKernel(
    cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp, size_t frame_size,
    uint32_t cu_page_mkey, uint32_t cuda_blocks, uint32_t cuda_threads);

using namespace hololink::operators;

//==============================================================================
// Internal implementation struct
//==============================================================================

struct HololinkTransceiverImpl {
  std::unique_ptr<GpuRoceTransceiver> transceiver;
};

//==============================================================================
// Lifecycle
//==============================================================================

hololink_transceiver_t hololink_create_transceiver(
    const char *device_name, int ib_port, unsigned tx_ibv_qp, int gpu_id,
    size_t frame_size, size_t page_size, unsigned num_pages,
    const char *peer_ip, int forward, int rx_only, int tx_only) {
  try {
    auto *impl = new HololinkTransceiverImpl();
    impl->transceiver = std::make_unique<GpuRoceTransceiver>(
        device_name, static_cast<unsigned>(ib_port), tx_ibv_qp, gpu_id,
        frame_size, page_size, num_pages, peer_ip, forward != 0, rx_only != 0,
        tx_only != 0);
    return reinterpret_cast<hololink_transceiver_t>(impl);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Failed to create GpuRoceTransceiver: " << e.what()
              << std::endl;
    return nullptr;
  } catch (...) {
    std::cerr << "ERROR: Failed to create GpuRoceTransceiver: unknown exception"
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

void *hololink_get_gpu_dev_qp(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->get_doca_gpu_dev_qp(0);
  }
  return nullptr;
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
    return impl->transceiver->get_rx_ring_stride_sz();
  }
  return 0;
}

unsigned hololink_get_num_pages(hololink_transceiver_t handle) {
  if (handle) {
    auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
    return impl->transceiver->get_rx_ring_stride_num();
  }
  return 0;
}

int hololink_prepare_receive_send(hololink_transceiver_t handle,
                                  size_t frame_size) {
  if (!handle)
    return 0;
  auto *impl = reinterpret_cast<HololinkTransceiverImpl *>(handle);
  auto *qp = static_cast<struct doca_gpu_dev_verbs_qp *>(
      impl->transceiver->get_doca_gpu_dev_qp(0));
  if (!qp)
    return 0;
  uint32_t mkey = htobe32(impl->transceiver->get_rkey());
  doca_error_t err =
      GpuRoceTransceiverPrepareKernel(0, qp, frame_size, mkey, 1, 64);
  if (err != DOCA_SUCCESS) {
    std::cerr << "ERROR: GpuRoceTransceiverPrepareKernel failed: "
              << doca_error_get_descr(err) << std::endl;
    return 0;
  }
  cudaStreamSynchronize(0);
  return 1;
}
