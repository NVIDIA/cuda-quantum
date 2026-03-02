/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Transport context for the DOCA/Hololink unified dispatch kernel.
/// Packed by the bridge tool and passed as the opaque transport_ctx pointer
/// through the transport-agnostic dispatcher API.
typedef struct {
  void *gpu_dev_qp;          ///< doca_gpu_dev_verbs_qp* handle
  uint8_t *rx_ring_data;     ///< Device pointer to RX ring data buffer
  size_t rx_ring_stride_sz;  ///< Stride (slot size) in the ring buffer
  uint32_t rx_ring_mkey;     ///< Network-byte-order memory key (htobe32(rkey))
  uint32_t rx_ring_stride_num; ///< Number of slots in the ring
  size_t frame_size;         ///< Actual frame/payload size within a slot
} doca_transport_ctx;

/// Unified dispatch kernel launch wrapper.
/// Matches cudaq_unified_launch_fn_t signature.
void cudaq_launch_unified_dispatch_kernel(
    void *transport_ctx, cudaq_function_entry_t *function_table,
    size_t func_count, volatile int *shutdown_flag, uint64_t *stats,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
