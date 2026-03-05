/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file hololink_doca_transport_ctx.h
/// @brief Hololink/DOCA transport context shared between the Hololink bridge
/// and its unified dispatch kernel.
///
/// The struct uses void* to hide the DOCA-specific qp handle so this header
/// has no DOCA dependency and is safe to include from public consumers.

#include <cstddef>
#include <cstdint>

/// Transport context populated by the bridge's get_transport_context(UNIFIED)
/// and consumed by hololink_launch_unified_dispatch.
struct hololink_doca_transport_ctx {
  void *gpu_dev_qp;            ///< doca_gpu_dev_verbs_qp* handle (opaque)
  uint8_t *rx_ring_data;       ///< Device pointer to RX ring data buffer
  size_t rx_ring_stride_sz;    ///< Stride (slot size) in the ring buffer
  uint32_t rx_ring_mkey;       ///< Network-byte-order memory key (htobe32(rkey))
  uint32_t rx_ring_stride_num; ///< Number of slots in the ring
  size_t frame_size;           ///< Actual frame/payload size within a slot
};
