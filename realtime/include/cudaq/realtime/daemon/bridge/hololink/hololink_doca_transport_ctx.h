/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Hololink/DOCA transport context for the unified dispatch kernel.
/// Packed by the Hololink bridge layer and passed as the opaque
/// transport_ctx pointer through the transport-agnostic dispatcher API.
typedef struct {
  void *gpu_dev_qp;         ///< doca_gpu_dev_verbs_qp* handle
  uint8_t *rx_ring_data;    ///< Device pointer to RX ring data buffer
  size_t rx_ring_stride_sz; ///< Stride (slot size) in the ring buffer
  uint32_t rx_ring_mkey;    ///< Network-byte-order memory key (`htobe32(rkey)`)
  uint32_t rx_ring_stride_num; ///< Number of slots in the ring
  size_t frame_size;           ///< Actual frame/payload size within a slot
  int use_bf;                  ///< Non-zero: use BlueFlame TX (dGPU).
                               ///< Zero: use NIC_HANDLER_AUTO (iGPU/CPU proxy).
} hololink_doca_transport_ctx;

#ifdef __cplusplus
}
#endif
