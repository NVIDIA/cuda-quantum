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

/// GpuRoceTransceiver/DOCA transport context for the unified dispatch kernel.
/// Packed by the GpuRoceTransceiver bridge layer and passed as the opaque
/// transport_ctx pointer through the transport-agnostic dispatcher API.
///
/// Requests land in RX slot N and the response to each goes out of TX slot N,
/// so the TX ring must have at least `rx_ring_stride_num` slots.  Only the TX
/// ring needs a key here: receive WQEs carry no buffer (the FPGA RDMA-writes
/// into the RX ring with the RX rkey it is given out of band).
typedef struct {
  void *gpu_dev_qp;            ///< doca_gpu_dev_verbs_qp* handle
  uint8_t *rx_ring_data;       ///< Device pointer to RX ring data buffer
  size_t rx_ring_stride_sz;    ///< Stride (slot size) in the RX ring
  uint32_t rx_ring_stride_num; ///< Number of slots in the RX ring
  uint8_t *tx_ring_data;       ///< Device pointer to TX ring data buffer
  size_t tx_ring_stride_sz;    ///< Stride (slot size) in the TX ring
  uint32_t tx_ring_mkey; ///< TX ring `lkey`, network byte order (`htobe32`)
  size_t frame_size;     ///< Actual frame/payload size within a slot
  int use_bf;            ///< Non-zero: use BlueFlame TX (dGPU).
                         ///< Zero: use NIC_HANDLER_AUTO (iGPU/CPU proxy).
} gpu_roce_doca_transport_ctx;

#ifdef __cplusplus
}
#endif
