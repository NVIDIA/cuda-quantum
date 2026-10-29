/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file gpu_roce_device_transport.cuh
/// @brief Device side of the GpuRoceTransceiver data plane
/// (gpu_roce_unified_transport.cu).
///
/// Includes no CUDA-Q realtime header: the transport names and types its
/// functions itself, and the gpu_roce bridge adapts them to the unified
/// dispatch kernel's hooks (gpu_roce_device_hooks.cu).

/// Storage `gpu_roce_dev_enter_kernel` needs for its session.
#define GPU_ROCE_DEV_SESSION_BYTES 256
#define GPU_ROCE_DEV_SESSION_ALIGN 16

extern "C" {

typedef enum {
  GPU_ROCE_DEV_RX_READY = 0,    ///< A request is ready
  GPU_ROCE_DEV_RX_SHUTDOWN = 1, ///< The shutdown flag is set
} gpu_roce_dev_rx_status_t;

/// Build the session in `storage` from `ctx`, a device-resident
/// gpu_roce_doca_transport_ctx, and return it; nullptr if it cannot run.
__device__ void *gpu_roce_dev_enter_kernel(void *storage, void *ctx,
                                           volatile int *shutdown_flag);

/// Block until a request arrives, or the shutdown flag is set.
__device__ gpu_roce_dev_rx_status_t gpu_roce_dev_rx_poll(void *session,
                                                         void **out_request,
                                                         void **out_response);

/// Send `response` and repost the receive it answers.
__device__ void gpu_roce_dev_tx_publish(void *session, void *response);
}
