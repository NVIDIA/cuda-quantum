/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file gpu_roce_device_hooks.cu
/// @brief The gpu_roce bridge's unified-kernel hooks
/// (unified_device_transport.cuh): forwarders to the device transport
/// (gpu_roce_device_transport.cuh), which keeps its own names and types.
/// Both are built as LTO-IR, so the run-time link inlines the transport
/// through these into the kernel.

#include "gpu_roce_device_transport.cuh"
#include "cudaq/realtime/daemon/dispatcher/unified_device_transport.cuh"

static_assert(GPU_ROCE_DEV_SESSION_BYTES <= CUDAQ_DEV_TRANSPORT_SESSION_BYTES &&
                  GPU_ROCE_DEV_SESSION_ALIGN <=
                      CUDAQ_DEV_TRANSPORT_SESSION_ALIGN,
              "the transport's session must fit the kernel's session storage");

extern "C" __device__ void *
cudaq_dev_enter_kernel(void *storage, void *ctx, volatile int *shutdown_flag) {
  return gpu_roce_dev_enter_kernel(storage, ctx, shutdown_flag);
}

extern "C" __device__ cudaq_rx_dev_status_t
cudaq_dev_rx_poll(void *session, void **out_request, void **out_response) {
  return gpu_roce_dev_rx_poll(session, out_request, out_response) ==
                 GPU_ROCE_DEV_RX_READY
             ? CUDAQ_RX_DEV_READY
             : CUDAQ_RX_DEV_SHUTDOWN;
}

extern "C" __device__ void cudaq_dev_tx_publish(void *session, void *response) {
  gpu_roce_dev_tx_publish(session, response);
}
