/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file unified_transport_none.cu
/// @brief Null implementation of the unified dispatch device data plane.
///
/// Compiled into libcudaq-realtime only when the build configures no
/// transport that supplies the hooks (see lib/daemon/CMakeLists.txt).  The
/// unified kernel calls these as __device__ functions, which CUDA resolves at
/// device-link time, so something must define them or the library does not
/// link at all -- and cudaq_dispatcher_start calls
/// cudaq_launch_unified_dispatch_device directly, so it has to exist in every
/// configuration.
///
/// `attach` refusing means the kernel returns immediately without touching
/// the transport context, so a caller that wires up unified dispatch in a
/// build with no transport gets a kernel that does nothing rather than
/// undefined behaviour.

#include "cudaq/realtime/daemon/dispatcher/unified_device_transport.cuh"

extern "C" __device__ void *
cudaq_dev_transport_attach(void * /*ctx*/, volatile int * /*shutdown_flag*/) {
  return nullptr;
}

extern "C" __device__ cudaq_rx_dev_status_t cudaq_dev_rx_poll(
    void * /*session*/, void ** /*out_request*/, void ** /*out_response*/) {
  return CUDAQ_RX_DEV_SHUTDOWN;
}

extern "C" __device__ void cudaq_dev_tx_publish(void * /*session*/,
                                                void * /*response*/) {}
