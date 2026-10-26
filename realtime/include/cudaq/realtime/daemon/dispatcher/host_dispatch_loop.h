/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

/// @file host_dispatch_loop.h
/// @brief The two CPU dispatch loops behind CUDAQ_DISPATCH_PATH_HOST.
///
/// Both loops serve CUDAQ_DISPATCH_HOST_CALL entries only.  Each request is
/// resolved against the function table and its `host_fn` runs inline on the
/// loop's thread; the response is published as soon as `host_fn` returns, so
/// there is no deferred completion.  A request whose entry is missing or has
/// another dispatch mode (DEVICE_CALL, GRAPH_LAUNCH: both belong to the GPU
/// dispatch path) is left for a peer dispatcher under `shared_ring_mode` and
/// dropped otherwise.
///
/// Neither loop touches the CUDA runtime, and neither selects a CUDA device.
/// A handler that uses the GPU selects its own device and finishes that work
/// before returning.
///
/// `cudaq_dispatcher_start` runs one of these on the dispatcher's host thread;
/// a program that drives its own ring without a dispatcher object may call
/// them directly.  Both block until `*shutdown_flag != 0` and then add the
/// number of handled requests to `*stats` (which may be NULL).

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Ring shape (CUDAQ_KERNEL_REGULAR): poll `ringbuffer->rx_flags_host`, run
/// the request's HOST_CALL handler into the matching TX slot, then store that
/// slot's address into `tx_flags_host` (address-as-flag) for the transport's
/// TX thread to send.  The transport's RX/TX adapter threads move bytes
/// wire<->ring around it.  `config` supplies `num_slots` and
/// `shared_ring_mode`.
void cudaq_host_ring_dispatch_loop(const cudaq_ringbuffer_t *ringbuffer,
                                   const cudaq_function_table_t *table,
                                   const cudaq_dispatcher_config_t *config,
                                   volatile int *shutdown_flag,
                                   uint64_t *stats);

/// Unified shape (CUDAQ_KERNEL_UNIFIED): one thread does RX, dispatch and TX
/// through the data plane's `rx_poll` / `tx_publish` hooks, which must both be
/// set.
void cudaq_host_unified_loop(cudaq_cpu_dataplane_t *cpu_dataplane,
                             const cudaq_function_table_t *table,
                             volatile int *shutdown_flag, uint64_t *stats);

#ifdef __cplusplus
}
#endif
