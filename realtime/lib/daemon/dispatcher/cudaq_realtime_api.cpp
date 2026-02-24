/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <cstdio>
#include <new>

struct cudaq_dispatch_manager_t {
  int reserved = 0;
};

struct cudaq_dispatcher_t {
  cudaq_dispatcher_config_t config{};
  cudaq_ringbuffer_t ringbuffer{};
  cudaq_function_table_t table{};
  cudaq_dispatch_launch_fn_t launch_fn = nullptr;
  volatile int *shutdown_flag = nullptr;
  uint64_t *stats = nullptr;
  cudaStream_t stream = nullptr;
  bool running = false;
  cudaq_host_dispatcher_handle_t *host_handle = nullptr;
};

static bool is_valid_kernel_type(cudaq_kernel_type_t kernel_type) {
  switch (kernel_type) {
  case CUDAQ_KERNEL_REGULAR:
  case CUDAQ_KERNEL_COOPERATIVE:
    return true;
  default:
    return false;
  }
}

static bool is_valid_dispatch_mode(cudaq_dispatch_mode_t dispatch_mode) {
  switch (dispatch_mode) {
  case CUDAQ_DISPATCH_DEVICE_CALL:
  case CUDAQ_DISPATCH_GRAPH_LAUNCH:
  case CUDAQ_DISPATCH_HOST_CALL:
    return true;
  default:
    return false;
  }
}

static cudaq_status_t validate_dispatcher(cudaq_dispatcher_t *dispatcher) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (!dispatcher->shutdown_flag || !dispatcher->stats)
    return CUDAQ_ERR_INVALID_ARG;
  if (!dispatcher->ringbuffer.rx_flags || !dispatcher->ringbuffer.tx_flags)
    return CUDAQ_ERR_INVALID_ARG;
  if (!dispatcher->table.entries || dispatcher->table.count == 0)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->config.num_slots == 0 || dispatcher->config.slot_size == 0)
    return CUDAQ_ERR_INVALID_ARG;

  if (dispatcher->config.backend == CUDAQ_BACKEND_HOST_LOOP) {
    if (!dispatcher->ringbuffer.rx_flags_host || !dispatcher->ringbuffer.tx_flags_host ||
        !dispatcher->ringbuffer.rx_data_host || !dispatcher->ringbuffer.tx_data_host)
      return CUDAQ_ERR_INVALID_ARG;
    return CUDAQ_OK;
  }

  if (!dispatcher->launch_fn)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->config.num_blocks == 0 ||
      dispatcher->config.threads_per_block == 0)
    return CUDAQ_ERR_INVALID_ARG;
  if (!is_valid_kernel_type(dispatcher->config.kernel_type) ||
      !is_valid_dispatch_mode(dispatcher->config.dispatch_mode))
    return CUDAQ_ERR_INVALID_ARG;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatch_manager_create(cudaq_dispatch_manager_t **out_mgr) {
  if (!out_mgr)
    return CUDAQ_ERR_INVALID_ARG;
  auto *mgr = new (std::nothrow) cudaq_dispatch_manager_t();
  if (!mgr)
    return CUDAQ_ERR_INTERNAL;
  *out_mgr = mgr;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatch_manager_destroy(cudaq_dispatch_manager_t *mgr) {
  if (mgr)
    delete mgr;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_create(cudaq_dispatch_manager_t *,
                                       const cudaq_dispatcher_config_t *config,
                                       cudaq_dispatcher_t **out_dispatcher) {
  if (!config || !out_dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  auto *dispatcher = new (std::nothrow) cudaq_dispatcher_t();
  if (!dispatcher)
    return CUDAQ_ERR_INTERNAL;
  dispatcher->config = *config;
  *out_dispatcher = dispatcher;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_destroy(cudaq_dispatcher_t *dispatcher) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->running && dispatcher->host_handle) {
    *dispatcher->shutdown_flag = 1;
    cudaq_host_dispatcher_stop(dispatcher->host_handle);
    dispatcher->host_handle = nullptr;
  }
  delete dispatcher;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_ringbuffer(cudaq_dispatcher_t *dispatcher,
                                const cudaq_ringbuffer_t *ringbuffer) {
  if (!dispatcher || !ringbuffer)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->ringbuffer = *ringbuffer;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_function_table(cudaq_dispatcher_t *dispatcher,
                                    const cudaq_function_table_t *table) {
  if (!dispatcher || !table)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->table = *table;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_set_control(cudaq_dispatcher_t *dispatcher,
                                            volatile int *shutdown_flag,
                                            uint64_t *stats) {
  if (!dispatcher || !shutdown_flag || !stats)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->shutdown_flag = shutdown_flag;
  dispatcher->stats = stats;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_launch_fn(cudaq_dispatcher_t *dispatcher,
                               cudaq_dispatch_launch_fn_t launch_fn) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->config.backend == CUDAQ_BACKEND_HOST_LOOP && launch_fn != nullptr)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->config.backend != CUDAQ_BACKEND_HOST_LOOP && !launch_fn)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->launch_fn = launch_fn;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_start(cudaq_dispatcher_t *dispatcher) {
  auto status = validate_dispatcher(dispatcher);
  if (status != CUDAQ_OK)
    return status;
  if (dispatcher->running)
    return CUDAQ_OK;

  int device_id = dispatcher->config.device_id;
  if (device_id < 0)
    device_id = 0;
  if (cudaSetDevice(device_id) != cudaSuccess)
    return CUDAQ_ERR_CUDA;

  if (dispatcher->config.backend == CUDAQ_BACKEND_HOST_LOOP) {
    dispatcher->host_handle = cudaq_host_dispatcher_start_thread(
        &dispatcher->ringbuffer, &dispatcher->table, &dispatcher->config,
        dispatcher->shutdown_flag, dispatcher->stats);
    if (!dispatcher->host_handle)
      return CUDAQ_ERR_INTERNAL;
    dispatcher->running = true;
    return CUDAQ_OK;
  }

  if (cudaStreamCreate(&dispatcher->stream) != cudaSuccess)
    return CUDAQ_ERR_CUDA;

  dispatcher->launch_fn(
      dispatcher->ringbuffer.rx_flags, dispatcher->ringbuffer.tx_flags,
      dispatcher->ringbuffer.rx_data, dispatcher->ringbuffer.tx_data,
      dispatcher->ringbuffer.rx_stride_sz, dispatcher->ringbuffer.tx_stride_sz,
      dispatcher->table.entries, dispatcher->table.count,
      dispatcher->shutdown_flag, dispatcher->stats,
      dispatcher->config.num_slots, dispatcher->config.num_blocks,
      dispatcher->config.threads_per_block, dispatcher->stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error in dispatcher launch: %s (%d)\n",
            cudaGetErrorString(err), err);
    cudaStreamDestroy(dispatcher->stream);
    dispatcher->stream = nullptr;
    return CUDAQ_ERR_CUDA;
  }

  dispatcher->running = true;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_stop(cudaq_dispatcher_t *dispatcher) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (!dispatcher->running)
    return CUDAQ_OK;

  if (dispatcher->config.backend == CUDAQ_BACKEND_HOST_LOOP &&
      dispatcher->host_handle) {
    *dispatcher->shutdown_flag = 1;
    cudaq_host_dispatcher_stop(dispatcher->host_handle);
    dispatcher->host_handle = nullptr;
    dispatcher->running = false;
    return CUDAQ_OK;
  }

  int shutdown = 1;
  if (cudaMemcpy(const_cast<int *>(dispatcher->shutdown_flag), &shutdown,
                 sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    return CUDAQ_ERR_CUDA;
  cudaStreamSynchronize(dispatcher->stream);
  cudaStreamDestroy(dispatcher->stream);
  dispatcher->stream = nullptr;
  dispatcher->running = false;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_get_processed(cudaq_dispatcher_t *dispatcher,
                                              uint64_t *out_packets) {
  if (!dispatcher || !out_packets || !dispatcher->stats)
    return CUDAQ_ERR_INVALID_ARG;

  if (dispatcher->config.backend == CUDAQ_BACKEND_HOST_LOOP) {
    *out_packets = *dispatcher->stats;
    return CUDAQ_OK;
  }

  if (cudaMemcpy(out_packets, dispatcher->stats, sizeof(uint64_t),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return CUDAQ_ERR_CUDA;

  return CUDAQ_OK;
}
