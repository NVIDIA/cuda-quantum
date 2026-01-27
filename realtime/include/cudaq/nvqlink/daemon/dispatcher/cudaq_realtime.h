/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef struct cudaq_dispatch_manager_t cudaq_dispatch_manager_t;
typedef struct cudaq_dispatcher_t cudaq_dispatcher_t;

// Error codes
typedef enum {
  CUDAQ_OK = 0,
  CUDAQ_ERR_INVALID_ARG = 1,
  CUDAQ_ERR_INTERNAL = 2,
  CUDAQ_ERR_CUDA = 3
} cudaq_status_t;

// Kernel synchronization type
typedef enum {
  CUDAQ_KERNEL_REGULAR = 0,
  CUDAQ_KERNEL_COOPERATIVE = 1
} cudaq_kernel_type_t;

// Dispatch invocation mode
typedef enum {
  CUDAQ_DISPATCH_DEVICE_CALL = 0,
  CUDAQ_DISPATCH_GRAPH_LAUNCH = 1
} cudaq_dispatch_mode_t;

// Dispatcher configuration
typedef struct {
  int device_id;                       // GPU device ID (>=0)
  uint32_t num_blocks;                 // grid size
  uint32_t threads_per_block;          // block size
  uint32_t num_slots;                  // ring buffer slots
  uint32_t slot_size;                  // bytes per slot
  uint32_t vp_id;                      // virtual port ID
  cudaq_kernel_type_t kernel_type;     // regular/cooperative kernel
  cudaq_dispatch_mode_t dispatch_mode; // device call/graph launch
} cudaq_dispatcher_config_t;

// GPU ring buffer pointers (device-visible mapped pointers)
typedef struct {
  volatile uint64_t *rx_flags; // device pointer
  volatile uint64_t *tx_flags; // device pointer
} cudaq_ringbuffer_t;

// Function table for device-side dispatch
typedef struct {
  void *
      *device_function_ptrs; // device pointer to array of device function ptrs
  uint32_t *function_ids;    // device pointer to array of function IDs
  size_t count;              // number of entries
} cudaq_function_table_t;

// Host launch function pointer type
typedef void (*cudaq_dispatch_launch_fn_t)(
    volatile uint64_t *rx_flags, volatile uint64_t *tx_flags,
    void **function_table, uint32_t *function_ids, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, size_t num_slots,
    uint32_t num_blocks, uint32_t threads_per_block, cudaStream_t stream);

// Default dispatch kernel launch helpers (from libcudaq-realtime.so)
void cudaq_launch_dispatch_kernel_regular(
    volatile uint64_t *rx_flags, volatile uint64_t *tx_flags,
    void **function_table, uint32_t *function_ids, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, size_t num_slots,
    uint32_t num_blocks, uint32_t threads_per_block, cudaStream_t stream);

void cudaq_launch_dispatch_kernel_cooperative(
    volatile uint64_t *rx_flags, volatile uint64_t *tx_flags,
    void **function_table, uint32_t *function_ids, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, size_t num_slots,
    uint32_t num_blocks, uint32_t threads_per_block, cudaStream_t stream);

// Manager lifecycle
cudaq_status_t
cudaq_dispatch_manager_create(cudaq_dispatch_manager_t **out_mgr);
cudaq_status_t cudaq_dispatch_manager_destroy(cudaq_dispatch_manager_t *mgr);

// Dispatcher lifecycle
cudaq_status_t cudaq_dispatcher_create(cudaq_dispatch_manager_t *mgr,
                                       const cudaq_dispatcher_config_t *config,
                                       cudaq_dispatcher_t **out_dispatcher);
cudaq_status_t cudaq_dispatcher_destroy(cudaq_dispatcher_t *dispatcher);

// Wiring inputs
cudaq_status_t
cudaq_dispatcher_set_ringbuffer(cudaq_dispatcher_t *dispatcher,
                                const cudaq_ringbuffer_t *ringbuffer);
cudaq_status_t
cudaq_dispatcher_set_function_table(cudaq_dispatcher_t *dispatcher,
                                    const cudaq_function_table_t *table);
cudaq_status_t cudaq_dispatcher_set_control(cudaq_dispatcher_t *dispatcher,
                                            volatile int *shutdown_flag,
                                            uint64_t *stats);
cudaq_status_t
cudaq_dispatcher_set_launch_fn(cudaq_dispatcher_t *dispatcher,
                               cudaq_dispatch_launch_fn_t launch_fn);

// Start/stop
cudaq_status_t cudaq_dispatcher_start(cudaq_dispatcher_t *dispatcher);
cudaq_status_t cudaq_dispatcher_stop(cudaq_dispatcher_t *dispatcher);

// Stats
cudaq_status_t cudaq_dispatcher_get_processed(cudaq_dispatcher_t *dispatcher,
                                              uint64_t *out_packets);

#ifdef __cplusplus
}
#endif
