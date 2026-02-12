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

// Payload type identifiers (matching PayloadTypeID in dispatch_kernel_launch.h)
typedef enum {
  CUDAQ_TYPE_UINT8 = 0x10,
  CUDAQ_TYPE_INT32 = 0x11,
  CUDAQ_TYPE_INT64 = 0x12,
  CUDAQ_TYPE_FLOAT32 = 0x13,
  CUDAQ_TYPE_FLOAT64 = 0x14,
  CUDAQ_TYPE_ARRAY_UINT8 = 0x20,
  CUDAQ_TYPE_ARRAY_INT32 = 0x21,
  CUDAQ_TYPE_ARRAY_FLOAT32 = 0x22,
  CUDAQ_TYPE_ARRAY_FLOAT64 = 0x23,
  CUDAQ_TYPE_BIT_PACKED = 0x30
} cudaq_payload_type_t;

// Type descriptor for arguments/results
typedef struct {
  uint8_t type_id;       // cudaq_payload_type_t value
  uint8_t reserved[3];   // padding
  uint32_t size_bytes;   // total size in bytes
  uint32_t num_elements; // number of elements (for arrays)
} cudaq_type_desc_t;

// Handler schema describing function signature
typedef struct {
  uint8_t num_args;             // number of arguments
  uint8_t num_results;          // number of results
  uint16_t reserved;            // padding
  cudaq_type_desc_t args[8];    // argument descriptors (max 8)
  cudaq_type_desc_t results[4]; // result descriptors (max 4)
} cudaq_handler_schema_t;

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
  uint8_t *rx_data;            // device pointer to RX data buffer
  uint8_t *tx_data;            // device pointer to TX data buffer
  size_t rx_stride_sz;         // size of each RX slot in bytes
  size_t tx_stride_sz;         // size of each TX slot in bytes
} cudaq_ringbuffer_t;

// Unified function table entry with schema
typedef struct {
  union {
    void *device_fn_ptr;        // for CUDAQ_DISPATCH_DEVICE_CALL
    cudaGraphExec_t graph_exec; // for CUDAQ_DISPATCH_GRAPH_LAUNCH
  } handler;
  uint32_t function_id;          // hash of function name (FNV-1a)
  uint8_t dispatch_mode;         // cudaq_dispatch_mode_t value
  uint8_t reserved[3];           // padding
  cudaq_handler_schema_t schema; // function signature schema
} cudaq_function_entry_t;

// Function table for device-side dispatch
typedef struct {
  cudaq_function_entry_t *entries; // device pointer to array of entries
  uint32_t count;                  // number of entries
} cudaq_function_table_t;

// Host launch function pointer type
typedef void (*cudaq_dispatch_launch_fn_t)(
    volatile uint64_t *rx_flags, volatile uint64_t *tx_flags,
    uint8_t *rx_data, uint8_t *tx_data,
    size_t rx_stride_sz, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, size_t num_slots,
    uint32_t num_blocks, uint32_t threads_per_block, cudaStream_t stream);

// Default dispatch kernel launch helpers (from libcudaq-realtime-dispatch.a)
void cudaq_launch_dispatch_kernel_regular(
    volatile uint64_t *rx_flags, volatile uint64_t *tx_flags,
    uint8_t *rx_data, uint8_t *tx_data,
    size_t rx_stride_sz, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, size_t num_slots,
    uint32_t num_blocks, uint32_t threads_per_block, cudaStream_t stream);

void cudaq_launch_dispatch_kernel_cooperative(
    volatile uint64_t *rx_flags, volatile uint64_t *tx_flags,
    uint8_t *rx_data, uint8_t *tx_data,
    size_t rx_stride_sz, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, size_t num_slots,
    uint32_t num_blocks, uint32_t threads_per_block, cudaStream_t stream);

// Graph-enabled dispatch kernels (requires compute capability 9.0+, sm_90+)
// These functions are only available when compiled for sm_90 or higher
#if defined(__CUDACC__) || defined(CUDA_VERSION)

//==============================================================================
// Graph-Based Dispatch API (Proper Device-Side Graph Launch Support)
//==============================================================================
//
// These functions properly support device-side cudaGraphLaunch() by wrapping
// the dispatch kernel in a graph that is instantiated with
// cudaGraphInstantiateFlagDeviceLaunch.
//
// Usage:
//   1. Allocate a GraphIOContext on the device (cudaMalloc)
//   2. Call cudaq_create_dispatch_graph_regular() to create the graph context
//   3. Call cudaq_launch_dispatch_graph() to launch the dispatch kernel
//   4. When done, call cudaq_destroy_dispatch_graph() to cleanup
//
// The dispatch kernel fills the GraphIOContext before each fire-and-forget
// graph launch.  The graph kernel reads input from io_ctx->rx_slot, writes
// the RPCResponse to io_ctx->tx_slot, and signals completion by writing
// io_ctx->tx_flag_value to *io_ctx->tx_flag after a __threadfence_system().

// Forward declaration for GraphIOContext (defined in dispatch_kernel_launch.h)
struct cudaq_graph_io_context;

// Opaque handle for graph-based dispatch context
typedef struct cudaq_dispatch_graph_context cudaq_dispatch_graph_context;

// Create a graph-based dispatch context for the regular kernel type.
// This creates a graph containing the dispatch kernel, instantiates it with
// cudaGraphInstantiateFlagDeviceLaunch, and uploads it to the device.
//
// graph_io_ctx: Device pointer to a GraphIOContext struct. The dispatch
//   kernel fills this before each fire-and-forget child graph launch so
//   the graph kernel knows where to read input and write output.
//
// Returns cudaSuccess on success, or an error code on failure.
cudaError_t cudaq_create_dispatch_graph_regular(
    volatile uint64_t *rx_flags, volatile uint64_t *tx_flags,
    uint8_t *rx_data, uint8_t *tx_data,
    size_t rx_stride_sz, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    void *graph_io_ctx, volatile int *shutdown_flag, uint64_t *stats,
    size_t num_slots, uint32_t num_blocks, uint32_t threads_per_block,
    cudaStream_t stream, cudaq_dispatch_graph_context **out_context);

// Launch the dispatch graph. The dispatch kernel inside this graph can call
// cudaGraphLaunch() to launch child graphs from device code.
cudaError_t cudaq_launch_dispatch_graph(cudaq_dispatch_graph_context *context,
                                        cudaStream_t stream);

// Destroy the dispatch graph context and release all resources.
cudaError_t cudaq_destroy_dispatch_graph(cudaq_dispatch_graph_context *context);

#endif

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

// Force eager CUDA module loading for dispatch kernels (occupancy query).
// Call before cudaq_dispatcher_start() to avoid lazy-loading deadlocks.
cudaError_t cudaq_dispatch_kernel_query_occupancy(int *out_blocks,
                                                  uint32_t threads_per_block);
cudaError_t cudaq_dispatch_kernel_cooperative_query_occupancy(
    int *out_blocks, uint32_t threads_per_block);

#ifdef __cplusplus
}
#endif
