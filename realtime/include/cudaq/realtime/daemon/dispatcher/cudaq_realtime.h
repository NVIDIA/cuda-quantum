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

#include "cudaq/realtime/daemon/dispatcher/cudaq_status.h"
#include "cudaq/realtime/daemon/dispatcher/rpc_wire_format.h"
#include "cudaq/realtime/device_code/device_code.h"
#include "cudaq/realtime/device_code/device_code_reader.h"

// Visibility marker for entry points that consumers reach via
// dlsym(RTLD_DEFAULT, ...) at runtime.  libcudaq-realtime-dispatch.a is built
// with hidden visibility + -Wl,--exclude-libs=ALL, so by default its symbols
// stay hidden inside the final binary even when the archive is absorbed.
// Marking individual symbols with default visibility opts them back into the
// binary's dynamic symbol table (when --export-dynamic is in effect on the
// linker command line for the exe), so a separately-loaded .so can resolve
// them by name without any explicit setter / constructor-shim plumbing on the
// consumer side.
#define CUDAQ_REALTIME_DISPATCH_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef struct cudaq_dispatch_manager_t cudaq_dispatch_manager_t;
typedef struct cudaq_dispatcher_t cudaq_dispatcher_t;

///@brief Opaque data structure storing the details of the transport layer
/// connection (see bridge_interface.h).
typedef void *cudaq_realtime_bridge_handle_t;

// Dispatch control path: GPU-resident persistent kernel vs CPU host loop.
typedef enum {
  CUDAQ_DISPATCH_PATH_DEVICE = 0,
  CUDAQ_DISPATCH_PATH_HOST = 1
} cudaq_dispatch_path_t;

// TX flag status returned by cudaq_host_ringbuffer_poll_tx_flag.
typedef enum {
  CUDAQ_TX_EMPTY = 0,
  CUDAQ_TX_IN_FLIGHT = 1,
  CUDAQ_TX_ERROR = 2,
  CUDAQ_TX_READY = 3
} cudaq_tx_status_t;

// Kernel synchronization type.  Value 1 is retired and must not be reused;
// CUDAQ_KERNEL_UNIFIED keeps its ABI value of 2.
typedef enum {
  CUDAQ_KERNEL_REGULAR = 0,
  CUDAQ_KERNEL_UNIFIED = 2
} cudaq_kernel_type_t;

// Dispatch invocation mode.
//
// On CUDAQ_DISPATCH_PATH_HOST:
//   - GRAPH_LAUNCH: launched on a graph worker (worker pool, cudaGraphLaunch).
//   - HOST_CALL:    invoked synchronously inline as a plain C++ function via
//                   entry->handler.host_fn.  Bypasses the worker pool.  No
//                   CUDA graph required.  Use for GPU-less bridges (e.g. the
//                   CPU RoCE transport in lib/cpu_transport/) or for
//                   handlers whose work fits on a single CPU thread.
//   - DEVICE_CALL:  table entries are dropped (DEVICE_CALL belongs on the
//                   GPU dispatch path).
typedef enum {
  CUDAQ_DISPATCH_DEVICE_CALL = 0,
  CUDAQ_DISPATCH_GRAPH_LAUNCH = 1,
  CUDAQ_DISPATCH_HOST_CALL = 2
} cudaq_dispatch_mode_t;

// Reserved DEVICE_CALL return status (NOT an error).  A device-call handler
// returns this to tell the device-graph scheduler that a runtime condition is
// met and the configured follow-up graph (triggered_graph_exec) should be
// fired fire-and-forget, after which the scheduler tail-self-relaunches to
// reset the device-graph fire-and-forget budget (see
// cudaq_create_dispatch_graph_regular).  The scheduler writes a normal
// status=0 ACK for the triggering request.  Chosen to not collide with
// success (0) or the negative error codes handlers conventionally return.
#define CUDAQ_DISPATCH_STATUS_TRIGGER_GRAPH 0x12A6E5

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
  int device_id;              // GPU device ID (>=0)
  uint32_t num_blocks;        // grid size; must be 1: a request reaches the
                              // handler threads through shared memory and
                              // block barriers, so a wider grid would need a
                              // cooperative launch
  uint32_t threads_per_block; // block size: the block every DEVICE_CALL
                              // handler runs on (>= 1)
  uint32_t num_slots;         // ring buffer slots
  // Bytes per slot: the stride, matching cudaq_ringbuffer_t's rx/tx_stride_sz.
  // Every dispatch shape reads it the same way, CUDAQ_KERNEL_UNIFIED included.
  //
  // KNOWN ISSUE: the dispatch kernels also hand the stride to DEVICE_CALL
  // handlers as `slot_size`, from which a handler derives its result capacity
  // as `slot_size - sizeof(RPCResponse)`.  That bounds the slot buffer
  // correctly, but a transport whose send descriptor is prepared for a
  // shorter fixed frame transmits only that prefix, so a handler can be told
  // it has more room than actually reaches the wire.  Both RoCE transports
  // are in that position (gpu_roce: frame_size vs page_size; cpu_roce:
  // frame_size vs slot_size).  Fixing it means plumbing the on-wire size to
  // both kernels, which reaches the public cudaq_dispatch_launch_fn_t; until
  // then the ring and unified paths at least behave identically.
  uint32_t slot_size;
  uint32_t vp_id;                      // virtual port ID
  cudaq_kernel_type_t kernel_type;     // regular or unified kernel
  cudaq_dispatch_mode_t dispatch_mode; // device call/graph launch
  cudaq_dispatch_path_t dispatch_path; // GPU kernel or CPU host loop
  int skip_tx_markers; // when non-zero, the host dispatcher will NOT write
                       // sentinel markers (CUDAQ_TX_FLAG_IN_FLIGHT) to
                       // tx_flags before graph launch.  Set this when an
                       // external GPU kernel (e.g. GpuRoceTransceiver TX) polls
                       // the same tx_flags array; the sentinel would be
                       // misinterpreted as a valid address.
  uint32_t shared_ring_mode; // when non-zero, the dispatcher cooperates with
                             // OTHER dispatchers on the SAME ring buffer.
                             // Slots whose function_id is not in this
                             // dispatcher's function table (or is in the
                             // table but does not match this dispatcher's
                             // expected dispatch_mode) are SKIPPED without
                             // clearing rx_flags -- the local cursor
                             // advances, leaving the slot for another
                             // dispatcher to pick up.  When zero (default),
                             // legacy behavior: unknown / wrong-mode slots
                             // are DROPPED (rx_flags cleared).  Both
                             // dispatchers sharing a ring must set this to
                             // non-zero; the partitioning invariant is that
                             // each function_id appears in AT MOST ONE
                             // dispatcher's function table.
} cudaq_dispatcher_config_t;

// GPU ring buffer pointers. For device backend use device pointers only.
// For CUDAQ_DISPATCH_PATH_HOST, also set the _host pointers (same pinned
// mapped allocation); the host loop polls rx_flags_host and uses host data.
typedef struct {
  volatile uint64_t *rx_flags; // device pointer
  volatile uint64_t *tx_flags; // device pointer
  uint8_t *rx_data;            // device pointer to RX data buffer
  uint8_t *tx_data;            // device pointer to TX data buffer
  size_t rx_stride_sz;         // size of each RX slot in bytes
  size_t tx_stride_sz;         // size of each TX slot in bytes
  // Host-side view (required when dispatch_path == CUDAQ_DISPATCH_PATH_HOST;
  // NULL otherwise)
  volatile uint64_t *rx_flags_host;
  volatile uint64_t *tx_flags_host;
  uint8_t *rx_data_host;
  uint8_t *tx_data_host;
} cudaq_ringbuffer_t;

// Host RPC callback for CUDAQ_DISPATCH_HOST_CALL.  The handler reads the
// RPCHeader + args from `rx_slot` (the inbound request, read-only) and writes
// the RPCResponse + result into `tx_slot` (the outbound slot the transport will
// send).  The two slots are distinct host buffers (separate RX/TX rings), so
// the handler must echo any preserved header fields (e.g. request_id,
// ptp_timestamp) from rx_slot into tx_slot explicitly.  `slot_size` is the
// usable byte size of each slot (RX and TX strides are equal by configuration;
// the smaller of the two is passed defensively).
typedef void (*cudaq_host_rpc_fn_t)(const void *rx_slot, void *tx_slot,
                                    size_t slot_size);

// Unified function table entry with schema
typedef struct {
  union {
    void *device_fn_ptr;         // for CUDAQ_DISPATCH_DEVICE_CALL
    cudaGraphExec_t graph_exec;  // for CUDAQ_DISPATCH_GRAPH_LAUNCH
    cudaq_host_rpc_fn_t host_fn; // for CUDAQ_DISPATCH_HOST_CALL
  } handler;
  uint32_t function_id;          // hash of function name (FNV-1a)
  uint8_t dispatch_mode;         // cudaq_dispatch_mode_t value
  uint8_t reserved[3];           // padding
  cudaq_handler_schema_t schema; // function signature schema
  // Optional sub-routing key for CUDAQ_DISPATCH_GRAPH_LAUNCH entries.  When
  // multiple GRAPH_LAUNCH entries share the same `function_id` (the multi-
  // instance pattern used by e.g. the QEC realtime decoder suite, where
  // the same `enqueue_syndromes` function name fronts N distinct captured
  // graphs -- one per decoder), the host monitor disambiguates them by
  // `routing_key`, matching it against the request payload's first 8
  // bytes (arg0).  Ignored when dispatch_mode != CUDAQ_DISPATCH_GRAPH_LAUNCH.
  // See proposals/cudaq_realtime_host_api.bs#host-path-graph-routing-key.
  uint64_t routing_key;
} cudaq_function_entry_t;

// Function table for device-side dispatch
typedef struct {
  cudaq_function_entry_t *entries; // device pointer to array of entries
  uint32_t count;                  // number of entries
} cudaq_function_table_t;

// Host launch function pointer type
typedef void (*cudaq_dispatch_launch_fn_t)(
    volatile uint64_t *rx_flags, volatile uint64_t *tx_flags, uint8_t *rx_data,
    uint8_t *tx_data, size_t rx_stride_sz, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, size_t num_slots,
    uint32_t num_blocks, uint32_t threads_per_block, cudaStream_t stream);

// Default dispatch kernel launch helper (from libcudaq-realtime-dispatch.a)
// for CUDAQ_KERNEL_REGULAR.  On the DEVICE ring path, cudaq_dispatcher_start
// calls it whenever no launch function was set, through a weak reference that
// binds at load time to the copy linked into the consumer binary; see
// cudaq_dispatcher_set_launch_fn for when to name one instead.
//
// Marked CUDAQ_REALTIME_DISPATCH_API so the symbol stays in the dynamic table
// after the archive is absorbed into a binary, which is what that weak
// reference binds to.  Consumer .so's that dlsym() it at runtime (e.g.
// cuda-qx's libcudaq-qec-realtime-decoding.so) rely on it too.
CUDAQ_REALTIME_DISPATCH_API void cudaq_launch_dispatch_kernel_regular(
    volatile uint64_t *rx_flags, volatile uint64_t *tx_flags, uint8_t *rx_data,
    uint8_t *tx_data, size_t rx_stride_sz, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, size_t num_slots,
    uint32_t num_blocks, uint32_t threads_per_block, cudaStream_t stream);

// Unified dispatch launch function pointer type.
// The unified kernel combines RDMA RX, RPC dispatch, and RDMA TX into a single
// kernel, eliminating inter-kernel flag handoff overhead.  Transport-specific
// details are passed via an opaque context pointer so the dispatcher API
// remains transport-agnostic.  `threads_per_block` is the one block the
// kernel runs as: thread 0 drives the transport and every thread calls each
// DEVICE_CALL handler (see DeviceRPCFunction in dispatch_kernel_launch.h).
// Only a transport that replaces the whole dispatch loop supplies one; the
// dispatcher's own unified kernel is linked at run time
// (cudaq_dispatcher_get_device_handler).
typedef void (*cudaq_unified_launch_fn_t)(
    void *transport_ctx, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, uint32_t threads_per_block,
    cudaStream_t stream);

// Graph-enabled dispatch kernels (requires compute capability 9.0+, sm_90+)
// These functions are only available when compiled for sm_90 or higher.
// Gate on any CUDA-aware TU: __CUDACC__ (nvcc), CUDA_VERSION (driver cuda.h),
// or CUDART_VERSION (runtime cuda_runtime.h).  Host TUs that include only
// <cuda_runtime.h> define CUDART_VERSION but NOT CUDA_VERSION, so the latter
// alone would hide this API (and the include guard then prevents a later
// <cuda.h> from re-exposing it) -- see qec_realtime_session consumers.
#if defined(__CUDACC__) || defined(CUDA_VERSION) || defined(CUDART_VERSION)

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
// triggered_graph_exec: Optional device-launchable graph (or NULL).  When a
//   DEVICE_CALL handler returns CUDAQ_DISPATCH_STATUS_TRIGGER_GRAPH, the
//   scheduler fires this graph fire-and-forget (it operates on state the
//   handler's library registered out-of-band -- it is NOT handed the
//   request's rx/tx slot), then tail-self-relaunches via
//   cudaGetCurrentGraphExec() so the device-graph fire-and-forget budget (120
//   per parent-graph execution) resets each relaunch.  This is what lets the
//   path sustain far more than 120 such triggered launches.  When NULL, the
//   sentinel is treated as a pass-through status and no self-relaunch occurs.
//
// Returns cudaSuccess on success, or an error code on failure.
// Marked CUDAQ_REALTIME_DISPATCH_API (default visibility) so that, after the
// dispatch archive is absorbed into a binary, these symbols stay in the
// binary's dynamic table and can satisfy a consumer .so that calls them (e.g.
// cudaq-qec-realtime-decoding.so's device-graph scheduler mode).  Without the
// default visibility the archive's hidden definition cannot be re-exported and
// the link fails with "hidden symbol ... referenced by DSO".
CUDAQ_REALTIME_DISPATCH_API cudaError_t cudaq_create_dispatch_graph_regular(
    volatile uint64_t *rx_flags, volatile uint64_t *tx_flags, uint8_t *rx_data,
    uint8_t *tx_data, size_t rx_stride_sz, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    void *graph_io_ctx, volatile int *shutdown_flag, uint64_t *stats,
    size_t num_slots, uint32_t num_blocks, uint32_t threads_per_block,
    cudaGraphExec_t triggered_graph_exec, cudaStream_t stream,
    cudaq_dispatch_graph_context **out_context);

// Launch the dispatch graph. The dispatch kernel inside this graph can call
// cudaGraphLaunch() to launch child graphs from device code.
CUDAQ_REALTIME_DISPATCH_API cudaError_t cudaq_launch_dispatch_graph(
    cudaq_dispatch_graph_context *context, cudaStream_t stream);

// Destroy the dispatch graph context and release all resources.
CUDAQ_REALTIME_DISPATCH_API cudaError_t
cudaq_destroy_dispatch_graph(cudaq_dispatch_graph_context *context);

// Read the trigger-path debug state: the device-side cudaGraphLaunch result
// of the most recent fire-and-forget of the triggered graph (-1000 = never
// fired), the count of fires, and the count of tail self-relaunches.  A
// healthy scheduler shows rc==0 with fires == tail_relaunches; {rc=0,
// fires=N, tails=N-1} means the triggered graph launched but never
// completed.  Uses asynchronous copies on a private non-blocking stream, so
// it is safe to call while the dispatch graph is resident (live or wedged).
// Any output pointer may be NULL to skip that value.
CUDAQ_REALTIME_DISPATCH_API cudaError_t cudaq_dispatch_get_trigger_debug(
    int *trigger_rc, unsigned long long *trigger_fires,
    unsigned long long *tail_relaunches);

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
// When a bridge is attached (cudaq_dispatcher_set_bridge), start() also hands
// the table to it through cudaq_bridge_set_function_table.
cudaq_status_t
cudaq_dispatcher_set_function_table(cudaq_dispatcher_t *dispatcher,
                                    const cudaq_function_table_t *table);
cudaq_status_t cudaq_dispatcher_set_control(cudaq_dispatcher_t *dispatcher,
                                            volatile int *shutdown_flag,
                                            uint64_t *stats);
// Optional on the DEVICE ring path: without it, start() calls the archive's
// cudaq_launch_dispatch_kernel_regular.  That default binds to the first copy
// of the archive exported in the process, so name the launcher here when this
// binary's copy may not be that one: when it hides the archive's symbols
// (-Wl,--exclude-libs), or shares the process with another binary that links
// the archive.  The kernel must share a CUDA module with the function table's
// DEVICE_CALL handlers.  Refused on the HOST path, and NULL is refused on the
// DEVICE path.
cudaq_status_t
cudaq_dispatcher_set_launch_fn(cudaq_dispatcher_t *dispatcher,
                               cudaq_dispatch_launch_fn_t launch_fn);

// Optional: provide a caller-managed pinned mailbox for GRAPH_LAUNCH workers.
// h_mailbox_bank must be allocated with cudaHostAlloc(..., cudaHostAllocMapped)
// and sized to at least (num_graph_launch_entries * sizeof(void*)).
// If set, the dispatcher uses this mailbox instead of allocating its own.
// The caller retains ownership and must free it after cudaq_dispatcher_destroy.
cudaq_status_t cudaq_dispatcher_set_mailbox(cudaq_dispatcher_t *dispatcher,
                                            void **h_mailbox_bank);

// Wire a transport bridge (bridge_interface.h) to this dispatcher.  This is
// the entry point for every bridge-backed dispatcher: the configured
// `dispatch_path` + `kernel_type` name the shape, and the dispatcher asks the
// bridge for the matching data plane:
//   - HOST + REGULAR:               the RING_BUFFER context; skip_tx_markers is
//                                   forced to 1 (the provider's TX pump owns
//                                   the wire).
//   - HOST + UNIFIED:               the provider's CPU data plane.
//   - DEVICE + REGULAR:             the RING_BUFFER context.  start() then
//                                   launches the 3-kernel dispatch kernel
//                                   from libcudaq-realtime-dispatch.a; see
//                                   cudaq_dispatcher_set_launch_fn.
//   - DEVICE + UNIFIED:             the UNIFIED context's launch override, if
//                                   the provider names one; otherwise the
//                                   provider's GPU data plane, linked
//                                   on the first handler lookup
//                                   (cudaq_dispatcher_get_device_handler).
//                                   A provider without an override answers
//                                   CUDAQ_ERR_UNSUPPORTED (or names no
//                                   launch_fn); any other error fails.
//                                   A provider with neither is refused.
// A `num_slots` / `slot_size` left at 0 in the config is filled from the
// bridge's ring geometry; a non-zero value that disagrees with it is an error.
//
// Everything the dispatcher needs is copied, so nothing the caller passes has
// to outlive this call; the bridge itself must outlive the dispatcher.  Call
// before cudaq_dispatcher_start.  A bridge drives exactly one dispatcher;
// wiring it to a second one is refused until the first is destroyed.
//
// Returns CUDAQ_ERR_UNSUPPORTED when the provider does not serve the requested
// shape (typically a provider started in the other mode), and
// CUDAQ_ERR_INVALID_ARG for an incomplete answer or a geometry mismatch.
cudaq_status_t
cudaq_dispatcher_set_bridge(cudaq_dispatcher_t *dispatcher,
                            cudaq_realtime_bridge_handle_t bridge);

// Bundle struct returned by bridge implementations for unified dispatch.
// Contains the bridge-provided launch function and its opaque transport state,
// keeping the dispatcher API transport-agnostic.
typedef struct {
  cudaq_unified_launch_fn_t
      launch_fn;       ///< Bridge-provided unified launch function
  void *transport_ctx; ///< Bridge-owned opaque transport state
} cudaq_unified_dispatch_ctx_t;

// DEPRECATED for bridge-backed dispatchers: use cudaq_dispatcher_set_bridge,
// which makes this call itself.
//
// Unified dispatch wiring -- pass a transport-specific launch function and
// an opaque context holding transport handles (e.g. DOCA QP, rkey).
// When set, cudaq_dispatcher_start() will invoke unified_launch_fn instead of
// the 3-kernel launch_fn.  Ringbuffer setup is not required for unified mode.
//
// `unified_launch_fn` is required: the dispatcher's own unified kernel runs
// only over a bridge's GPU data plane.  It receives config.threads_per_block
// and must keep the one-block threading contract (see DeviceRPCFunction in
// dispatch_kernel_launch.h).  Whichever kernel runs must be device-linked
// into the same module as the DEVICE_CALL handlers in the function table.
// `transport_ctx` is required too, and
// config.slot_size must give the slot stride, which reaches the kernel as
// its tx_stride_sz.
cudaq_status_t
cudaq_dispatcher_set_unified_launch(cudaq_dispatcher_t *dispatcher,
                                    cudaq_unified_launch_fn_t unified_launch_fn,
                                    void *transport_ctx);

// DEPRECATED for bridge-backed dispatchers: use cudaq_dispatcher_set_bridge,
// which makes this call itself.
//
// CPU data-plane wiring for the host unified path. `cpu_dataplane` is a
// `cudaq_cpu_dataplane_t*` obtained from the bridge. When set, the dispatcher
// runs the single-thread unified loop over it.  The struct is copied, as
// cudaq_dispatcher_set_ringbuffer copies its ring; the provider state its
// `ctx` points to must outlive the dispatcher.
cudaq_status_t
cudaq_dispatcher_set_cpu_dataplane(cudaq_dispatcher_t *dispatcher,
                                   void *cpu_dataplane);

// Start/stop
cudaq_status_t cudaq_dispatcher_start(cudaq_dispatcher_t *dispatcher);
cudaq_status_t cudaq_dispatcher_stop(cudaq_dispatcher_t *dispatcher);

// Stats
cudaq_status_t cudaq_dispatcher_get_processed(cudaq_dispatcher_t *dispatcher,
                                              uint64_t *out_packets);

//==============================================================================
// Ring buffer slot helpers (producer / consumer side)
//==============================================================================
// These encapsulate the RPC wire format and flag-signalling protocol so that
// producers and consumers don't need to know about magic constants, the
// "address-as-flag" convention, or the tx_flags state machine.

// Write an RPC request (RPCHeader + payload) into slot `slot_idx`.
// payload_len must satisfy CUDAQ_RPC_HEADER_SIZE + payload_len <= rx_stride_sz.
cudaq_status_t cudaq_host_ringbuffer_write_rpc_request(
    const cudaq_ringbuffer_t *rb, uint32_t slot_idx, uint32_t function_id,
    const void *payload, uint32_t payload_len, uint32_t request_id,
    uint64_t ptp_timestamp);

// Signal that slot `slot_idx` has data ready for the dispatcher.
// Stores the host address of the slot into rx_flags_host[slot_idx].
void cudaq_host_ringbuffer_signal_slot(const cudaq_ringbuffer_t *rb,
                                       uint32_t slot_idx);

// Poll tx_flags_host[slot_idx] and classify the result.
// If status == CUDAQ_TX_ERROR and out_cuda_error is non-NULL, the CUDA error
// code is written there.
cudaq_tx_status_t
cudaq_host_ringbuffer_poll_tx_flag(const cudaq_ringbuffer_t *rb,
                                   uint32_t slot_idx, int *out_cuda_error);

// Check whether a slot is available for reuse (both rx and tx flags are 0).
int cudaq_host_ringbuffer_slot_available(const cudaq_ringbuffer_t *rb,
                                         uint32_t slot_idx);

// Clear tx_flags_host[slot_idx] after consuming the response.
void cudaq_host_ringbuffer_clear_slot(const cudaq_ringbuffer_t *rb,
                                      uint32_t slot_idx);

// Release a worker back to the idle pool after the graph has completed.
// This is the consumer-side counterpart to the dispatcher's internal
// idle_mask acquisition — without this call the worker stays "busy" forever.
cudaq_status_t cudaq_host_release_worker(cudaq_dispatcher_t *dispatcher,
                                         int worker_id);

// Force eager CUDA module loading for dispatch kernels (occupancy query).
// Call before cudaq_dispatcher_start() to avoid lazy-loading deadlocks.
cudaError_t cudaq_dispatch_kernel_query_occupancy(int *out_blocks,
                                                  uint32_t threads_per_block);

// Push the shared_ring_mode flag into the DEVICE_LOOP kernel's __constant__
// memory.  Must be called BEFORE cudaq_dispatcher_start() launches the
// device kernel; otherwise the kernel will start with shared_ring_mode=0.
//
// IMPORTANT: cudaq_dispatcher_start() does NOT call this for you. The
// __constant__ symbol lives in libcudaq-realtime-dispatch.a, which is
// linked directly into consumers (not into libcudaq-realtime.so), so the
// dispatcher manager cannot reach the symbol from inside the shared
// library.  Consumers that set config.shared_ring_mode = 1 must also call
// cudaq_dispatch_kernel_set_shared_ring_mode(1) before starting the
// dispatcher.  The HOST_LOOP path reads config.shared_ring_mode directly
// and does NOT require this call.
//
// CUDAQ_REALTIME_DISPATCH_API: see cudaq_launch_dispatch_kernel_regular for
// the rationale -- consumers (e.g. cuda-qx's libcudaq-qec-realtime-decoding.so)
// resolve this entry point via dlsym(RTLD_DEFAULT, ...) at runtime.
CUDAQ_REALTIME_DISPATCH_API cudaError_t
cudaq_dispatch_kernel_set_shared_ring_mode(uint32_t enabled);

// Run-time device link of a DEVICE + UNIFIED dispatcher.
//
// The unified dispatch kernel, the bridge's device hooks and the
// application's DEVICE_CALL handlers come from three binaries, and must share
// one CUDA module.  After cudaq_dispatcher_set_bridge, the application asks
// for each handler's address with cudaq_dispatcher_get_device_handler; the
// first call links the module.  See realtime/docs/device_code_contract.md.

// What cudaq_dispatcher_get_device_handler calls, with `images` the
// relocatable device code of the binary that holds the handler.  Call that
// wrapper rather than this.
//
// The first call on a DEVICE + UNIFIED dispatcher, after
// cudaq_dispatcher_set_bridge, links the unified dispatch kernel, the bridge's
// device hooks and `images` into one CUDA module with nvJitLink and loads it;
// a failed link leaves the dispatcher unlinked, so a later call retries.
// Later calls reuse the module and must pass the same images: all handlers of
// one dispatcher come from one binary.  Refused for a bridge that supplies a
// launch override.
cudaq_status_t cudaq_dispatcher_get_device_handler_in(
    cudaq_dispatcher_t *dispatcher, const cudaq_device_image_t *images,
    size_t count, const char *name, void **out_fn);

// Address, in the dispatcher's linked module, of the DEVICE_CALL handler
// `name`, exported with CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(name).  Call it
// from the binary that holds the handler: it hands the dispatcher that
// binary's device code, and the first call links the module.  Every
// DEVICE_CALL entry of the function table must hold an address returned here:
// start() refuses any other, since an address taken in the application's own
// module is not valid in the linked one.
//
// `static` on purpose: each binary must run its own copy, which reads its own
// device code.
static inline cudaq_status_t
cudaq_dispatcher_get_device_handler(cudaq_dispatcher_t *dispatcher,
                                    const char *name, void **out_fn) {
  cudaq_device_image_t image;
  const cudaq_status_t status = cudaq_device_code_of_this_binary(&image);
  if (status != CUDAQ_OK)
    return status;
  return cudaq_dispatcher_get_device_handler_in(dispatcher, &image, 1, name,
                                                out_fn);
}

#ifdef __cplusplus
}
#endif
