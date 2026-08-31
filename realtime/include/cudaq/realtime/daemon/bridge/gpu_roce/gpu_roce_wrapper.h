/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file gpu_roce_wrapper.h
/// @brief C interface to GpuRoceTransceiver.
///
/// This wrapper avoids `fmt` library conflicts between GpuRoceTransceiver
/// (which uses Holoscan's `fmt`) and CUDA files compiled by nvcc.

#ifndef GPU_ROCE_WRAPPER_H
#define GPU_ROCE_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for GpuRoceTransceiver
typedef void *gpu_roce_transceiver_t;

//==============================================================================
// Transceiver lifecycle
//==============================================================================

/**
 * Create a new GpuRoceTransceiver.
 *
 * @param device_name IB device name (e.g., "rocep1s0f0")
 * @param ib_port IB port number
 * @param tx_ibv_qp Remote QP number (FPGA default: 2)
 * @param gpu_id CUDA GPU device ID for DOCA GPUNetIO
 * @param frame_size Size of each frame (cu_frame_size)
 * @param page_size Size of each page/slot (cu_page_size)
 * @param num_pages Number of pages (ring buffer slots)
 * @param peer_ip Peer IP address
 * @param forward 1 to run forward (echo) kernel
 * @param rx_only 1 to run RX-only kernel
 * @param tx_only 1 to run TX-only kernel
 * @return Handle to transceiver, or NULL on failure
 */
gpu_roce_transceiver_t gpu_roce_create_transceiver(
    const char *device_name, int ib_port, unsigned tx_ibv_qp, int gpu_id,
    size_t frame_size, size_t page_size, unsigned num_pages,
    const char *peer_ip, int forward, int rx_only, int tx_only);

/**
 * Destroy a transceiver and free resources.
 */
void gpu_roce_destroy_transceiver(gpu_roce_transceiver_t handle);

/**
 * Start the transceiver (initializes DOCA resources, creates QP/CQ).
 * @return 1 on success, 0 on failure
 */
int gpu_roce_start(gpu_roce_transceiver_t handle);

/**
 * Close the transceiver (signals shutdown).
 */
void gpu_roce_close(gpu_roce_transceiver_t handle);

/**
 * Run the blocking monitor (launches GPU kernels and waits).
 * This function blocks until close() is called.
 */
void gpu_roce_blocking_monitor(gpu_roce_transceiver_t handle);

/**
 * Force CPU+GPU accessible allocation for ring flags and data only.
 * Does NOT change CQ/QP UMEMs or TX kernel handler (BlueFlame preserved).
 * Must be called after create and before start().
 * Required when a CPU thread needs to read ring flags/data directly
 * (e.g. HOST_LOOP dispatcher on Grace-Blackwell dGPU).
 */
void gpu_roce_set_cpu_ring_buffers(gpu_roce_transceiver_t handle, int enable);

//==============================================================================
// QP information (for RDMA setup)
//==============================================================================

uint32_t gpu_roce_get_qp_number(gpu_roce_transceiver_t handle);
uint32_t gpu_roce_get_rkey(gpu_roce_transceiver_t handle);
uint64_t gpu_roce_get_buffer_addr(gpu_roce_transceiver_t handle);

/** Get the DOCA GPU device QP handle (doca_gpu_dev_verbs_qp*).
 *  Needed by the unified dispatch kernel for direct DOCA verbs calls. */
void *gpu_roce_get_gpu_dev_qp(gpu_roce_transceiver_t handle);

//==============================================================================
// Ring buffer access
//==============================================================================

/** Get device pointer to RX ring data buffer. */
void *gpu_roce_get_rx_ring_data_addr(gpu_roce_transceiver_t handle);

/** Get device pointer to RX ring flag array. */
uint64_t *gpu_roce_get_rx_ring_flag_addr(gpu_roce_transceiver_t handle);

/** Get device pointer to TX ring data buffer. */
void *gpu_roce_get_tx_ring_data_addr(gpu_roce_transceiver_t handle);

/** Get device pointer to TX ring flag array. */
uint64_t *gpu_roce_get_tx_ring_flag_addr(gpu_roce_transceiver_t handle);

/** Force eager CUDA module loading by querying kernel occupancy.
 *  Call before launching any persistent kernels.
 *  Returns true on success (all kernels valid). */
bool gpu_roce_query_kernel_occupancy(void);

/** Get the page (slot) size configured for this transceiver. */
size_t gpu_roce_get_page_size(gpu_roce_transceiver_t handle);

/** Get the number of pages (slots) configured for this transceiver. */
unsigned gpu_roce_get_num_pages(gpu_roce_transceiver_t handle);

/**
 * Pre-post receive WQEs and initialize the send WQE template.
 *
 * Calls GpuRoceTransceiverPrepareKernel with the transceiver's GPU QP,
 * frame_size, and RX ring `rkey`.  Use this when start() didn't call the
 * prepare kernel (e.g. unified mode on iGPU with forward=false).
 *
 * @param frame_size  Actual frame/payload size for send WQE setup
 * @return 1 on success, 0 on failure
 */
int gpu_roce_prepare_receive_send(gpu_roce_transceiver_t handle,
                                  size_t frame_size);

#ifdef __cplusplus
}
#endif

#endif // GPU_ROCE_WRAPPER_H
