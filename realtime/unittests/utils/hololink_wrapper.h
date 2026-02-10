/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hololink_wrapper.h
/// @brief C interface to Hololink GpuRoceTransceiver.
///
/// This wrapper avoids fmt library conflicts between Hololink (which uses
/// Holoscan's fmt) and CUDA files compiled by nvcc.

#ifndef HOLOLINK_WRAPPER_H
#define HOLOLINK_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for GpuRoceTransceiver
typedef void *hololink_transceiver_t;

//==============================================================================
// Transceiver lifecycle
//==============================================================================

/**
 * Create a new Hololink transceiver.
 *
 * @param device_name IB device name (e.g., "rocep1s0f0")
 * @param ib_port IB port number
 * @param frame_size Size of each frame (cu_frame_size)
 * @param page_size Size of each page/slot (cu_page_size)
 * @param num_pages Number of pages (ring buffer slots)
 * @param peer_ip Peer IP address (use "0.0.0.0" for deferred connection)
 * @param forward 1 to run forward (echo) kernel
 * @param rx_only 1 to run RX-only kernel
 * @param tx_only 1 to run TX-only kernel
 * @return Handle to transceiver, or NULL on failure
 */
hololink_transceiver_t hololink_create_transceiver(const char *device_name,
                                                   int ib_port,
                                                   size_t frame_size,
                                                   size_t page_size,
                                                   unsigned num_pages,
                                                   const char *peer_ip,
                                                   int forward, int rx_only,
                                                   int tx_only);

/**
 * Destroy a transceiver and free resources.
 */
void hololink_destroy_transceiver(hololink_transceiver_t handle);

/**
 * Start the transceiver (initializes DOCA resources, creates QP/CQ).
 * @return 1 on success, 0 on failure
 */
int hololink_start(hololink_transceiver_t handle);

/**
 * Close the transceiver (signals shutdown).
 */
void hololink_close(hololink_transceiver_t handle);

/**
 * Run the blocking monitor (launches GPU kernels and waits).
 * This function blocks until close() is called.
 */
void hololink_blocking_monitor(hololink_transceiver_t handle);

//==============================================================================
// QP information (for RDMA setup)
//==============================================================================

uint32_t hololink_get_qp_number(hololink_transceiver_t handle);
uint32_t hololink_get_rkey(hololink_transceiver_t handle);
uint64_t hololink_get_buffer_addr(hololink_transceiver_t handle);

/**
 * Get the local GID for this transceiver.
 * @param handle Transceiver handle
 * @param gid_out Buffer to receive 16-byte GID
 * @return 1 on success, 0 on failure
 */
int hololink_get_gid(hololink_transceiver_t handle, uint8_t *gid_out);

//==============================================================================
// Deferred QP connection
//==============================================================================

/**
 * Connect the QP to a remote peer (for deferred connection mode).
 * Call this after start() when peer_ip was "0.0.0.0".
 * @param handle Transceiver handle
 * @param remote_gid 16-byte remote GID
 * @param remote_qpn Remote QP number
 * @return 1 on success, 0 on failure
 */
int hololink_reconnect_qp(hololink_transceiver_t handle,
                           const uint8_t *remote_gid, uint32_t remote_qpn);

//==============================================================================
// Ring buffer access
//==============================================================================

/** Get device pointer to RX ring data buffer. */
void *hololink_get_rx_ring_data_addr(hololink_transceiver_t handle);

/** Get device pointer to RX ring flag array. */
uint64_t *hololink_get_rx_ring_flag_addr(hololink_transceiver_t handle);

/** Get device pointer to TX ring data buffer. */
void *hololink_get_tx_ring_data_addr(hololink_transceiver_t handle);

/** Get device pointer to TX ring flag array. */
uint64_t *hololink_get_tx_ring_flag_addr(hololink_transceiver_t handle);

/** Get host-accessible pointer to TX ring flag array. */
uint64_t *hololink_get_tx_ring_flag_host_addr(hololink_transceiver_t handle);

/** Get host-accessible pointer to RX ring flag array. */
uint64_t *hololink_get_rx_ring_flag_host_addr(hololink_transceiver_t handle);

/** Force eager CUDA module loading by querying kernel occupancy.
 *  Call before launching any persistent kernels.
 *  Returns true on success (all kernels valid). */
bool hololink_query_kernel_occupancy(void);

/** Get the page (slot) size configured for this transceiver. */
size_t hololink_get_page_size(hololink_transceiver_t handle);

/** Get the number of pages (slots) configured for this transceiver. */
unsigned hololink_get_num_pages(hololink_transceiver_t handle);

#ifdef __cplusplus
}
#endif

#endif // HOLOLINK_WRAPPER_H
