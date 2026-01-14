/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/registry/gpu_function_registry.h"
#include "cudaq/nvqlink/network/serialization/gpu_input_stream.h"
#include "cudaq/nvqlink/network/serialization/gpu_output_stream.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <doca_error.h>
#include <doca_gpunetio_dev_verbs_twosided.cuh>

#include <cstdint>

namespace cudaq::nvqlink {

constexpr std::uint32_t RWQE_TERMINATE_KEY = 0x100;
// TODO: We always use non-inline sends. Add inline send variants if needed.

struct __align__(8) RPCHeader {
  std::uint64_t sequence_number;
  std::uint32_t function_id;
  std::uint32_t payload_length;
};

struct __align__(8) RPCResponse {
  std::uint64_t sequence_number;
  std::int32_t status;
  std::uint32_t result_length;
};

/// @brief Receive a single packet from CQ
///
__device__ inline std::uint32_t
receive_packet(doca_gpu_dev_verbs_cq *cq_rq, std::uint8_t *cqe_buffer,
               const std::uint32_t cqe_mask,
               const doca_gpu_dev_verbs_ticket_t out_ticket) {
  doca_gpu_dev_verbs_poll_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                DOCA_GPUNETIO_VERBS_QP_RQ>(cq_rq, out_ticket);

  mlx5_cqe64 *cqe64 = reinterpret_cast<mlx5_cqe64 *>(
      cqe_buffer + ((out_ticket & cqe_mask) * DOCA_GPUNETIO_VERBS_CQE_SIZE));

  return doca_gpu_dev_verbs_bswap32(cqe64->imm_inval_pkey) & 0xFFF;
}

/// @brief Build and submit response WQE
///
__device__ inline void
send_response(doca_gpu_dev_verbs_qp *qp, std::uint64_t wqe_idx,
              std::uint8_t *buffer, const std::uint32_t page_size,
              const std::uint32_t buffer_mkey, const unsigned page,
              const std::uint32_t response_size) {
  // Debug: Print send parameters
  printf("DOCA send_response: wqe_idx=%lu, page=%u, page_size=%u, "
         "response_size=%u, mkey=0x%x\n",
         (unsigned long)wqe_idx, page, page_size, response_size, buffer_mkey);
  printf("DOCA send_response: addr=0x%x (page*page_size)\n", page * page_size);

  // COPY FROM HOLOLINK: Lines 84-91 (WQE pointer and index setup)
  std::uint16_t wqe_idx_16 = static_cast<std::uint16_t>(wqe_idx);
  doca_gpu_dev_verbs_wqe *wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

  wqe_ptr->snd_cseg.opmod_idx_opcode =
      doca_gpu_dev_verbs_bswap32((static_cast<std::uint32_t>(wqe_idx_16)
                                  << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                 DOCA_GPUNETIO_MLX5_OPCODE_SEND);

  // CRITICAL: Use iova-relative address, NOT GPU virtual address!
  // Memory is registered with iova=0, so addr should be (page * page_size)
  // offset from 0. This matches Hololink: wqe_ptr->dseg1.addr =
  // bswap64(cu_page_size * page)
  wqe_ptr->dseg1.addr =
      doca_gpu_dev_verbs_bswap64(static_cast<std::uint64_t>(page) * page_size);
  wqe_ptr->dseg1.byte_count = doca_gpu_dev_verbs_bswap32(
      static_cast<std::uint32_t>(sizeof(RPCResponse) + response_size));
  wqe_ptr->dseg1.lkey = buffer_mkey;

  printf("DOCA send_response: WQE prepared - byte_count=%u, submitting...\n",
         static_cast<std::uint32_t>(sizeof(RPCResponse) + response_size));

  doca_gpu_dev_verbs_mark_wqes_ready<
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, wqe_idx, wqe_idx);

  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                            DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                            DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(
      qp, wqe_idx + 1);

  printf("DOCA send_response: SEND submitted to NIC\n");
}

/// @brief Repost receive WQE after processing
///
__device__ inline std::uint64_t repost_receive(doca_gpu_dev_verbs_qp *qp,
                                               std::uint64_t rwqe_idx) {
  // COPY FROM HOLOLINK: Lines 132-136
  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                            DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                            DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                            DOCA_GPUNETIO_VERBS_QP_RQ>(qp, rwqe_idx + 1);

  return rwqe_idx;
}

/// @brief Main RPC processing kernel
///
__global__ void
rpc_dispatch_kernel(doca_gpu_dev_verbs_qp *qp, std::uint32_t *exit_flag,
                    std::uint8_t *buffer, const std::uint32_t page_size,
                    const std::uint32_t buffer_mkey, const unsigned num_pages,
                    GPUFunctionRegistry *registry) {
  doca_gpu_dev_verbs_cq *cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
  std::uint8_t *cqe_buffer = reinterpret_cast<std::uint8_t *>(
      __ldg(reinterpret_cast<uintptr_t *>(&cq_rq->cqe_daddr)));
  const std::uint32_t cqe_mask = (__ldg(&cq_rq->cqe_num) - 1);

  doca_gpu_dev_verbs_ticket_t out_ticket = threadIdx.x;
  std::uint64_t wqe_idx = threadIdx.x;

  // Debug: Print kernel start (only once per block)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf(
        "DOCA RPC Kernel: Started, exit_flag=%u, num_pages=%u, registry=%p\n",
        *exit_flag, num_pages, registry);
    printf("DOCA RPC Kernel: page_size=%u, buffer_mkey=0x%x, buffer=%p\n",
           page_size, buffer_mkey, buffer);
  }

  // Kernel exits when exit_flag is set AND a packet arrives
  while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
    unsigned page = receive_packet(cq_rq, cqe_buffer, cqe_mask, out_ticket);

    if (page < num_pages) {
      // Parse RPC header from received data
      std::uint8_t *page_ptr = buffer + page * page_size;
      RPCHeader *header = reinterpret_cast<RPCHeader *>(page_ptr);
      std::uint8_t *payload = page_ptr + sizeof(RPCHeader);

      // Debug: Print received packet info
      if (threadIdx.x == 0) {
        printf("DOCA RPC: Received packet - page=%u, seq=%lu, func_id=%u, "
               "payload_len=%u\n",
               page, header->sequence_number, header->function_id,
               header->payload_length);
      }

      // Create streams for function dispatch
      GPUInputStream in(payload, header->payload_length);
      std::uint8_t *response_payload = page_ptr + sizeof(RPCResponse);
      std::size_t response_capacity = page_size - sizeof(RPCResponse);
      GPUOutputStream out(response_payload, response_capacity);

      // Dispatch to registered function
      std::int32_t status = 0;
      if (registry != nullptr) {
        GPUFunctionWrapper *func = registry->lookup(header->function_id);
        if (func != nullptr) {
          if (threadIdx.x == 0) {
            printf("DOCA RPC: Found function wrapper at %p\n", (void *)func);
            printf("DOCA RPC: Function pointer = %p\n", (void *)func->invoke);
            printf("DOCA RPC: Invoking function %u...\n", header->function_id);
          }

          // Call the function - if this crashes or hangs, the pointer is
          // invalid
          if (func->invoke != nullptr) {
            status = func->invoke(in, out);
            if (threadIdx.x == 0) {
              printf(
                  "DOCA RPC: Function returned status=%d, bytes_written=%zu\n",
                  status, out.bytes_written());
            }
          } else {
            status = -3;
            if (threadIdx.x == 0) {
              printf("DOCA RPC: ERROR - Function pointer is NULL!\n");
            }
          }
        } else {
          status = -1;
          printf("DOCA RPC: Function %u not found\n", header->function_id);
        }
      } else {
        status = -2;
        printf("DOCA RPC: Registry is null\n");
      }

      // Build response header
      RPCResponse response{.sequence_number = header->sequence_number,
                           .status = status,
                           .result_length =
                               static_cast<std::uint32_t>(out.bytes_written())};
      *reinterpret_cast<RPCResponse *>(page_ptr) = response;

      if (threadIdx.x == 0) {
        printf("DOCA RPC: Response built - seq=%lu, status=%d, result_len=%u\n",
               response.sequence_number, response.status,
               response.result_length);
      }

      // Only thread 0 sends (to avoid duplicate sends)
      if (threadIdx.x == 0) {
        // Debug: print values before call
        std::uint32_t bytes_written =
            static_cast<std::uint32_t>(out.bytes_written());
        printf("DOCA RPC: About to call send_response with page_size=%u, "
               "bytes_written=%u\n",
               page_size, bytes_written);
        send_response(qp, wqe_idx, buffer, page_size, buffer_mkey, page,
                      bytes_written);
      }
    } else {
      printf("DOCA RPC: Invalid page %u, ignoring\n", page);
    }
    wqe_idx += blockDim.x * gridDim.x;
    out_ticket = repost_receive(qp, wqe_idx);
  }
}

/// @brief Prepare WQEs kernel
///
__global__ void prepare_wqes_kernel(doca_gpu_dev_verbs_qp *qp,
                                    const std::uint32_t max_rpc_size,
                                    const std::uint32_t buffer_mkey) {
  // Debug: Log kernel execution
  if (threadIdx.x == 0) {
    printf("DOCA prepare_wqes_kernel: Starting with %d threads, "
           "max_rpc_size=%u, mkey=0x%x\n",
           blockDim.x, max_rpc_size, buffer_mkey);
  }

  doca_gpu_dev_verbs_ticket_t out_ticket;
  doca_gpu_dev_verbs_wqe *wqe_ptr;
  doca_gpu_dev_verbs_wqe_ctrl_seg cseg;

  doca_gpu_dev_verbs_recv(
      qp,
      doca_gpu_dev_verbs_addr{
          .addr = 0, .key = doca_gpu_dev_verbs_bswap32(RWQE_TERMINATE_KEY)},
      0, &out_ticket);

  wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, threadIdx.x);
  cseg.opmod_idx_opcode =
      doca_gpu_dev_verbs_bswap32((static_cast<std::uint32_t>(threadIdx.x)
                                  << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                 DOCA_GPUNETIO_MLX5_OPCODE_SEND);
  cseg.fm_ce_se = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

  cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | 2);
  wqe_ptr->dseg1.byte_count =
      doca_gpu_dev_verbs_bswap32(static_cast<std::uint32_t>(max_rpc_size));
  wqe_ptr->dseg1.lkey = buffer_mkey;

  doca_gpu_dev_verbs_store_wqe_seg(
      reinterpret_cast<std::uint64_t *>(&wqe_ptr->dseg0),
      reinterpret_cast<std::uint64_t *>(&cseg));

  __syncthreads(); // Ensure all threads have finished preparing their WQEs
  if (threadIdx.x == blockDim.x - 1) {
    printf("DOCA prepare_wqes_kernel: Submitting %d receive WQEs to NIC "
           "(out_ticket=%lu)\n",
           blockDim.x, (unsigned long)out_ticket);
    // Last thread submits all WQEs (0 to blockDim.x)
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                              DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                              DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                              DOCA_GPUNETIO_VERBS_QP_RQ>(qp, blockDim.x);
    printf("DOCA prepare_wqes_kernel: WQEs submitted successfully\n");
  }
}

extern "C" {

doca_error_t doca_prepare_wqes(cudaStream_t stream, doca_gpu_dev_verbs_qp *qp,
                               std::uint32_t max_rpc_size,
                               std::uint32_t buffer_mkey,
                               std::uint32_t num_wqes) {
  cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess) {
    fprintf(stderr, "[%s:%d] CUDA failed with %s\n", __FILE__, __LINE__,
            cudaGetErrorString(result));
    return DOCA_ERROR_BAD_STATE;
  }

  prepare_wqes_kernel<<<1, num_wqes, 0, stream>>>(qp, max_rpc_size,
                                                  buffer_mkey);

  result = cudaGetLastError();
  if (result != cudaSuccess) {
    fprintf(stderr, "[%s:%d] CUDA failed with %s\n", __FILE__, __LINE__,
            cudaGetErrorString(result));
    return DOCA_ERROR_BAD_STATE;
  }

  return DOCA_SUCCESS;
}

doca_error_t doca_rpc_kernel(cudaStream_t stream, doca_gpu_dev_verbs_qp *qp,
                             std::uint32_t *exit_flag, std::uint8_t *buffer,
                             std::uint32_t page_size, std::uint32_t buffer_mkey,
                             unsigned num_pages, GPUFunctionRegistry *registry,
                             std::uint32_t cuda_blocks,
                             std::uint32_t cuda_threads) {
  cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess) {
    fprintf(stderr, "[%s:%d] CUDA failed with %s\n", __FILE__, __LINE__,
            cudaGetErrorString(result));
    return DOCA_ERROR_BAD_STATE;
  }

  // Launch RPC dispatch kernel
  rpc_dispatch_kernel<<<cuda_blocks, cuda_threads, 0, stream>>>(
      qp, exit_flag, buffer, page_size, buffer_mkey, num_pages, registry);

  result = cudaGetLastError();
  if (result != cudaSuccess) {
    fprintf(stderr, "[%s:%d] CUDA failed with %s\n", __FILE__, __LINE__,
            cudaGetErrorString(result));
    return DOCA_ERROR_BAD_STATE;
  }

  return DOCA_SUCCESS;
}

} // extern "C"

} // namespace cudaq::nvqlink
