/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file unified_dispatch_kernel.cu
/// @brief Single-kernel (unified) dispatch: RDMA RX + RPC dispatch + RDMA TX
/// in one GPU kernel, using DOCA GPUNetIO verbs for lowest latency.
///
/// This is a transport-specific implementation for DOCA/Hololink.  The kernel
/// is launched through the transport-agnostic dispatcher API via
/// cudaq_unified_launch_fn_t + void* transport_ctx.

#include "cudaq/realtime/daemon/dispatcher/unified_dispatch_kernel.cuh"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <doca_gpunetio_dev_verbs_twosided.cuh>

#define DSEG_SIZE_2 12
#define DSEG_SIZE_3 28
#define MAX_SEND_INLINE_WQE 44

using namespace cudaq::realtime;

//==============================================================================
// Device helpers -- thin wrappers around DOCA verbs calls.
// These match the patterns used by Hololink's gpu_roce_transceiver_kernel.cu
// (prepare_receive_send, forward_bf, send_bf, receive, repost_receive).
//==============================================================================

/// Spin-poll the CQE owner bit with periodic shutdown_flag checks.  Inlines
/// the DOCA CQ state update (fence + consumer-index advance) to avoid the
/// double CQE read that calling poll_cq_at would cause.  Returns UINT32_MAX
/// on shutdown; otherwise returns the stride from the CQE immediate field.
__device__ static inline std::uint32_t
unified_poll_receive(struct doca_gpu_dev_verbs_cq *cq_rq, std::uint8_t *cqe,
                     std::uint32_t cqe_mask,
                     doca_gpu_dev_verbs_ticket_t ticket,
                     volatile int *shutdown_flag) {
  auto *cqe64 = reinterpret_cast<struct mlx5_cqe64 *>(
      cqe + ((ticket & cqe_mask) * DOCA_GPUNETIO_VERBS_CQE_SIZE));
  std::uint32_t cqe_num = cqe_mask + 1;
  int spin = 0;
  std::uint8_t opown;
  do {
    opown = doca_gpu_dev_verbs_load_relaxed_sys_global(
        reinterpret_cast<uint8_t *>(&cqe64->op_own));
    if (!((opown & MLX5_CQE_OWNER_MASK) ^ !!(ticket & cqe_num)))
      break;
    if (++spin >= 1024) {
      spin = 0;
      if (*shutdown_flag)
        return UINT32_MAX;
    }
  } while (true);
  doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
  doca_gpu_dev_verbs_atomic_max<std::uint64_t,
                                DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      &cq_rq->cqe_ci, ticket + 1);
  return doca_gpu_dev_verbs_bswap32(cqe64->imm_inval_pkey) & 0xFFF;
}

__device__ static inline void
unified_repost_receive(struct doca_gpu_dev_verbs_qp *qp,
                       std::uint64_t rwqe_idx) {
  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                            DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA,
                            DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                            DOCA_GPUNETIO_VERBS_QP_RQ>(qp, rwqe_idx + 1);
}

__device__ static inline const cudaq_function_entry_t *
unified_lookup_entry(std::uint32_t function_id,
                     cudaq_function_entry_t *entries, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (entries[i].function_id == function_id)
      return &entries[i];
  }
  return nullptr;
}

//==============================================================================
// Unified dispatch kernel -- single thread, single block.
//
// The WQE format (inline vs pointer-based) must match what Hololink's
// prepare_receive_send kernel wrote into the hardware WQE buffer.  For
// frame_size <= MAX_SEND_INLINE_WQE (44), inline send is used and the
// response data is copied into the WQE; otherwise, pointer-based send.
//==============================================================================

__global__ void unified_dispatch_kernel_bf(
    struct doca_gpu_dev_verbs_qp *qp, volatile int *shutdown_flag,
    std::uint8_t *ring_buf, std::size_t ring_buf_stride_sz,
    std::uint32_t ring_buf_mkey, std::uint32_t ring_buf_stride_num,
    std::size_t frame_size, cudaq_function_entry_t *function_table,
    std::size_t func_count, std::uint64_t *stats) {
  if (qp == nullptr)
    return;

  auto *cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
  auto *cqe = reinterpret_cast<std::uint8_t *>(
      __ldg(reinterpret_cast<uintptr_t *>(&cq_rq->cqe_daddr)));
  const std::uint32_t cqe_mask = __ldg(&cq_rq->cqe_num) - 1;

  const bool use_inline = (frame_size <= MAX_SEND_INLINE_WQE);

  // Prepare WQE in shared memory -- must match the format that
  // prepare_receive_send wrote into the hardware WQE buffer.
  __shared__ struct doca_gpu_dev_verbs_wqe wqe_sh;
  {
    struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
    cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(
        (0u << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
        DOCA_GPUNETIO_MLX5_OPCODE_SEND);
    cseg.fm_ce_se = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

    if (!use_inline) {
      cseg.qpn_ds =
          doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | 2);
      wqe_sh.dseg1.byte_count =
          doca_gpu_dev_verbs_bswap32(static_cast<std::uint32_t>(frame_size));
      wqe_sh.dseg1.lkey = ring_buf_mkey;
    } else {
      std::uint32_t ds;
      if (frame_size <= DSEG_SIZE_2)
        ds = 2;
      else if (frame_size <= DSEG_SIZE_3)
        ds = 3;
      else
        ds = 4;
      cseg.qpn_ds =
          doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | ds);
      wqe_sh.dseg1.byte_count = doca_gpu_dev_verbs_bswap32(
          static_cast<std::uint32_t>(frame_size) | MLX5_INLINE_SEG);
    }

    doca_gpu_dev_verbs_store_wqe_seg(
        reinterpret_cast<std::uint64_t *>(&wqe_sh.dseg0),
        reinterpret_cast<std::uint64_t *>(&cseg));
  }

  doca_gpu_dev_verbs_ticket_t cq_ticket = 0;
  std::uint64_t sq_wqe_idx = 0;
  std::uint64_t packet_count = 0;

  while (true) {
    std::uint32_t stride =
        unified_poll_receive(cq_rq, cqe, cqe_mask, cq_ticket, shutdown_flag);
    if (stride == UINT32_MAX)
      break;
    if (stride >= ring_buf_stride_num) {
      sq_wqe_idx++;
      unified_repost_receive(qp, sq_wqe_idx);
      cq_ticket = sq_wqe_idx;
      continue;
    }

    auto *slot =
        ring_buf + static_cast<std::uint64_t>(stride) * ring_buf_stride_sz;
    auto *header = reinterpret_cast<RPCHeader *>(slot);

    if (header->magic == RPC_MAGIC_REQUEST) {
      std::uint32_t function_id = header->function_id;
      std::uint32_t arg_len = header->arg_len;
      std::uint32_t request_id = header->request_id;
      std::uint64_t ptp_timestamp = header->ptp_timestamp;

      const cudaq_function_entry_t *entry =
          unified_lookup_entry(function_id, function_table, func_count);

      int status = -1;
      std::uint32_t result_len = 0;

      if (entry != nullptr &&
          entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL) {
        auto func = reinterpret_cast<DeviceRPCFunction>(
            entry->handler.device_fn_ptr);
        void *arg_buffer = static_cast<void *>(header + 1);
        auto *output_buffer = slot + sizeof(RPCResponse);
        auto max_result_len = static_cast<std::uint32_t>(
            frame_size - sizeof(RPCResponse));

        status =
            func(arg_buffer, output_buffer, arg_len, max_result_len,
                 &result_len);
      }

      auto *response = reinterpret_cast<RPCResponse *>(slot);
      response->magic = RPC_MAGIC_RESPONSE;
      response->status = status;
      response->result_len = result_len;
      response->request_id = request_id;
      response->ptp_timestamp = ptp_timestamp;
    }

    // --- Send response ---
    auto wqe_idx16 = static_cast<std::uint16_t>(sq_wqe_idx);
    wqe_sh.snd_cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(
        (static_cast<std::uint32_t>(wqe_idx16)
         << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
        DOCA_GPUNETIO_MLX5_OPCODE_SEND);

    if (!use_inline) {
      wqe_sh.dseg1.addr = doca_gpu_dev_verbs_bswap64(
          static_cast<std::uint64_t>(ring_buf_stride_sz) * stride);
    } else {
      // Copy response data inline into the WQE, matching the layout used
      // by Hololink's send_bf<GPU_ROCE_MAX_FRAME_SIZE_*> templates.
      auto *src = reinterpret_cast<std::uint32_t *>(slot);
      if (frame_size >= 4)
        wqe_sh.dseg1.lkey = src[0];
      if (frame_size >= 8)
        reinterpret_cast<std::uint32_t *>(&wqe_sh.dseg1)[2] = src[1];
      if (frame_size >= 12)
        reinterpret_cast<std::uint32_t *>(&wqe_sh.dseg1)[3] = src[2];
      if (frame_size >= 16)
        wqe_sh.dseg2.byte_count = src[3];
      if (frame_size >= 20)
        wqe_sh.dseg2.lkey = src[4];
      if (frame_size >= 24)
        reinterpret_cast<std::uint32_t *>(&wqe_sh.dseg2)[2] = src[5];
      if (frame_size >= 28)
        reinterpret_cast<std::uint32_t *>(&wqe_sh.dseg2)[3] = src[6];
      if (frame_size >= 32)
        wqe_sh.dseg3.byte_count = src[7];
      if (frame_size >= 36)
        wqe_sh.dseg3.lkey = src[8];
      if (frame_size >= 40)
        reinterpret_cast<std::uint32_t *>(&wqe_sh.dseg3)[2] = src[9];
      if (frame_size >= 44)
        reinterpret_cast<std::uint32_t *>(&wqe_sh.dseg3)[3] = src[10];
    }

    doca_gpu_dev_verbs_mark_wqes_ready<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(qp, sq_wqe_idx,
                                                        sq_wqe_idx);

    doca_gpu_dev_verbs_submit_bf<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                 DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(
        qp, sq_wqe_idx + 1, &wqe_sh);

    sq_wqe_idx++;
    unified_repost_receive(qp, sq_wqe_idx);
    cq_ticket = sq_wqe_idx;
    packet_count++;
  }

  atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
}

//==============================================================================
// Host launch wrapper -- matches cudaq_unified_launch_fn_t signature.
//==============================================================================

extern "C" void cudaq_launch_unified_dispatch_kernel(
    void *transport_ctx, cudaq_function_entry_t *function_table,
    size_t func_count, volatile int *shutdown_flag, uint64_t *stats,
    cudaStream_t stream) {
  auto *ctx = static_cast<doca_transport_ctx *>(transport_ctx);

  unified_dispatch_kernel_bf<<<1, 1, 0, stream>>>(
      static_cast<struct doca_gpu_dev_verbs_qp *>(ctx->gpu_dev_qp),
      shutdown_flag, ctx->rx_ring_data, ctx->rx_ring_stride_sz,
      ctx->rx_ring_mkey, ctx->rx_ring_stride_num, ctx->frame_size,
      function_table, func_count, stats);
}
