/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file unified_dispatch_kernel.cu
/// @brief Hololink/DOCA unified dispatch: RDMA RX + RPC dispatch + RDMA TX
/// in one GPU kernel, using Hololink's gpu_roce_transceiver.cuh device
/// functions for WQE preparation and TX.
///
/// Two TX paths selected at runtime via the use_bf flag:
///   dGPU: send_bf (BlueFlame, shared-memory WQE) -- lowest latency.
///   iGPU: send (NIC_HANDLER_AUTO -> CPU proxy) -- required because the
///         GPU cannot ring the NIC doorbell directly on integrated GPUs.
///
/// Compiled into libcudaq-realtime-bridge-hololink.so (transport-specific).
/// The core libcudaq-realtime-dispatch.a no longer contains this file.

#include "cudaq/realtime/daemon/bridge/hololink/hololink_doca_transport_ctx.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cuda_runtime.h>
#include <cstdint>

#include "gpu_roce_transceiver.cuh"

using namespace cudaq::realtime;

//==============================================================================
// Device helpers
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
// Two TX paths selected at runtime via the use_bf flag:
//   use_bf=true  (dGPU): prepare_send_shared + send_bf  (BlueFlame, smem WQE)
//   use_bf=false (iGPU): prepare_receive_send + send    (NIC_HANDLER_AUTO)
//
// Receive WQEs are pre-posted by the host before this kernel launches.
//==============================================================================

__global__ void hololink_unified_dispatch_kernel(
    struct doca_gpu_dev_verbs_qp *qp, volatile int *shutdown_flag,
    std::uint8_t *ring_buf, std::size_t ring_buf_stride_sz,
    std::uint32_t ring_buf_mkey, std::uint32_t ring_buf_stride_num,
    std::size_t frame_size, cudaq_function_entry_t *function_table,
    std::size_t func_count, std::uint64_t *stats, int use_bf) {
  if (qp == nullptr)
    return;

  auto *cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
  auto *cqe = reinterpret_cast<std::uint8_t *>(
      __ldg(reinterpret_cast<uintptr_t *>(&cq_rq->cqe_daddr)));
  const std::uint32_t cqe_mask = __ldg(&cq_rq->cqe_num) - 1;

  const bool use_inline = (frame_size <= MAX_SEND_INLINE_WQE);

  // Receive WQEs are pre-posted by the host (GpuRoceTransceiverPrepareKernel
  // in start() on dGPU, or hololink_prepare_receive_send() on iGPU).
  __shared__ struct doca_gpu_dev_verbs_wqe wqe_sh;

  if (use_bf) {
    prepare_send_shared(qp, &wqe_sh, frame_size, ring_buf_mkey);
  } else {
    prepare_receive_send(qp, frame_size, ring_buf_mkey);
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
      repost_receive(qp, sq_wqe_idx);
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

    auto buffer_addr =
        static_cast<std::uint64_t>(ring_buf_stride_sz) * stride;
    if (use_bf) {
      // dGPU: send first, then repost (original order).  Reposting before
      // send adds ~400ns by serializing a PCIe write ahead of BlueFlame.
      if (!use_inline) {
        send_bf<GPU_ROCE_MAX_FRAME_SIZE_0B>(qp, &wqe_sh, sq_wqe_idx,
                                            buffer_addr);
      } else {
        send_bf<GPU_ROCE_MAX_FRAME_SIZE_44B>(
            qp, &wqe_sh, sq_wqe_idx,
            reinterpret_cast<std::uint64_t>(slot));
      }
      sq_wqe_idx++;
      repost_receive(qp, sq_wqe_idx);
      cq_ticket = sq_wqe_idx;
    } else {
      // iGPU: repost first, then send.  The CPU proxy may batch doorbell
      // writes, so ensure the NIC has a fresh receive WQE before any delay.
      sq_wqe_idx++;
      repost_receive(qp, sq_wqe_idx);
      cq_ticket = sq_wqe_idx;
      if (!use_inline) {
        send<GPU_ROCE_MAX_FRAME_SIZE_0B>(qp, sq_wqe_idx - 1, buffer_addr);
      } else {
        send<GPU_ROCE_MAX_FRAME_SIZE_44B>(
            qp, sq_wqe_idx - 1, reinterpret_cast<std::uint64_t>(slot));
      }
    }

    packet_count++;
  }

  atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
}

//==============================================================================
// Host launch wrapper -- matches cudaq_unified_launch_fn_t signature.
//==============================================================================

extern "C" void hololink_launch_unified_dispatch(
    void *transport_ctx, cudaq_function_entry_t *function_table,
    size_t func_count, volatile int *shutdown_flag, uint64_t *stats,
    cudaStream_t stream) {
  auto *ctx = static_cast<hololink_doca_transport_ctx *>(transport_ctx);

  hololink_unified_dispatch_kernel<<<1, 1, 0, stream>>>(
      static_cast<struct doca_gpu_dev_verbs_qp *>(ctx->gpu_dev_qp),
      shutdown_flag, ctx->rx_ring_data, ctx->rx_ring_stride_sz,
      ctx->rx_ring_mkey, ctx->rx_ring_stride_num, ctx->frame_size,
      function_table, func_count, stats, ctx->use_bf);
}
