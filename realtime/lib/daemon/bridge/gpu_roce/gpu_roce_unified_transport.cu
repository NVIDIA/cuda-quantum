/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file gpu_roce_unified_transport.cu
/// @brief GpuRoceTransceiver/DOCA implementation of the unified dispatch
/// device data plane (dispatcher/unified_device_transport.cuh).
///
/// This is the only TU in the unified path that sees DOCA.  The kernel itself
/// lives in dispatcher/unified_dispatch_kernel.cu, compiled without any DOCA
/// or HSB include path; it reaches the wire exclusively through the three
/// hooks below, device-linked into this shared library by `nvcc -dlink`.
///
/// Two TX paths selected at runtime via the use_bf flag:
///   dGPU: send_bf (BlueFlame, shared-memory WQE) -- lowest latency.
///   iGPU: send (NIC_HANDLER_AUTO -> CPU proxy) -- required because the
///         GPU cannot ring the NIC doorbell directly on integrated GPUs.
///
/// Compiled into libcudaq-realtime-bridge-gpu-roce.so (transport-specific).

#include "cudaq/realtime/daemon/bridge/gpu_roce/gpu_roce_doca_transport_ctx.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/unified_device_transport.cuh"

#include <cstdint>
#include <cuda_runtime.h>

#include "gpu_roce_transceiver.cuh"

//==============================================================================
// Device-side session
//
// Holds what were kernel locals before the transport split: the derived DOCA
// handles, the RX/TX cursors, and the BlueFlame work-queue entry.  Lives in
// shared memory (see cudaq_dev_transport_attach), so it must stay trivially
// constructible -- no member initializers, no constructors.
//==============================================================================

namespace {

struct gpu_roce_session {
  struct doca_gpu_dev_verbs_qp *qp;
  struct doca_gpu_dev_verbs_cq *cq_rq;
  std::uint8_t *cqe;
  std::uint32_t cqe_mask;

  std::uint8_t *ring_buf;
  std::size_t stride_sz;
  std::uint32_t stride_num;
  std::size_t frame_size;

  int use_bf;      ///< Non-zero: BlueFlame TX (dGPU)
  bool use_inline; ///< Frame fits in an inline WQE

  volatile int *shutdown_flag;

  doca_gpu_dev_verbs_ticket_t cq_ticket;
  std::uint64_t sq_wqe_idx;

  /// Pre-built send WQE for the BlueFlame path.  `send_bf` indexes this by
  /// threadIdx.x, which is always 0 here (the kernel is <<<1,1>>>).
  struct doca_gpu_dev_verbs_wqe wqe_sh;
};

/// Spin-poll the CQE owner bit with periodic shutdown_flag checks.  Inlines
/// the DOCA CQ state update (fence + consumer-index advance) to avoid the
/// double CQE read that calling poll_cq_at would cause.  Returns UINT32_MAX
/// on shutdown; otherwise returns the stride from the CQE immediate field.
__device__ inline std::uint32_t poll_receive(gpu_roce_session &s) {
  auto *cqe64 = reinterpret_cast<struct mlx5_cqe64 *>(
      s.cqe + ((s.cq_ticket & s.cqe_mask) * DOCA_GPUNETIO_VERBS_CQE_SIZE));
  std::uint32_t cqe_num = s.cqe_mask + 1;
  int spin = 0;
  std::uint8_t opown;
  do {
    opown = doca_gpu_dev_verbs_load_relaxed_sys_global(
        reinterpret_cast<uint8_t *>(&cqe64->op_own));
    if (!((opown & MLX5_CQE_OWNER_MASK) ^ !!(s.cq_ticket & cqe_num)))
      break;
    if (++spin >= 1024) {
      spin = 0;
      if (*s.shutdown_flag)
        return UINT32_MAX;
    }
  } while (true);
  doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
  doca_gpu_dev_verbs_atomic_max<std::uint64_t,
                                DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      &s.cq_rq->cqe_ci, s.cq_ticket + 1);
  return doca_gpu_dev_verbs_bswap32(cqe64->imm_inval_pkey) & 0xFFF;
}

/// Hand the NIC a fresh receive WQE and step the completion cursor onto it.
__device__ inline void recycle_receive(gpu_roce_session &s) {
  s.sq_wqe_idx++;
  repost_receive(s.qp, s.sq_wqe_idx);
  s.cq_ticket = s.sq_wqe_idx;
}

} // namespace

//==============================================================================
// Device data-plane hooks
//==============================================================================

extern "C" __device__ void *
cudaq_dev_transport_attach(void *ctx, volatile int *shutdown_flag) {
  // Statically allocated per block.  Declaring it here rather than in the
  // kernel is what keeps doca_gpu_dev_verbs_wqe out of the core TU.
  __shared__ gpu_roce_session s;

  auto *tctx = static_cast<gpu_roce_doca_transport_ctx *>(ctx);
  if (tctx == nullptr)
    return nullptr;

  auto *qp = static_cast<struct doca_gpu_dev_verbs_qp *>(tctx->gpu_dev_qp);
  if (qp == nullptr)
    return nullptr;

  s.qp = qp;
  s.cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
  s.cqe = reinterpret_cast<std::uint8_t *>(
      __ldg(reinterpret_cast<uintptr_t *>(&s.cq_rq->cqe_daddr)));
  s.cqe_mask = __ldg(&s.cq_rq->cqe_num) - 1;

  s.ring_buf = tctx->rx_ring_data;
  s.stride_sz = tctx->rx_ring_stride_sz;
  s.stride_num = tctx->rx_ring_stride_num;
  s.frame_size = tctx->frame_size;

  s.use_bf = tctx->use_bf;
  s.use_inline = (tctx->frame_size <= MAX_SEND_INLINE_WQE);
  s.shutdown_flag = shutdown_flag;

  s.cq_ticket = 0;
  s.sq_wqe_idx = 0;

  // Receive WQEs are pre-posted by the host (GpuRoceTransceiverPrepareKernel
  // in start() on dGPU, or gpu_roce_prepare_receive_send() on iGPU); only the
  // send side is prepared here.
  if (s.use_bf)
    prepare_send_shared(qp, &s.wqe_sh, s.frame_size, tctx->rx_ring_mkey);
  else
    prepare_receive_send(qp, s.frame_size, tctx->rx_ring_mkey);

  return &s;
}

extern "C" __device__ cudaq_rx_dev_status_t
cudaq_dev_rx_poll(void *session, void **out_frame) {
  auto &s = *static_cast<gpu_roce_session *>(session);

  while (true) {
    std::uint32_t stride = poll_receive(s);
    // Only reachable once the shutdown flag is set; poll_receive spins
    // otherwise.
    if (stride == UINT32_MAX)
      return CUDAQ_RX_DEV_SHUTDOWN;

    // An out-of-range stride addresses no slot, so there is nothing to
    // dispatch: recycle the receive WQE and keep waiting rather than surface
    // a frame the caller could not read.
    if (stride >= s.stride_num) {
      recycle_receive(s);
      continue;
    }

    *out_frame = s.ring_buf + static_cast<std::uint64_t>(stride) * s.stride_sz;
    return CUDAQ_RX_DEV_READY;
  }
}

extern "C" __device__ void cudaq_dev_tx_publish(void *session, void *frame) {
  auto &s = *static_cast<gpu_roce_session *>(session);

  // The send descriptor wants the frame's offset into the registered buffer,
  // which rx_poll already turned into an address; recover it by difference
  // rather than keeping a parallel slot index.
  auto buffer_addr = static_cast<std::uint64_t>(
      static_cast<std::uint8_t *>(frame) - s.ring_buf);

  if (s.use_bf) {
    // dGPU: send first, then repost.  Reposting before send adds ~400ns by
    // serializing a PCIe write ahead of BlueFlame.
    if (!s.use_inline) {
      send_bf<GPU_ROCE_MAX_FRAME_SIZE_0B>(s.qp, &s.wqe_sh, s.sq_wqe_idx,
                                          buffer_addr);
    } else {
      send_bf<GPU_ROCE_MAX_FRAME_SIZE_44B>(
          s.qp, &s.wqe_sh, s.sq_wqe_idx,
          reinterpret_cast<std::uint64_t>(frame));
    }
    recycle_receive(s);
  } else {
    // iGPU: repost first, then send.  The CPU proxy may batch doorbell
    // writes, so ensure the NIC has a fresh receive WQE before any delay.
    recycle_receive(s);
    if (!s.use_inline) {
      send<GPU_ROCE_MAX_FRAME_SIZE_0B>(s.qp, s.sq_wqe_idx - 1, buffer_addr);
    } else {
      send<GPU_ROCE_MAX_FRAME_SIZE_44B>(s.qp, s.sq_wqe_idx - 1,
                                        reinterpret_cast<std::uint64_t>(frame));
    }
  }
}

//==============================================================================
// Host side
//==============================================================================

extern "C" int
gpu_roce_unified_ctx_to_device(const gpu_roce_doca_transport_ctx *host_ctx,
                               void **out_device_ctx) {
  if (host_ctx == nullptr || out_device_ctx == nullptr)
    return cudaErrorInvalidValue;

  void *device_ctx = nullptr;
  cudaError_t err = cudaMalloc(&device_ctx, sizeof(*host_ctx));
  if (err != cudaSuccess)
    return err;

  err = cudaMemcpy(device_ctx, host_ctx, sizeof(*host_ctx),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(device_ctx);
    return err;
  }

  *out_device_ctx = device_ctx;
  return cudaSuccess;
}

extern "C" void gpu_roce_unified_ctx_free(void *device_ctx) {
  if (device_ctx != nullptr)
    cudaFree(device_ctx);
}

// This transport supplies no cudaq_unified_launch_fn_t.  It implements the
// three device hooks above, so cudaq_dispatcher_start() runs the built-in
// cudaq_launch_unified_dispatch_device over them; the consumer passes NULL
// for the launch function and wires config.num_slots / config.slot_size from
// the ring geometry exactly as it would for the 3-kernel path.
