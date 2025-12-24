/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/channels/doca/doca_queue_pair.h"
#include "cudaq/nvqlink/utils/instrumentation/domains.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"

#include <cuda_runtime.h>
#include <doca_gpunetio_verbs_def.h>
#include <infiniband/mlx5dv.h>

#include <cstdlib>
#include <unistd.h>

namespace cudaq::nvqlink {

static std::size_t get_page_size() {
  long ret = sysconf(_SC_PAGESIZE);
  if (ret == -1)
    return 4096; // 4KB, default Linux page size
  return static_cast<std::size_t>(ret);
}

static std::uint32_t align_up_uint32(std::uint32_t value,
                                     std::uint32_t alignment) {
  std::uint64_t remainder = (value % alignment);
  if (remainder == 0)
    return value;
  return static_cast<std::uint32_t>(value + (alignment - remainder));
}

#define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))

constexpr std::uint32_t VERBS_TEST_DBR_SIZE = 8;

static std::uint32_t calc_qp_external_umem_size(std::uint32_t rq_nwqes,
                                                std::uint32_t sq_nwqes) {
  std::uint32_t rq_ring_size = 0;
  std::uint32_t sq_ring_size = 0;

  if (rq_nwqes != 0)
    rq_ring_size =
        static_cast<std::uint32_t>(rq_nwqes * sizeof(mlx5_wqe_data_seg));
  if (sq_nwqes != 0)
    sq_ring_size =
        static_cast<std::uint32_t>(sq_nwqes * sizeof(doca_gpu_dev_verbs_wqe));

  return align_up_uint32(rq_ring_size + sq_ring_size, get_page_size());
}

DOCAQueuePair::DOCAQueuePair(std::uint32_t wqe_num, doca_gpu *gdev,
                             doca_dev *ndev, doca_uar *uar,
                             doca_verbs_context *vctx, doca_verbs_pd *vpd,
                             doca_verbs_cq *cq_rq, doca_verbs_cq *cq_sq)
    : wqe_num_(wqe_num), gdev_(gdev), ndev_(ndev), uar_(uar), vctx_(vctx),
      vpd_(vpd), cq_rq_(cq_rq), cq_sq_(cq_sq) {}

DOCAQueuePair::~DOCAQueuePair() {
  // Correct DOCA resource cleanup order:
  // 1. Destroy QP (releases references to external UMEMs)
  // 2. Destroy UMEMs (memory registrations)
  // 3. Free GPU memory (underlying buffers)
  //
  // Note: gpu_qp_ is managed by DOCA internally, we only destroy the verbs QP
  if (qp_)
    doca_verbs_qp_destroy(qp_);
  if (umem_)
    doca_umem_destroy(umem_);
  if (umem_dbr_)
    doca_umem_destroy(umem_dbr_);
  if (umem_dev_ptr_)
    doca_gpu_mem_free(gdev_, umem_dev_ptr_);
  if (umem_dbr_dev_ptr_)
    doca_gpu_mem_free(gdev_, umem_dbr_dev_ptr_);
}

doca_error_t DOCAQueuePair::create() {
  std::size_t dbr_umem_align_sz;
  doca_verbs_qp_init_attr *qp_init_attr = nullptr;
  doca_error_t result = DOCA_SUCCESS;
  std::uint32_t external_umem_size;

  result = doca_verbs_qp_init_attr_create(&qp_init_attr);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to create QP init attributes: {}",
                      doca_error_get_descr(result));
    return result;
  }

  // Set QP attributes
  result = doca_verbs_qp_init_attr_set_external_uar(qp_init_attr, uar_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set external UAR for QP: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_init_attr_set_pd(qp_init_attr, vpd_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set PD for QP: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_init_attr_set_external_datapath_en(qp_init_attr, 1);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                      "Failed to set external datapath for QP: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // Set send/receive CQs
  result = doca_verbs_qp_init_attr_set_receive_cq(qp_init_attr, cq_rq_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set receive CQ: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_init_attr_set_send_cq(qp_init_attr, cq_sq_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set send CQ: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // Set WR capacities
  result = doca_verbs_qp_init_attr_set_sq_wr(qp_init_attr, wqe_num_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set SQ WR: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_init_attr_set_rq_wr(qp_init_attr, wqe_num_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set RQ WR: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // Send sge and recv sge
  result = doca_verbs_qp_init_attr_set_send_max_sges(qp_init_attr, 1);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set send max sges: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_init_attr_set_receive_max_sges(qp_init_attr, 1);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set receive max sges: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // Set QP type to UC (Unreliable Connected)
  result =
      doca_verbs_qp_init_attr_set_qp_type(qp_init_attr, DOCA_VERBS_QP_TYPE_UC);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set QP type: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // Set sq_sig_all to 0
  result = doca_verbs_qp_init_attr_set_sq_sig_all(qp_init_attr, 0);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set sq sig all: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // Allocate external UMEM for WQE ring
  external_umem_size = calc_qp_external_umem_size(wqe_num_, wqe_num_);

  result = doca_gpu_mem_alloc(gdev_, external_umem_size, get_page_size(),
                              DOCA_GPU_MEM_TYPE_GPU, &umem_dev_ptr_, nullptr);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                      "Failed to alloc GPU memory for external UMEM QP");
    goto exit;
  }

  result = doca_umem_gpu_create(
      gdev_, ndev_, umem_dev_ptr_, external_umem_size,
      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
          DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
      &umem_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to create GPU UMEM: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_init_attr_set_external_umem(qp_init_attr, umem_, 0);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                      "Failed to set DOCA Verbs QP external UMEM");
    goto exit;
  }

  // Allocate doorbell record UMEM
  dbr_umem_align_sz = ROUND_UP(VERBS_TEST_DBR_SIZE, get_page_size());
  result =
      doca_gpu_mem_alloc(gdev_, dbr_umem_align_sz, get_page_size(),
                         DOCA_GPU_MEM_TYPE_GPU, &umem_dbr_dev_ptr_, nullptr);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                      "Failed to alloc GPU memory for DBR UMEM");
    goto exit;
  }

  result = doca_umem_gpu_create(
      gdev_, ndev_, umem_dbr_dev_ptr_, dbr_umem_align_sz,
      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
          DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
      &umem_dbr_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to create GPU UMEM for DBR: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result =
      doca_verbs_qp_init_attr_set_external_dbr_umem(qp_init_attr, umem_dbr_, 0);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                      "Failed to set DOCA Verbs QP external DBR UMEM");
    goto exit;
  }

  // Create the QP
  result = doca_verbs_qp_create(vctx_, qp_init_attr, &qp_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to create QP: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // Export to GPU
  result = doca_gpu_verbs_export_qp(
      gdev_, ndev_, qp_, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
      umem_dbr_dev_ptr_, cq_sq_, cq_rq_, &gpu_qp_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to export GPU Verbs QP");
    goto exit;
  }

  result = doca_gpu_verbs_get_qp_dev(gpu_qp_, &gpu_dev_qp_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to get QP device handle");
    goto exit;
  }

exit:
  doca_verbs_qp_init_attr_destroy(qp_init_attr);
  return result;
}

doca_error_t DOCAQueuePair::connect(const doca_verbs_gid &remote_gid,
                                    std::uint32_t gid_index,
                                    std::uint32_t dest_qp_num) {
  doca_error_t result;
  doca_verbs_qp_attr *verbs_qp_attr = nullptr;
  doca_verbs_ah_attr *ah_attr = nullptr;
  std::uint32_t rst2init_mask = DOCA_UC_QP_RST2INIT_REQ_ATTR_MASK;
  std::uint32_t init2rtr_mask = DOCA_UC_QP_INIT2RTR_REQ_ATTR_MASK;
  std::uint32_t rtr2rts_mask = DOCA_UC_QP_RTR2RTS_REQ_ATTR_MASK;

  // Create AH and QP attributes
  result = doca_verbs_ah_attr_create(vctx_, &ah_attr);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to create AH attributes: {}",
                      doca_error_get_descr(result));
    return result;
  }

  result = doca_verbs_qp_attr_create(&verbs_qp_attr);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to create QP attributes: {}",
                      doca_error_get_descr(result));
    doca_verbs_ah_attr_destroy(ah_attr);
    return result;
  }

  // Set AH attributes
  result = doca_verbs_ah_attr_set_gid(ah_attr, remote_gid);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set remote GID: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_ah_attr_set_dlid(ah_attr, 0); // IB only parameter
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set dlid: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_ah_attr_set_addr_type(ah_attr, DOCA_VERBS_ADDR_TYPE_IPv4);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set address type: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_ah_attr_set_sgid_index(ah_attr, gid_index);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set SGID index: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_ah_attr_set_hop_limit(ah_attr, 0xFF);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set hop limit: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // Set QP attributes
  result =
      doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, DOCA_MTU_SIZE_4K_BYTES);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set path MTU: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, 0);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set RQ PSN: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, 0);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set SQ PSN: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, ah_attr);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set AH attr: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_port_num(verbs_qp_attr, 1);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set port num: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, 14);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set ACK timeout: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, 7);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set retry count: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, 6);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set RNR retry: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, 12);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set min RNR timer: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, 1);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set allow remote write: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, 1);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set allow remote read: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, dest_qp_num);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set dest QP num: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // QP State Transitions

  // RST → INIT
  result = doca_verbs_qp_attr_set_next_state(verbs_qp_attr,
                                             DOCA_VERBS_QP_STATE_INIT);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set next state INIT: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_modify(qp_, verbs_qp_attr, rst2init_mask);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to modify QP RST→INIT: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // INIT → RTR
  result =
      doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTR);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set next state RTR: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_modify(qp_, verbs_qp_attr, init2rtr_mask);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to modify QP INIT→RTR: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // RTR → RTS
  result =
      doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTS);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set next state RTS: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_qp_modify(qp_, verbs_qp_attr, rtr2rts_mask);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to modify QP RTR→RTS: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

exit:
  doca_verbs_qp_attr_destroy(verbs_qp_attr);
  doca_verbs_ah_attr_destroy(ah_attr);
  return result;
}

} // namespace cudaq::nvqlink
