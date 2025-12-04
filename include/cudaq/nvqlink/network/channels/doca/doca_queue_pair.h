/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>

#include <array>
#include <cstdint>

namespace cudaq::nvqlink {

//===------------------------------------------------------------------------===
// QP state transition masks
//===------------------------------------------------------------------------===

constexpr std::uint32_t DOCA_UC_QP_RST2INIT_REQ_ATTR_MASK =
    DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
    DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM;

constexpr std::uint32_t DOCA_UC_QP_INIT2RTR_REQ_ATTR_MASK =
    DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
    DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
    DOCA_VERBS_QP_ATTR_AH_ATTR;

constexpr std::uint32_t DOCA_UC_QP_RTR2RTS_REQ_ATTR_MASK =
    DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN;

/// @brief DOCA Queue Pair wrapper with GPU-external datapath
///
/// Manages:
/// - QP creation with external datapath
/// - GPU memory for WQE ring and doorbell records
/// - QP state transitions (RST → INIT → RTR → RTS)
/// - GPU export for kernel access
///
/// RAII: Cleans up resources in destructor.
///
class DOCAQueuePair {
public:
  DOCAQueuePair(std::uint32_t wqe_num, doca_gpu *gdev, doca_dev *ndev,
                doca_uar *uar, doca_verbs_context *vctx, doca_verbs_pd *vpd,
                doca_verbs_cq *cq_rq, doca_verbs_cq *cq_sq);

  ~DOCAQueuePair();

  // Disable copy/move
  DOCAQueuePair(const DOCAQueuePair &) = delete;
  DOCAQueuePair &operator=(const DOCAQueuePair &) = delete;

  /// @brief Create and initialize the QP
  doca_error_t create();

  /// @brief Transition QP to RTS (Ready to Send) state
  ///
  /// Performs UC QP state transitions:
  /// 1. RST → INIT
  /// 2. INIT → RTR (Ready to Receive)
  /// 3. RTR → RTS (Ready to Send)
  ///
  doca_error_t connect(const doca_verbs_gid &remote_gid,
                       std::uint32_t gid_index, std::uint32_t dest_qp_num);

  /// @brief Get the DOCA Verbs QP handle (host-side)
  doca_verbs_qp *get() const { return qp_; }

  /// @brief Get the GPU Verbs QP handle (host-side)
  doca_gpu_verbs_qp *get_gpu() const { return gpu_qp_; }

  /// @brief Get the GPU device QP handle (device-side accessible)
  doca_gpu_dev_verbs_qp *get_gpu_dev() const { return gpu_dev_qp_; }

private:
  std::uint32_t wqe_num_;
  doca_gpu *gdev_;
  doca_dev *ndev_;
  doca_uar *uar_;
  doca_verbs_context *vctx_;
  doca_verbs_pd *vpd_;
  doca_verbs_cq *cq_rq_;
  doca_verbs_cq *cq_sq_;

  doca_verbs_qp *qp_ = nullptr;
  void *umem_dev_ptr_ = nullptr;
  doca_umem *umem_ = nullptr;
  doca_umem *umem_dbr_ = nullptr;
  void *umem_dbr_dev_ptr_ = nullptr;
  doca_gpu_verbs_qp *gpu_qp_ = nullptr;
  doca_gpu_dev_verbs_qp *gpu_dev_qp_ = nullptr;
};

} // namespace cudaq::nvqlink
