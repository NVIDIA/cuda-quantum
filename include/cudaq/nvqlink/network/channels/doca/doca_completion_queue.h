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

#include <cstdint>

namespace cudaq::nvqlink {

/// @brief DOCA Completion Queue wrapper with GPU-external datapath
///
/// Manages:
/// - GPU memory for CQ entries (via doca_gpu_mem_alloc)
/// - DOCA UMEM registration
/// - CQ creation with external datapath enabled
///
/// RAII: Cleans up resources in destructor.
///
class DOCACompletionQueue {
public:
  DOCACompletionQueue(std::uint32_t cqe_num, doca_gpu *gdev, doca_dev *ndev,
                      doca_uar *uar, doca_verbs_context *vctx);

  ~DOCACompletionQueue();

  // Disable copy/move
  DOCACompletionQueue(const DOCACompletionQueue &) = delete;
  DOCACompletionQueue &operator=(const DOCACompletionQueue &) = delete;

  /// @brief Create and initialize the CQ
  /// @return DOCA_SUCCESS or error code
  doca_error_t create();

  /// @brief Get the DOCA Verbs CQ handle
  doca_verbs_cq *get() const { return cq_; }

private:
  doca_gpu *gdev_;
  doca_dev *ndev_;
  doca_uar *uar_;
  doca_verbs_context *vctx_;
  std::uint32_t cqe_num_;

  doca_verbs_cq *cq_ = nullptr;
  void *umem_dev_ptr_ = nullptr;
  doca_umem *umem_ = nullptr;
};

} // namespace cudaq::nvqlink
