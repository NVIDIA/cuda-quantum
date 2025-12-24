/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/channels/doca/doca_completion_queue.h"
#include "cudaq/nvqlink/utils/instrumentation/domains.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"

#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>

#include <cstdlib>
#include <cstring>
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

static std::uint32_t calc_cq_external_umem_size(std::uint32_t queue_size) {
  constexpr std::uint32_t VERBS_TEST_DBR_SIZE = 8;
  std::uint32_t cqe_buf_size = 0;

  if (queue_size != 0)
    cqe_buf_size = static_cast<std::uint32_t>(queue_size * sizeof(mlx5_cqe64));

  return align_up_uint32(cqe_buf_size + VERBS_TEST_DBR_SIZE, get_page_size());
}

static void mlx5_init_cqes(mlx5_cqe64 *cqes, std::uint32_t nb_cqes) {
  for (std::uint32_t cqe_idx = 0; cqe_idx < nb_cqes; cqe_idx++)
    cqes[cqe_idx].op_own =
        (MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) |
        MLX5_CQE_OWNER_MASK;
}

DOCACompletionQueue::DOCACompletionQueue(std::uint32_t cqe_num, doca_gpu *gdev,
                                         doca_dev *ndev, doca_uar *uar,
                                         doca_verbs_context *vctx)
    : cqe_num_(cqe_num), gdev_(gdev), ndev_(ndev), uar_(uar), vctx_(vctx) {}

DOCACompletionQueue::~DOCACompletionQueue() {
  // Correct DOCA resource cleanup order:
  // 1. Destroy CQ (releases reference to external UMEM)
  // 2. Destroy UMEM (memory registration)
  // 3. Free GPU memory (underlying buffer)
  if (cq_ != nullptr)
    doca_verbs_cq_destroy(cq_);
  if (umem_)
    doca_umem_destroy(umem_);
  if (umem_dev_ptr_)
    doca_gpu_mem_free(gdev_, umem_dev_ptr_);
}

doca_error_t DOCACompletionQueue::create() {
  doca_verbs_cq_attr *cq_attr = nullptr;
  std::uint32_t external_umem_size;
  mlx5_cqe64 *cq_ring_haddr = nullptr;
  doca_error_t result = DOCA_SUCCESS;
  cudaError_t result_cuda;

  result = doca_verbs_cq_attr_create(&cq_attr);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to create CQ attributes: {}",
                      doca_error_get_descr(result));
    return result;
  }

  result = doca_verbs_cq_attr_set_external_uar(cq_attr, uar_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set external UAR: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_cq_attr_set_cq_size(cq_attr, cqe_num_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set CQ size: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_cq_attr_set_external_datapath_en(cq_attr, 1);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                      "Failed to set external datapath enable: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result =
      doca_verbs_cq_attr_set_entry_size(cq_attr, DOCA_VERBS_CQ_ENTRY_SIZE_64);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set CQ entry size: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result = doca_verbs_cq_attr_set_cq_overrun(cq_attr, 1);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set CQ overrun: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  // Allocate and initialize CQ ring buffer
  external_umem_size = calc_cq_external_umem_size(cqe_num_);

  cq_ring_haddr = static_cast<mlx5_cqe64 *>(
      std::calloc(external_umem_size, sizeof(std::uint8_t)));
  if (cq_ring_haddr == nullptr) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to allocate CQ host ring buffer");
    goto exit;
  }

  mlx5_init_cqes(cq_ring_haddr, cqe_num_);

  result = doca_gpu_mem_alloc(gdev_, external_umem_size, get_page_size(),
                              DOCA_GPU_MEM_TYPE_GPU, &umem_dev_ptr_, nullptr);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                      "Failed to allocate GPU memory for CQ: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

  result_cuda = cudaMemcpy(umem_dev_ptr_, cq_ring_haddr, external_umem_size,
                           cudaMemcpyDefault);
  if (result_cuda != cudaSuccess) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to copy CQ ring buffer to GPU");
    result = DOCA_ERROR_DRIVER;
    goto exit;
  }

  // Create UMEM
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

  result = doca_verbs_cq_attr_set_external_umem(cq_attr, umem_, 0);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to set CQ external UMEM");
    goto exit;
  }

  // Create the CQ
  result = doca_verbs_cq_create(vctx_, cq_attr, &cq_);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to create CQ: {}",
                      doca_error_get_descr(result));
    goto exit;
  }

exit:
  doca_verbs_cq_attr_destroy(cq_attr);
  if (cq_ring_haddr)
    std::free(cq_ring_haddr);
  return result;
}

} // namespace cudaq::nvqlink
