/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatExpectation.h"
#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatUtils.h"
#include "common/Logger.h"
namespace cudaq {
CuDensityMatExpectation::CuDensityMatExpectation(cudensitymatHandle_t handle,
                                                 cudensitymatOperator_t op)
    : m_handle(handle), m_hamOp(op) {
  HANDLE_CUDM_ERROR(
      cudensitymatCreateExpectation(m_handle, m_hamOp, &m_expectation));
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(m_handle, &m_workspace));
}

CuDensityMatExpectation::~CuDensityMatExpectation() {
  if (m_workspace)
    cudensitymatDestroyWorkspace(m_workspace);
  if (m_expectation)
    cudensitymatDestroyExpectation(m_expectation);
}

void CuDensityMatExpectation::prepare(cudensitymatState_t state) {
  HANDLE_CUDM_ERROR(cudensitymatExpectationPrepare(
      m_handle, m_expectation, state, CUDENSITYMAT_COMPUTE_64F,
      dynamics::Context::getRecommendedWorkSpaceLimit(), m_workspace, 0x0));
}
std::vector<std::complex<double>>
CuDensityMatExpectation::compute(cudensitymatState_t state, double time,
                                 int64_t batchSize) {
  std::size_t requiredBufferSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(
      m_handle, m_workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
      CUDENSITYMAT_WORKSPACE_SCRATCH, &requiredBufferSize));

  void *workspaceBuffer = nullptr;
  if (requiredBufferSize > 0) {
    workspaceBuffer = dynamics::Context::getCurrentContext()->getScratchSpace(
        requiredBufferSize);

    // Attach workspace buffer
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(
        m_handle, m_workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
        CUDENSITYMAT_WORKSPACE_SCRATCH, workspaceBuffer, requiredBufferSize));
  }

  auto *expectationValue_d = cudaq::dynamics::createArrayGpu(
      std::vector<std::complex<double>>(batchSize, {0.0, 0.0}));
  {
    cudaq::dynamics::PerfMetricScopeTimer metricTimer(
        "cudensitymatExpectationCompute");
    HANDLE_CUDM_ERROR(cudensitymatExpectationCompute(
        m_handle, m_expectation, time, batchSize, 0, nullptr, state,
        expectationValue_d, m_workspace, 0x0));
  }
  std::vector<std::complex<double>> result(batchSize);
  HANDLE_CUDA_ERROR(cudaMemcpy(result.data(), expectationValue_d,
                               batchSize * sizeof(std::complex<double>),
                               cudaMemcpyDefault));
  cudaq::dynamics::destroyArrayGpu(expectationValue_d);
  return result;
}
} // namespace cudaq
