/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudm_expectation.h"
#include "common/Logger.h"
#include "cudm_error_handling.h"
#include "cudm_helpers.h"
#include <iostream>

namespace cudaq {
cudm_expectation::cudm_expectation(cudensitymatHandle_t handle,
                                   cudensitymatOperator_t op)
    : m_handle(handle), m_hamOp(op) {
  HANDLE_CUDM_ERROR(
      cudensitymatCreateExpectation(m_handle, m_hamOp, &m_expectation));
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(m_handle, &m_workspace));
}

cudm_expectation::~cudm_expectation() {
  if (m_workspace)
    cudensitymatDestroyWorkspace(m_workspace);
  if (m_expectation)
    cudensitymatDestroyExpectation(m_expectation);
}

void cudm_expectation::prepare(cudensitymatState_t state) {
  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.80);

  HANDLE_CUDM_ERROR(cudensitymatExpectationPrepare(
      m_handle, m_expectation, state, CUDENSITYMAT_COMPUTE_64F, freeMem,
      m_workspace, 0x0));
}
std::complex<double> cudm_expectation::compute(cudensitymatState_t state,
                                               double time) {
  // TODO: create a global scratch buffer
  std::size_t requiredBufferSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(
      m_handle, m_workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
      CUDENSITYMAT_WORKSPACE_SCRATCH, &requiredBufferSize));

  void *workspaceBuffer = nullptr;
  if (requiredBufferSize > 0) {
    cudaq::info("Required buffer size for expectation compute: {}",
                requiredBufferSize);
    // Allocate GPU storage for workspace buffer
    const std::size_t bufferVolume =
        requiredBufferSize / sizeof(std::complex<double>);
    workspaceBuffer = cudm_helper::create_array_gpu(
        std::vector<std::complex<double>>(bufferVolume, {0.0, 0.0}));

    // Attach workspace buffer
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(
        m_handle, m_workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
        CUDENSITYMAT_WORKSPACE_SCRATCH, workspaceBuffer, requiredBufferSize));
  }

  auto *expectationValue_d = cudm_helper::create_array_gpu(
      std::vector<std::complex<double>>(1, {0.0, 0.0}));
  HANDLE_CUDM_ERROR(cudensitymatExpectationCompute(
      m_handle, m_expectation, time, 0, nullptr, state, expectationValue_d,
      m_workspace, 0x0));
  std::complex<double> result;
  HANDLE_CUDA_ERROR(cudaMemcpy(&result, expectationValue_d,
                               sizeof(std::complex<double>),
                               cudaMemcpyDefault));
  cudm_helper::destroy_array_gpu(expectationValue_d);
  if (workspaceBuffer) {
    cudm_helper::destroy_array_gpu(workspaceBuffer);
  }
  return result;
}
} // namespace cudaq
