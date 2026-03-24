/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatTimeStepper.h"
#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatUtils.h"
#include <map>

namespace cudaq {
CuDensityMatTimeStepper::CuDensityMatTimeStepper(
    cudensitymatHandle_t handle, cudensitymatOperator_t liouvillian)
    : m_handle(handle), m_liouvillian(liouvillian){};

state CuDensityMatTimeStepper::compute(
    const state &inputState, double t,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  auto *simState =
      cudaq::state_helper::getSimulationState(const_cast<state *>(&inputState));
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  if (!castSimState)
    throw std::runtime_error("Invalid state.");
  CuDensityMatState &state = *castSimState;

  // Create a new state for the next step
  auto next_state = CuDensityMatState::zero_like(state);
  assert(next_state.getBatchSize() == state.getBatchSize());
  computeImpl(state.get_impl(), next_state.get_impl(), t, parameters,
              state.getBatchSize());
  return cudaq::state(
      std::make_unique<CuDensityMatState>(std::move(next_state)).release());
}

void CuDensityMatTimeStepper::computeImpl(
    cudensitymatState_t inState, cudensitymatState_t outState, double t,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    int64_t batchSize) {
  // Prepare workspace
  cudensitymatWorkspaceDescriptor_t workspace;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(m_handle, &workspace));

  {
    cudaq::dynamics::PerfMetricScopeTimer metricTimer(
        "cudensitymatOperatorPrepareAction");
    // Prepare the operator for action
    HANDLE_CUDM_ERROR(cudensitymatOperatorPrepareAction(
        m_handle, m_liouvillian, inState, outState, CUDENSITYMAT_COMPUTE_64F,
        dynamics::Context::getRecommendedWorkSpaceLimit(), workspace, 0x0));
  }

  // Query required workspace buffer size
  std::size_t requiredBufferSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(
      m_handle, workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
      CUDENSITYMAT_WORKSPACE_SCRATCH, &requiredBufferSize));

  void *workspaceBuffer = nullptr;
  if (requiredBufferSize > 0) {
    workspaceBuffer = dynamics::Context::getCurrentContext()->getScratchSpace(
        requiredBufferSize);

    // Attach workspace buffer
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(
        m_handle, workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
        CUDENSITYMAT_WORKSPACE_SCRATCH, workspaceBuffer, requiredBufferSize));
  }

  // Apply the operator action
  std::map<std::string, std::complex<double>> sortedParameters(
      parameters.begin(), parameters.end());
  const auto numComplexParams = sortedParameters.size();
  std::vector<std::complex<double>> paramValues;
  paramValues.reserve(numComplexParams * batchSize);
  // Note: for batch, params is F-order 2d-array of user-defined real parameter
  // values: params[numParams, batchSize].
  for (int i = 0; i < batchSize; ++i) {
    for (const auto &[k, v] : sortedParameters) {
      paramValues.emplace_back(v);
    }
  }
  double *param_d =
      static_cast<double *>(cudaq::dynamics::createArrayGpu(paramValues));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  {
    cudaq::dynamics::PerfMetricScopeTimer metricTimer(
        "cudensitymatOperatorComputeAction");
    HANDLE_CUDM_ERROR(cudensitymatOperatorComputeAction(
        m_handle, m_liouvillian, t, batchSize, numComplexParams * 2, param_d,
        inState, outState, workspace, 0x0));
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  }

  // Cleanup
  cudaq::dynamics::destroyArrayGpu(param_d);
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspace));
}

} // namespace cudaq
