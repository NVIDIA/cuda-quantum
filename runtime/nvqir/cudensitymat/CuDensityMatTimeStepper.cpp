/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatTimeStepper.h"
#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
namespace cudaq {
CuDensityMatTimeStepper::CuDensityMatTimeStepper(
    cudensitymatHandle_t handle, cudensitymatOperator_t liouvillian)
    : m_handle(handle), m_liouvillian(liouvillian){};

state CuDensityMatTimeStepper::compute(
    const state &inputState, double t, double step_size,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  if (step_size == 0.0) {
    throw std::runtime_error("Step size cannot be zero.");
  }

  auto *simState =
      cudaq::state_helper::getSimulationState(const_cast<state *>(&inputState));
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  if (!castSimState)
    throw std::runtime_error("Invalid state.");
  CuDensityMatState &state = *castSimState;
  // Prepare workspace
  cudensitymatWorkspaceDescriptor_t workspace;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(m_handle, &workspace));

  // Create a new state for the next step
  auto next_state = CuDensityMatState::zero_like(state);

  if (!next_state.is_initialized()) {
    throw std::runtime_error("Next state failed to initialize.");
  }

  if (state.get_hilbert_space_dims() != next_state.get_hilbert_space_dims()) {
    throw std::runtime_error("As the dimensions of both the old and the new "
                             "state do no match, the "
                             "operator cannot act on the states.");
  }

  // Prepare the operator for action
  HANDLE_CUDM_ERROR(cudensitymatOperatorPrepareAction(
      m_handle, m_liouvillian, state.get_impl(), next_state.get_impl(),
      CUDENSITYMAT_COMPUTE_64F,
      dynamics::Context::getRecommendedWorkSpaceLimit(), workspace, 0x0));

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
  std::vector<double> paramValues;
  for (const auto &[k, v] : sortedParameters) {
    paramValues.emplace_back(v.real());
    paramValues.emplace_back(v.imag());
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDM_ERROR(cudensitymatOperatorComputeAction(
      m_handle, m_liouvillian, t, paramValues.size(), paramValues.data(),
      state.get_impl(), next_state.get_impl(), workspace, 0x0));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Cleanup
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspace));

  return cudaq::state(
      std::make_unique<CuDensityMatState>(std::move(next_state)).release());
}

} // namespace cudaq