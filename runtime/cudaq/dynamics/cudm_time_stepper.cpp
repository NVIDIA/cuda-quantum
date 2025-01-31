/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/cudm_time_stepper.h"
#include "cudaq/cudm_error_handling.h"
#include "cudaq/cudm_helpers.h"
#include <iostream>

namespace cudaq {
cudm_time_stepper::cudm_time_stepper(cudensitymatHandle_t handle,
                                     cudensitymatOperator_t liouvillian)
    : handle_(handle), liouvillian_(liouvillian) {}

cudm_state cudm_time_stepper::compute(cudm_state &state, double t,
                                      double step_size) {
  if (step_size == 0.0) {
    throw std::runtime_error("Step size cannot be zero.");
  }

  if (!state.is_initialized()) {
    throw std::runtime_error("State is not initialized.");
  }

  if (!handle_) {
    throw std::runtime_error("cudm_time_stepper handle is not initialized.");
  }

  if (!liouvillian_) {
    throw std::runtime_error("Liouvillian is not initialized.");
  }

  // Prepare workspace
  cudensitymatWorkspaceDescriptor_t workspace;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle_, &workspace));

  // Query free gpu memory and allocate workspace buffer
  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  // Take 80% of free memory
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.80);

  // Create a new state for the next step
  std::vector<std::complex<double>> zero_initiailized_data(
      state.get_raw_data().size(), {0.0, 0.0});
  cudm_state next_state(handle_, zero_initiailized_data,
                        state.get_hilbert_space_dims());

  if (!next_state.is_initialized()) {
    throw std::runtime_error("Next state failed to initialize.");
  }

  if (state.get_hilbert_space_dims() != next_state.get_hilbert_space_dims()) {
    throw std::runtime_error(
        "As the dimensions of both the old and the new state do no match, the "
        "operator cannot act on the states.");
  }

  // Prepare the operator for action
  HANDLE_CUDM_ERROR(cudensitymatOperatorPrepareAction(
      handle_, liouvillian_, state.get_impl(), next_state.get_impl(),
      CUDENSITYMAT_COMPUTE_64F, freeMem, workspace, 0x0));

  // Query required workspace buffer size
  std::size_t requiredBufferSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(
      handle_, workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
      CUDENSITYMAT_WORKSPACE_SCRATCH, &requiredBufferSize));

  void *workspaceBuffer = nullptr;
  if (requiredBufferSize > 0) {
    // Allocate GPU storage for workspace buffer
    const std::size_t bufferVolume =
        requiredBufferSize / sizeof(std::complex<double>);
    workspaceBuffer = create_array_gpu(
        std::vector<std::complex<double>>(bufferVolume, {0.0, 0.0}));

    // Attach workspace buffer
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(
        handle_, workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
        CUDENSITYMAT_WORKSPACE_SCRATCH, workspaceBuffer, requiredBufferSize));
  }

  // Apply the operator action
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDM_ERROR(cudensitymatOperatorComputeAction(
      handle_, liouvillian_, t, 1, std::vector<double>({step_size}).data(),
      state.get_impl(), next_state.get_impl(), workspace, 0x0));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Cleanup
  destroy_array_gpu(workspaceBuffer);
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspace));

  return next_state;
}
} // namespace cudaq