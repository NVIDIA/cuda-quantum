/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/cudm_time_stepper.h"
#include "cudaq/cudm_error_handling.h"
#include <iostream>

namespace cudaq {
cudm_time_stepper::cudm_time_stepper(cudensitymatHandle_t handle, cudensitymatOperator_t liouvillian) : handle_(handle), liouvillian_(liouvillian) {}

void cudm_time_stepper::compute(cudm_state &state, double t, double step_size) {
    if (!state.is_initialized()) {
        throw std::runtime_error("State is not initialized.");
    }

    std::cout << "Preparing workspace ..." << std::endl;
    // Prepare workspace
    cudensitymatWorkspaceDescriptor_t workspace;
    HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle_, &workspace));
    if (!workspace) {
        throw std::runtime_error("Failed to create workspace for the operator.");
    }

    std::cout << "Create a new state for the next step ..." << std::endl;
    // Create a new state for the next step
    cudm_state next_state(state.to_density_matrix().get_raw_data());
    next_state.init_state(state.get_hilbert_space_dims());

    if (!next_state.is_initialized()) {
        throw std::runtime_error("Next state failed to initialize.");
    }

    if (!handle_) {
        throw std::runtime_error("cudm_time_stepper handle is not initializes.");
    }

    if (!liouvillian_) {
        throw std::runtime_error("Liouvillian is not initialized.");
    }

    std::cout << "cudensitymatOperatorComputeAction ..." << std::endl;
    HANDLE_CUDM_ERROR(cudensitymatOperatorComputeAction(handle_, liouvillian_, t, 0, nullptr, state.get_impl(), next_state.get_impl(), workspace, 0));

    std::cout << "Update the state ..." << std::endl;
    // Update the state
    state = std::move(next_state);
    
    std::cout << "Clean up workspace ..." << std::endl;
    // Clean up workspace
    HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspace));
}
}