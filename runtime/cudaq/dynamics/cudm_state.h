/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cudensitymat.h>
#include <vector>

namespace cudaq {
class cudm_mat_state {
public:
    cudm_mat_state(cudensitymatHandle_t handle, cudensitymatStatePurity_t purity, int num_modes, const std::vector<int64_t> &mode_extents);
    ~cudm_mat_state();

    void scale(double factor, cudaStream_t stream);
    cudensitymatState_t get() const;

private:
    cudensitymatState_t state;
    cudensitymatHandle_t handle;
};
}