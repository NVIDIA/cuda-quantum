/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/operators.h"
#include "cudaq/utils/tensor.h"
#include <cudensitymat.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace cudaq {
std::vector<std::complex<double>> flatten_matrix(const matrix_2 &matrix);

void scale_state(cudensitymatHandle_t handle, cudensitymatState_t state,
                 double scale_factor, cudaStream_t stream);

cudensitymatOperator_t
compute_lindblad_operator(cudensitymatHandle_t handle,
                          const std::vector<matrix_2> &c_ops,
                          const std::vector<int64_t> &mode_extents);

template <typename HandlerTy>
cudensitymatOperator_t convert_to_cudensitymat_operator(
    cudensitymatHandle_t handle,
    const std::map<std::string, std::complex<double>> &parameters,
    const operator_sum<HandlerTy> &op,
    const std::vector<int64_t> &mode_extents);

cudensitymatOperator_t construct_liovillian(
    cudensitymatHandle_t handle, const cudensitymatOperator_t &hamiltonian,
    const std::vector<cudensitymatOperator_t> &collapse_operators,
    double gamma);

// Function for creating an array copy in GPU memory
void *create_array_gpu(const std::vector<std::complex<double>> &cpu_array);

// Function to detsroy a previously created array copy in GPU memory
void destroy_array_gpu(void *gpu_array);

extern template cudensitymatOperator_t
convert_to_cudensitymat_operator<cudaq::matrix_operator>(
    cudensitymatHandle_t, const std::map<std::string, std::complex<double>> &,
    const operator_sum<cudaq::matrix_operator> &, const std::vector<int64_t> &);
} // namespace cudaq