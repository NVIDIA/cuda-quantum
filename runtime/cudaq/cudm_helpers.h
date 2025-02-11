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

cudensitymatWrappedScalarCallback_t
_wrap_callback(const scalar_operator &scalar_op);

cudensitymatWrappedTensorCallback_t
_wrap_tensor_callback(const matrix_operator &op);

void append_scalar_to_term(cudensitymatHandle_t handle,
                           cudensitymatOperatorTerm_t term,
                           const scalar_operator &scalar_op);

std::map<int, int> convert_dimensions(const std::vector<int64_t> &mode_extents);

std::vector<int64_t>
get_subspace_extents(const std::vector<int64_t> &mode_extents,
                     const std::vector<int> &degrees);

cudensitymatElementaryOperator_t create_elementary_operator(
    cudensitymatHandle_t handle, const std::vector<int64_t> &subspace_extents,
    const std::vector<std::complex<double>> &flat_matrix);

void append_elementary_operator_to_term(
    cudensitymatHandle_t handle, cudensitymatOperatorTerm_t term,
    const cudensitymatElementaryOperator_t &elem_op,
    const std::vector<int> &degrees, const std::vector<int64_t> &mode_extents,
    const cudensitymatWrappedTensorCallback_t &wrapped_tensor_callback);

// Function for creating an array copy in GPU memory
void *create_array_gpu(const std::vector<std::complex<double>> &cpu_array);

// Function to detsroy a previously created array copy in GPU memory
void destroy_array_gpu(void *gpu_array);

extern template cudensitymatOperator_t
convert_to_cudensitymat_operator<cudaq::matrix_operator>(
    cudensitymatHandle_t, const std::map<std::string, std::complex<double>> &,
    const operator_sum<cudaq::matrix_operator> &, const std::vector<int64_t> &);
} // namespace cudaq