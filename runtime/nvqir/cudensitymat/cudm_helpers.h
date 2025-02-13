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
class cudm_helper {
public:
  explicit cudm_helper(cudensitymatHandle_t handle);
  ~cudm_helper();

  // Matrix flattening
  static std::vector<std::complex<double>>
  flatten_matrix(const matrix_2 &matrix);

  // State Operations
  void scale_state(cudensitymatState_t state, double scale_factor,
                   cudaStream_t stream);

  // Compute Lindblad Operator
  cudensitymatOperator_t
  compute_lindblad_operator(const std::vector<matrix_2> &c_ops,
                            const std::vector<int64_t> &mode_extents);

  // Convert operator sum to cudensitymat operator
  template <typename HandlerTy>
  cudensitymatOperator_t convert_to_cudensitymat_operator(
      const std::map<std::string, std::complex<double>> &parameters,
      const operator_sum<HandlerTy> &op,
      const std::vector<int64_t> &mode_extents);

  // Construct Liouvillian
  cudensitymatOperator_t construct_liouvillian(
      const operator_sum<cudaq::matrix_operator> &op,
      const std::vector<operator_sum<cudaq::matrix_operator> *>
          &collapse_operators,
      const std::vector<int64_t> &mode_extents,
      const std::map<std::string, std::complex<double>> &parameters,
      bool is_master_equation);

  // Construct Liouvillian
  cudensitymatOperator_t construct_liouvillian(
      const cudensitymatOperator_t &hamiltonian,
      const std::vector<cudensitymatOperator_t> &collapse_operators,
      double gamma);

  // Helper Functions
  std::map<int, int>
  convert_dimensions(const std::vector<int64_t> &mode_extents);
  std::vector<int64_t>
  get_subspace_extents(const std::vector<int64_t> &mode_extents,
                       const std::vector<int> &degrees);

  // Callback Wrappers
  static cudensitymatWrappedScalarCallback_t
  _wrap_callback(const scalar_operator &scalar_op);
  static cudensitymatWrappedTensorCallback_t
  _wrap_tensor_callback(const matrix_operator &op);

  // Elementary Operator Functions
  void append_scalar_to_term(cudensitymatOperatorTerm_t term,
                             const scalar_operator &scalar_op);
  cudensitymatElementaryOperator_t create_elementary_operator(
      const cudaq::matrix_operator &elem_op,
      const std::map<std::string, std::complex<double>> &parameters,
      const std::vector<int64_t> &mode_extents);
  void append_elementary_operator_to_term(
      cudensitymatOperatorTerm_t term,
      const std::vector<cudensitymatElementaryOperator_t> &elem_ops,
      const std::vector<std::vector<int>> &degrees, bool is_dagger);

  // GPU memory management
  static void *
  create_array_gpu(const std::vector<std::complex<double>> &cpu_array);
  static void destroy_array_gpu(void *gpu_array);

private:
  cudensitymatHandle_t handle;
};

extern template cudensitymatOperator_t
cudm_helper::convert_to_cudensitymat_operator<cudaq::matrix_operator>(
    const std::map<std::string, std::complex<double>> &,
    const operator_sum<cudaq::matrix_operator> &, const std::vector<int64_t> &);
} // namespace cudaq