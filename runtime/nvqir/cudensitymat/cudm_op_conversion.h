/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudm_helpers.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <cudensitymat.h>
#include <map>

namespace cudaq {
class cudm_op_conversion {
public:
  cudm_op_conversion(const cudensitymatHandle_t handle,
                     const std::map<int, int> &dimensions,
                     const std::shared_ptr<Schedule> schedule = nullptr);

  // Tensor product of two operator terms
  std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
               std::complex<double>>
  tensor(const std::variant<cudensitymatOperatorTerm_t,
                            cudensitymatWrappedScalarCallback_t,
                            std::complex<double>> &op1,
         const std::variant<cudensitymatOperatorTerm_t,
                            cudensitymatWrappedScalarCallback_t,
                            std::complex<double>> &op2);

  // Multiplication of two operator terms
  std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
               std::complex<double>>
  mul(const std::variant<cudensitymatOperatorTerm_t,
                         cudensitymatWrappedScalarCallback_t,
                         std::complex<double>> &op1,
      const std::variant<cudensitymatOperatorTerm_t,
                         cudensitymatWrappedScalarCallback_t,
                         std::complex<double>> &op2);

  // Addition of two operator terms
  std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
               std::complex<double>>
  add(const std::variant<cudensitymatOperatorTerm_t,
                         cudensitymatWrappedScalarCallback_t,
                         std::complex<double>> &op1,
      const std::variant<cudensitymatOperatorTerm_t,
                         cudensitymatWrappedScalarCallback_t,
                         std::complex<double>> &op2);

  // Evaluate an operator and convert it to cudensitymatOperatorTerm_t
  std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
               std::complex<double>>
  evaluate(const std::variant<scalar_operator, matrix_operator,
                              product_operator<matrix_operator>> &op);

private:
  cudensitymatHandle_t handle_;
  std::map<int, int> dimensions_;
  std::shared_ptr<Schedule> schedule_;

  cudensitymatOperatorTerm_t
  _callback_mult_op(const cudensitymatWrappedScalarCallback_t &scalar,
                    const cudensitymatOperatorTerm_t &op);
  cudensitymatOperatorTerm_t
  _scalar_to_op(const cudensitymatWrappedScalarCallback_t &scalar);
  // cudensitymatWrappedScalarCallback_t _wrap_callback(const scalar_operator
  // &op); cudensitymatWrappedTensorCallback_t _wrap_callback_tensor(const
  // matrix_operator &op);

  std::vector<std::complex<double>> get_identity_matrix();

  std::vector<int64_t> get_space_mode_extents();
};
} // namespace cudaq