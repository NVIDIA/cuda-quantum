/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/cudm_helpers.h"
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
  cudensitymatOperatorTerm_t tensor(const cudensitymatOperatorTerm_t &op1,
                                    const cudensitymatOperatorTerm_t &op2);

  // Multiplication of two operator terms
  cudensitymatOperatorTerm_t mul(const cudensitymatOperatorTerm_t &op1,
                                 const cudensitymatOperatorTerm_t &op2);

  // Addition of two operator terms
  cudensitymatOperatorTerm_t add(const cudensitymatOperatorTerm_t &op1,
                                 const cudensitymatOperatorTerm_t &op2);

  // Evaluate an operator and convert it to cudensitymatOperatorTerm_t
  cudensitymatOperatorTerm_t evaluate(const matrix_operator &op);

  // Convert a scalar to a cudensitymat operator term
  cudensitymatOperatorTerm_t _scalar_to_op(const scalar_operator &scalar);

  // Multiplies a scalar callback with a cudensitymat operator term
  cudensitymatOperatorTerm_t
  _callback_mult_op(cudensitymatScalarCallback_t scalar,
                    cudensitymatOperatorTerm_t op);

  // Wrap a matrix operator as a cudensitymat tensor callback
  cudensitymatTensorCallback_t _wrap_callback_tensor(const matrix_operator &op);

private:
  std::map<int, int> dimensions_;
  std::shared_ptr<Schedule> schedule_;
  cudensitymatHandle_t handle_;

  std::map<cudensitymatOperatorTerm_t,
           std::vector<cudensitymatElementaryOperator_t>>
      _termtoElemOps;
  std::map<cudensitymatOperatorTerm_t, std::vector<int32_t>> _termtoModes;
  std::map<cudensitymatOperatorTerm_t, std::vector<int32_t>> _termtoDuals;
};
} // namespace cudaq