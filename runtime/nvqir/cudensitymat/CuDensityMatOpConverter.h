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
#include <deque>
#include <unordered_set>

namespace cudaq {
namespace dynamics {
class CuDensityMatOpConverter {
public:
  CuDensityMatOpConverter(cudensitymatHandle_t handle) : m_handle(handle){};

  /// @brief Convert a matrix operator to a cudensity matrix operator.
  /// @param parameters The parameters of the operator.
  /// @param op The matrix operator to convert.
  /// @param modeExtents The extents of the modes.
  /// @return The converted operator.
  cudensitymatOperator_t convertToCudensitymatOperator(
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      const operator_sum<cudaq::matrix_operator> &op,
      const std::vector<int64_t> &modeExtents);

  /// @brief Construct a Liouvillian operator.
  /// @param ham The Hamiltonian operator.
  /// @param collapseOperators The collapse operators.
  /// @param modeExtents The extents of the modes.
  /// @param parameters The parameters of the operators.
  /// @param isMasterEquation Whether the Liouvillian is a master equation.
  /// @return The constructed Liouvillian operator.
  cudensitymatOperator_t constructLiouvillian(
      const operator_sum<cudaq::matrix_operator> &ham,
      const std::vector<operator_sum<cudaq::matrix_operator>>
          &collapseOperators,
      const std::vector<int64_t> &modeExtents,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      bool isMasterEquation);

  ~CuDensityMatOpConverter();

private:
  std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
  convertToCudensitymat(
      const operator_sum<cudaq::matrix_operator> &op,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      const std::vector<int64_t> &modeExtents);
  cudensitymatElementaryOperator_t createElementaryOperator(
      const cudaq::matrix_operator &elemOp,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      const std::vector<int64_t> &modeExtents);
  cudensitymatOperatorTerm_t createProductOperatorTerm(
      const std::vector<cudensitymatElementaryOperator_t> &elemOps,
      const std::vector<int64_t> &modeExtents,
      const std::vector<std::vector<int>> &degrees,
      const std::vector<std::vector<int>> &dualModalities);

  std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
  computeLindbladTerms(
      const operator_sum<cudaq::matrix_operator> &collapseOp,
      const std::vector<int64_t> &modeExtents,
      const std::unordered_map<std::string, std::complex<double>> &parameters);

  struct ScalarCallBackContext {
    scalar_operator scalarOp;
    std::vector<std::string> paramNames;
    ScalarCallBackContext(const scalar_operator &scalar_op,
                          const std::vector<std::string> &paramNames)
        : scalarOp(scalar_op), paramNames(paramNames){};
  };

  struct TensorCallBackContext {
    matrix_operator tensorOp;
    std::vector<std::string> paramNames;

    TensorCallBackContext(const matrix_operator &tensor_op,
                          const std::vector<std::string> &param_names)
        : tensorOp(tensor_op), paramNames(param_names){};
  };

  cudensitymatWrappedScalarCallback_t
  wrapScalarCallback(const scalar_operator &scalarOp,
                     const std::vector<std::string> &paramNames);
  cudensitymatWrappedTensorCallback_t
  wrapTensorCallback(const matrix_operator &matrixOp,
                     const std::vector<std::string> &paramNames);

private:
  cudensitymatHandle_t m_handle;
  // Things that we create that need to be cleaned up.
  // Use a set so that it's safe to push pointer multiple times.
  std::unordered_set<void *> m_deviceBuffers;
  std::unordered_set<cudensitymatElementaryOperator_t> m_elementaryOperators;
  std::unordered_set<cudensitymatOperatorTerm_t> m_operatorTerms;
  std::deque<ScalarCallBackContext> m_scalarCallbacks;
  std::deque<TensorCallBackContext> m_tensorCallbacks;
};
} // namespace dynamics
} // namespace cudaq
