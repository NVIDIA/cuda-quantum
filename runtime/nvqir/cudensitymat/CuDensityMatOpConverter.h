/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/operators.h"
#include "cudaq/utils/matrix.h"
#include <cudensitymat.h>
#include <deque>
#include <unordered_set>

namespace cudaq::dynamics {
class CuDensityMatOpConverter {
public:
  CuDensityMatOpConverter(cudensitymatHandle_t handle);

  /// @brief Convert a matrix operator to a `cudensitymat` matrix operator.
  /// @param parameters The parameters of the operator.
  /// @param ops The matrix operator to convert.
  /// @param modeExtents The extents of the modes.
  /// @return The converted operator.
  cudensitymatOperator_t convertToCudensitymatOperator(
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      const std::vector<sum_op<cudaq::matrix_handler>> &ops,
      const std::vector<int64_t> &modeExtents);
  cudensitymatOperator_t convertToCudensitymatOperator(
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      const sum_op<cudaq::matrix_handler> &op,
      const std::vector<int64_t> &modeExtents) {
    return convertToCudensitymatOperator(
        parameters, std::vector<sum_op<cudaq::matrix_handler>>{op},
        modeExtents);
  }

  /// @brief Construct a Liouvillian operator.
  /// @param hamOperators The Hamiltonian operator.
  /// @param collapseOperators The collapse operators.
  /// @param modeExtents The extents of the modes.
  /// @param parameters The parameters of the operators.
  /// @param isMasterEquation Whether the Liouvillian is a master equation.
  /// @return The constructed Liouvillian operator.
  cudensitymatOperator_t constructLiouvillian(
      const std::vector<sum_op<cudaq::matrix_handler>> &hamOperators,
      const std::vector<std::vector<sum_op<cudaq::matrix_handler>>>
          &collapseOperators,
      const std::vector<int64_t> &modeExtents,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      bool isMasterEquation);
  /// @brief  Construct a Liouvillian operator from a super operator.
  /// @param superOps The super operators.
  /// @param modeExtents The extents of the modes.
  /// @param parameters The parameters of the operators.
  /// @return The constructed Liouvillian operator.
  cudensitymatOperator_t constructLiouvillian(
      const std::vector<super_op> &superOps,
      const std::vector<int64_t> &modeExtents,
      const std::unordered_map<std::string, std::complex<double>> &parameters);

  /// @brief Clear the current callback context
  // Callback context may contain Python objects, hence needs to be clear before
  // shutdown to prevent race condition.
  void clearCallbackContext();

  ~CuDensityMatOpConverter();

private:
  cudensitymatOperatorTerm_t createBatchedProductTerm(
      const std::vector<product_op<cudaq::matrix_handler>> &prodTerms,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      const std::vector<int64_t> &modeExtents);

  std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
  convertToCudensitymat(
      const sum_op<cudaq::matrix_handler> &op,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      const std::vector<int64_t> &modeExtents);
  cudensitymatElementaryOperator_t createElementaryOperator(
      const std::vector<cudaq::matrix_handler> &elemOps,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      const std::vector<int64_t> &modeExtents);
  cudensitymatOperatorTerm_t createProductOperatorTerm(
      const std::vector<cudensitymatElementaryOperator_t> &elemOps,
      const std::vector<int64_t> &modeExtents,
      const std::vector<std::vector<std::size_t>> &degrees,
      const std::vector<std::vector<int>> &dualModalities);

  std::vector<std::pair<std::vector<cudaq::scalar_operator>,
                        cudensitymatOperatorTerm_t>>
  computeLindbladTerms(
      const std::vector<sum_op<cudaq::matrix_handler>> &batchedCollapseOps,
      const std::vector<int64_t> &modeExtents,
      const std::unordered_map<std::string, std::complex<double>> &parameters);

  struct ScalarCallBackContext {
    std::vector<scalar_operator> scalarOps;
    std::vector<std::string> paramNames;
    ScalarCallBackContext(const std::vector<scalar_operator> &scalar_ops,
                          const std::vector<std::string> &paramNames)
        : scalarOps(scalar_ops), paramNames(paramNames){};
  };

  struct TensorCallBackContext {
    std::vector<matrix_handler> tensorOps;
    std::vector<std::string> paramNames;
    cudaq::dimension_map dimensions;

    TensorCallBackContext(const std::vector<matrix_handler> &tensor_ops,
                          const std::vector<std::string> &param_names,
                          const cudaq::dimension_map &dims)
        : tensorOps(tensor_ops), paramNames(param_names), dimensions(dims){};
  };

  cudensitymatWrappedScalarCallback_t
  wrapScalarCallback(const std::vector<scalar_operator> &scalarOps,
                     const std::vector<std::string> &paramNames);
  cudensitymatWrappedTensorCallback_t
  wrapTensorCallback(const std::vector<matrix_handler> &matrixOps,
                     const std::vector<std::string> &paramNames,
                     const cudaq::dimension_map &dims);
  void appendToCudensitymatOperator(
      cudensitymatOperator_t &cudmOperator,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      const std::vector<sum_op<cudaq::matrix_handler>> &ops,
      const std::vector<int64_t> &modeExtents, int32_t duality);

  static std::vector<std::complex<double>>
  flattenMatrixColumnMajor(const cudaq::complex_matrix &matrix);

  static std::vector<std::vector<product_op<cudaq::matrix_handler>>>
  splitToBatch(const std::vector<sum_op<cudaq::matrix_handler>> &ops);

  void appendBatchedTermToOperator(cudensitymatOperator_t op,
                                   cudensitymatOperatorTerm_t term,
                                   const std::vector<scalar_operator> coeffs,
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
  int m_minDimensionDiag = 4;
  int m_maxDiagonalsDiag = 1;
};
} // namespace cudaq::dynamics
