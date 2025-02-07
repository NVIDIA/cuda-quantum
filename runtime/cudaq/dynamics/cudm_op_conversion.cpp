/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/cudm_op_conversion.h"
#include "cudaq/cudm_error_handling.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace cudaq;

namespace cudaq {
cudm_op_conversion::cudm_op_conversion(const cudensitymatHandle_t handle,
                                       const std::map<int, int> &dimensions,
                                       std::shared_ptr<Schedule> schedule)
    : handle_(handle), dimensions_(dimensions), schedule_(schedule) {}

cudensitymatOperatorTerm_t cudm_op_conversion::_scalar_to_op(
    const cudensitymatWrappedScalarCallback_t &scalar) {
  cudensitymatOperatorTerm_t op_term;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle_, dimensions_.size(),
                                                   nullptr, &op_term));

  cudensitymatElementaryOperator_t identity;
  HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
      handle_, 1, nullptr, CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr,
      CUDA_C_64F, nullptr, {nullptr, nullptr}, &identity));

  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
      handle_, op_term, 1, &identity, nullptr, nullptr, {1.0, 0.0}, scalar));

  return op_term;
}

cudensitymatOperatorTerm_t cudm_op_conversion::_callback_mult_op(
    const cudensitymatWrappedScalarCallback_t &scalar,
    const cudensitymatOperatorTerm_t &op) {
  if (!op) {
    throw std::invalid_argument("Invalid operator term (nullptr).");
  }

  cudensitymatOperatorTerm_t new_opterm;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle_, dimensions_.size(),
                                                   nullptr, &new_opterm));

  HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle_, new_opterm, op, 0,
                                                   {1.0, 0.0}, scalar));

  return new_opterm;
}

std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
             double>
cudm_op_conversion::tensor(
    const std::variant<cudensitymatOperatorTerm_t,
                       cudensitymatWrappedScalarCallback_t, double> &op1,
    const std::variant<cudensitymatOperatorTerm_t,
                       cudensitymatWrappedScalarCallback_t, double> &op2) {
  if (std::holds_alternative<double>(op1) ||
      std::holds_alternative<double>(op2)) {
    return std::get<double>(op1) * std::get<double>(op2);
  }

  cudensitymatOperatorTerm_t result;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle_, dimensions_.size(),
                                                   nullptr, &result));

  HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
      handle_, result, std::get<cudensitymatOperatorTerm_t>(op1), 0, {1.0, 0.0},
      {nullptr, nullptr}));
  HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
      handle_, result, std::get<cudensitymatOperatorTerm_t>(op2), 0, {1.0, 0.0},
      {nullptr, nullptr}));

  return result;
}

std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
             double>
cudm_op_conversion::mul(
    const std::variant<cudensitymatOperatorTerm_t,
                       cudensitymatWrappedScalarCallback_t, double> &op1,
    const std::variant<cudensitymatOperatorTerm_t,
                       cudensitymatWrappedScalarCallback_t, double> &op2) {
  return tensor(op1, op2);
}

std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
             double>
cudm_op_conversion::add(
    const std::variant<cudensitymatOperatorTerm_t,
                       cudensitymatWrappedScalarCallback_t, double> &op1,
    const std::variant<cudensitymatOperatorTerm_t,
                       cudensitymatWrappedScalarCallback_t, double> &op2) {
  if (std::holds_alternative<double>(op1) ||
      std::holds_alternative<double>(op2)) {
    return std::get<double>(op1) + std::get<double>(op2);
  }

  cudensitymatOperatorTerm_t result;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle_, dimensions_.size(),
                                                   nullptr, &result));

  HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
      handle_, result, std::get<cudensitymatOperatorTerm_t>(op1), 0, {1.0, 0.0},
      {nullptr, nullptr}));
  HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
      handle_, result, std::get<cudensitymatOperatorTerm_t>(op2), 0, {1.0, 0.0},
      {nullptr, nullptr}));

  return result;
}

std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
             std::complex<double>>
cudm_op_conversion::evaluate(
    const std::variant<scalar_operator, matrix_operator,
                       product_operator<matrix_operator>> &op) {
  if (std::holds_alternative<scalar_operator>(op)) {
    const scalar_operator &scalar_op = std::get<scalar_operator>(op);

    ScalarCallbackFunction generator = scalar_op.get_generator();

    if (!generator) {
      return scalar_op.evaluate({});
    } else {
      return _wrap_callback(scalar_op);
    }
  }

  if (std::holds_alternative<matrix_operator>(op)) {
    const matrix_operator &mat_op = std::get<matrix_operator>(op);

    cudensitymatOperatorTerm_t opterm;
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
        handle_, dimensions_.size(), nullptr, &opterm));

    cudensitymatElementaryOperator_t elem_op;
    cudensitymatWrappedTensorCallback_t callback =
        _wrap_callback_tensor(mat_op);

    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
        handle_, mat_op.degrees.size(), nullptr,
        CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr, CUDA_C_64F, nullptr,
        callback, &elem_op));

    HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
        handle_, opterm, 1, &elem_op, mat_op.degrees.data(), nullptr,
        {1.0, 0.0}, {nullptr, nullptr}));

    return opterm;
  }

  if (std::holds_alternative<product_operator<matrix_operator>>(op)) {
    throw std::runtime_error(
        "Handling of product_operator<matrix_operator> is not implemented.");
  }

  throw std::runtime_error(
      "Unknown operator type in cudm_op_conversion::evaluate.");
}

cudensitymatWrappedScalarCallback_t
cudm_op_conversion::_wrap_callback(const scalar_operator &scalar_op) {
  try {
    std::complex<double> evaluatedValue = scalar_op.evaluate({});

    cudensitymatWrappedScalarCallback_t wrapped_callback;
    wrapped_callback.callback = nullptr;
    wrapped_callback.wrapper = new std::complex<double>(evaluatedValue);
    return wrapped_callback;
  } catch (const std::exception &) {
  }

  ScalarCallbackFunction generator = scalar_op.get_generator();

  if (!generator) {
    throw std::runtime_error(
        "scalar_operator does not have a valid generator function.");
  }

  auto callback = [](double time, int32_t num_params, const double params[],
                     cudaDataType_t data_type,
                     void *scalar_storage) -> int32_t {
    try {
      scalar_operator *scalar_op =
          static_cast<scalar_operator *>(scalar_storage);

      std::map<std::string, std::complex<double>> param_map;
      for (size_t i = 0; i < num_params; i++) {
        param_map[std::to_string(i)] = params[i];
      }

      std::complex<double> result = scalar_op->evaluate(param_map);

      if (data_type == CUDA_C_64F) {
        *reinterpret_cast<cuDoubleComplex *>(scalar_storage) =
            make_cuDoubleComplex(result.real(), result.imag());
      } else if (data_type == CUDA_C_32F) {
        *reinterpret_cast<cuFloatComplex *>(scalar_storage) =
            make_cuFloatComplex(static_cast<float>(result.real()),
                                static_cast<float>(result.imag()));
      } else {
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }

      return CUDENSITYMAT_STATUS_SUCCESS;
    } catch (const std::exception &e) {
      std::cerr << "Error in scalar callback: " << e.what() << std::endl;
      return CUDENSITYMAT_STATUS_INTERNAL_ERROR;
    }
  };

  cudensitymatWrappedScalarCallback_t wrappedCallback;
  wrappedCallback.callback = callback;
  wrappedCallback.wrapper = new scalar_operator(scalar_op);

  return wrappedCallback;
}

cudensitymatWrappedTensorCallback_t
cudm_op_conversion::_wrap_callback_tensor(const matrix_operator &op) {
  auto callback =
      [](cudensitymatElementaryOperatorSparsity_t sparsity, int32_t num_modes,
         const int64_t mode_extents[], const int32_t diagonal_offsets[],
         double time, int32_t num_params, const double params[],
         cudaDataType_t data_type, void *tensor_storage) -> int32_t {
    try {
      matrix_operator *mat_op = static_cast<matrix_operator *>(tensor_storage);

      std::map<std::string, std::complex<double>> param_map;
      for (size_t i = 0; i < num_params; i++) {
        param_map[std::to_string(i)] = params[i];
      }

      matrix_2 matrix_data = mat_op->to_matrix({}, param_map);

      std::size_t rows = matrix_data.get_rows();
      std::size_t cols = matrix_data.get_columns();

      if (num_modes != rows) {
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }

      if (data_type == CUDA_C_64F) {
        cuDoubleComplex *storage =
            static_cast<cuDoubleComplex *>(tensor_storage);
        for (size_t i = 0; i < rows; i++) {
          for (size_t j = 0; j < cols; j++) {
            storage[i * cols + j] = make_cuDoubleComplex(
                matrix_data[{i, j}].real(), matrix_data[{i, j}].imag());
          }
        }
      } else if (data_type == CUDA_C_32F) {
        cuFloatComplex *storage = static_cast<cuFloatComplex *>(tensor_storage);
        for (size_t i = 0; i < rows; i++) {
          for (size_t j = 0; j < cols; j++) {
            storage[i * cols + j] = make_cuFloatComplex(
                static_cast<float>(matrix_data[{i, j}].real()),
                static_cast<float>(matrix_data[{i, j}].imag()));
          }
        }
      } else {
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }

      return CUDENSITYMAT_STATUS_SUCCESS;
    } catch (const std::exception &e) {
      std::cerr << "Error in tensor callback: " << e.what() << std::endl;
      return CUDENSITYMAT_STATUS_INTERNAL_ERROR;
    }
  };

  cudensitymatWrappedTensorCallback_t wrapped_callback;
  wrapped_callback.callback = callback;
  wrapped_callback.wrapper = new matrix_operator(op);

  return wrapped_callback;
}

} // namespace cudaq