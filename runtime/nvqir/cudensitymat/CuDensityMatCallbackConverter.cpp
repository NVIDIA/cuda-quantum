/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatOpConverter.h"
#include "CuDensityMatUtils.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include <iostream>
#include <map>
#include <ranges>

cudensitymatWrappedScalarCallback_t
cudaq::dynamics::CuDensityMatOpConverter::wrapScalarCallback(
    const std::vector<scalar_operator> &scalarOps,
    const std::vector<std::string> &paramNames) {
  m_scalarCallbacks.push_back(ScalarCallBackContext(scalarOps, paramNames));
  ScalarCallBackContext *storedCallbackContext = &m_scalarCallbacks.back();
  using WrapperFuncType =
      int32_t (*)(cudensitymatScalarCallback_t, double, int64_t, int32_t,
                  const double[], cudaDataType_t, void *);

  auto wrapper = [](cudensitymatScalarCallback_t callback, double time,
                    int64_t batchSize, int32_t numParams, const double params[],
                    cudaDataType_t dataType, void *scalarStorage) -> int32_t {
    try {
      ScalarCallBackContext *context =
          reinterpret_cast<ScalarCallBackContext *>(callback);
      if (numParams != 2 * context->paramNames.size())
        throw std::runtime_error(
            fmt::format("[Internal Error] Invalid number of callback "
                        "parameters encountered. Expected {} double params "
                        "representing {} complex values but received {}.",
                        2 * context->paramNames.size(),
                        context->paramNames.size(), numParams));
      if (batchSize != context->scalarOps.size())
        throw std::runtime_error(
            fmt::format("[Internal Error] Invalid batch size encountered. "
                        "Expected {} but received {}.",
                        context->scalarOps.size(), batchSize));
      auto *tdCoef = static_cast<std::complex<double> *>(scalarStorage);
      std::unordered_map<std::string, std::complex<double>> param_map;
      for (size_t i = 0; i < context->paramNames.size(); ++i) {
        param_map[context->paramNames[i]] =
            std::complex<double>(params[2 * i], params[2 * i + 1]);
        CUDAQ_DBG("Callback param name {}, batch size {}, value {}",
                  context->paramNames[i], batchSize,
                  param_map[context->paramNames[i]]);
      }
      for (int64_t i = 0; i < batchSize; ++i) {
        scalar_operator &storedOp = context->scalarOps[i];
        tdCoef[i] = storedOp.is_constant() ? storedOp.evaluate()
                                           : storedOp.evaluate(param_map);
        CUDAQ_DBG("Scalar callback constant value = {}", tdCoef[i]);
      }
      return CUDENSITYMAT_STATUS_SUCCESS;
    } catch (const std::exception &e) {
      std::cerr << "Error in scalar callback: " << e.what() << std::endl;
      return CUDENSITYMAT_STATUS_INTERNAL_ERROR;
    }
  };

  cudensitymatWrappedScalarCallback_t wrappedCallback;
  wrappedCallback.callback =
      reinterpret_cast<cudensitymatScalarCallback_t>(storedCallbackContext);
  wrappedCallback.device = CUDENSITYMAT_CALLBACK_DEVICE_CPU;
  wrappedCallback.wrapper =
      reinterpret_cast<void *>(static_cast<WrapperFuncType>(wrapper));
  return wrappedCallback;
}

cudensitymatWrappedTensorCallback_t
cudaq::dynamics::CuDensityMatOpConverter::wrapTensorCallback(
    const std::vector<matrix_handler> &matrixOps,
    const std::vector<std::string> &paramNames,
    const cudaq::dimension_map &dims) {
  m_tensorCallbacks.push_back(
      TensorCallBackContext(matrixOps, paramNames, dims));
  TensorCallBackContext *storedCallbackContext = &m_tensorCallbacks.back();
  using WrapperFuncType = int32_t (*)(
      cudensitymatTensorCallback_t, cudensitymatElementaryOperatorSparsity_t,
      int32_t, const int64_t[], const int32_t[], double, int64_t, int32_t,
      const double[], cudaDataType_t, void *, cudaStream_t);

  auto wrapper = [](cudensitymatTensorCallback_t callback,
                    cudensitymatElementaryOperatorSparsity_t sparsity,
                    int32_t num_modes, const int64_t modeExtents[],
                    const int32_t diagonal_offsets[], double time,
                    int64_t batchSize, int32_t num_params,
                    const double params[], cudaDataType_t data_type,
                    void *tensor_storage, cudaStream_t stream) -> int32_t {
    try {
      auto *context = reinterpret_cast<TensorCallBackContext *>(callback);
      std::vector<matrix_handler> &storedOps = context->tensorOps;
      const bool allSame = [&]() {
        const auto &firstOp = storedOps[0];
        return std::all_of(
            storedOps.begin(), storedOps.end(),
            [&firstOp](const auto &op) { return op == firstOp; });
      }();
      if (num_modes <= 0) {
        std::cerr << "num_modes is invalid: " << num_modes << std::endl;
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }

      if (num_params != 2 * context->paramNames.size())
        throw std::runtime_error(
            fmt::format("[Internal Error] Invalid number of tensor callback "
                        "parameters. Expected {} double values "
                        "representing {} complex parameters but received "
                        "{}.",
                        std::to_string(2 * context->paramNames.size()),
                        std::to_string(context->paramNames.size()),
                        std::to_string(num_params)));
      // If all ops are the same, we treat it as non-batched.
      if ((!allSame && batchSize != storedOps.size()) ||
          (batchSize > storedOps.size()))
        throw std::runtime_error(fmt::format(
            "[Internal Error] Invalid batch size encountered. "
            "Expected {} but received {}.",
            std::to_string(storedOps.size()), std::to_string(batchSize)));

      std::unordered_map<std::string, std::complex<double>> param_map;
      for (size_t i = 0; i < context->paramNames.size(); ++i) {
        param_map[context->paramNames[i]] =
            std::complex<double>(params[2 * i], params[2 * i + 1]);
        CUDAQ_DBG("Tensor callback param name {}, value {}",
                  context->paramNames[i], param_map[context->paramNames[i]]);
      }

      cudaq::dimension_map &dimensions = context->dimensions;
      std::size_t totalDim = 1;
      for (std::size_t i = 0; i < num_modes; ++i) {
        totalDim *= modeExtents[i];
      }

      if (dimensions.empty()) {
        std::cerr << "Dimension map is empty!" << std::endl;
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }
      const std::size_t tensorSize =
          sparsity == CUDENSITYMAT_OPERATOR_SPARSITY_NONE
              ? totalDim * totalDim * batchSize
              : totalDim * batchSize;
      const std::vector<std::complex<double>> flatMatrix = [&]() {
        std::vector<std::complex<double>> flatMatrix;
        flatMatrix.reserve(tensorSize);
        for (int i = 0; i < batchSize; ++i) {
          auto &storedOp = storedOps[i];
          if (sparsity == CUDENSITYMAT_OPERATOR_SPARSITY_NONE) {
            // Flatten the matrix in column-major order
            complex_matrix matrix_data =
                storedOp.to_matrix(dimensions, param_map);
            auto flattened = flattenMatrixColumnMajor(matrix_data);
            flatMatrix.insert(flatMatrix.end(), flattened.begin(),
                              flattened.end());
          } else {
            // Diagonal matrix case
            auto [mDiagData, _] =
                storedOp.to_diagonal_matrix(dimensions, param_map);
            flatMatrix.insert(flatMatrix.end(), mDiagData.begin(),
                              mDiagData.end());
          }
        }

        return flatMatrix;
      }();

      if (data_type == CUDA_C_64F) {
        memcpy(tensor_storage, flatMatrix.data(),
               flatMatrix.size() * sizeof(cuDoubleComplex));
      } else if (data_type == CUDA_C_32F) {
        std::vector<std::complex<float>> flatMatrix_float(flatMatrix.begin(),
                                                          flatMatrix.end());

        memcpy(tensor_storage, flatMatrix_float.data(),
               flatMatrix_float.size() * sizeof(cuFloatComplex));
      } else {
        std::cerr << "Invalid CUDA data type: " << data_type << std::endl;
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }

      return CUDENSITYMAT_STATUS_SUCCESS;
    } catch (const std::exception &e) {
      std::cerr << "Error in tensor callback: " << e.what() << std::endl;
      return CUDENSITYMAT_STATUS_INTERNAL_ERROR;
    }
  };

  cudensitymatWrappedTensorCallback_t wrappedCallback;
  wrappedCallback.callback =
      reinterpret_cast<cudensitymatTensorCallback_t>(storedCallbackContext);
  wrappedCallback.device = CUDENSITYMAT_CALLBACK_DEVICE_CPU;
  wrappedCallback.wrapper =
      reinterpret_cast<void *>(static_cast<WrapperFuncType>(wrapper));

  return wrappedCallback;
}
