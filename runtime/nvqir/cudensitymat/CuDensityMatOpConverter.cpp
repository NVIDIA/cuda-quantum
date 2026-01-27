/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatOpConverter.h"
#include "BatchingUtils.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatUtils.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include <iostream>
#include <map>
#include <ranges>

namespace {
std::vector<int64_t>
getSubspaceExtents(const std::vector<int64_t> &modeExtents,
                   const std::vector<std::size_t> &degrees) {
  std::vector<int64_t> subspaceExtents;

  for (std::size_t degree : degrees) {
    if (degree >= modeExtents.size())
      throw std::out_of_range("Degree exceeds modeExtents size.");

    subspaceExtents.push_back(modeExtents[degree]);
  }

  return subspaceExtents;
}

cudaq::dimension_map
convertDimensions(const std::vector<int64_t> &modeExtents) {

  cudaq::dimension_map dimensions;
  for (size_t i = 0; i < modeExtents.size(); ++i)
    dimensions[i] = static_cast<std::size_t>(modeExtents[i]);

  return dimensions;
}

} // namespace

// Function to flatten a matrix into a 1D array (column major)
std::vector<std::complex<double>>
cudaq::dynamics::CuDensityMatOpConverter::flattenMatrixColumnMajor(
    const cudaq::complex_matrix &matrix) {
  std::vector<std::complex<double>> flatMatrix;
  flatMatrix.reserve(matrix.size());
  for (size_t col = 0; col < matrix.cols(); col++) {
    for (size_t row = 0; row < matrix.rows(); row++) {
      flatMatrix.push_back(matrix[{row, col}]);
    }
  }

  return flatMatrix;
}

std::vector<std::vector<cudaq::product_op<cudaq::matrix_handler>>>
cudaq::dynamics::CuDensityMatOpConverter::splitToBatch(
    const std::vector<sum_op<cudaq::matrix_handler>> &ops) {
  if (ops.empty())
    throw std::invalid_argument("At least 1 operator is required");
  const std::size_t num_terms = ops.front().num_terms();
  for (std::size_t i = 1; i < ops.size(); ++i) {
    if (ops[i].num_terms() != num_terms) {
      throw std::invalid_argument(
          "All operators must have the same number of terms");
    }
  }

  // Split the operators into batches.
  std::vector<std::vector<product_op<cudaq::matrix_handler>>> batches(
      ops.size());
  for (auto &productOps : batches) {
    productOps.reserve(num_terms);
  }
  for (std::size_t i = 0; i < ops.size(); ++i) {
    for (std::size_t j = 0; j < num_terms; ++j) {
      batches[i].emplace_back(ops[i][j]);
    }
  }
  for (auto &productOps : batches) {
    // Sort the product terms by their degrees.
    std::ranges::stable_sort(productOps.begin(), productOps.end(),
                             [](const product_op<cudaq::matrix_handler> &lhs,
                                const product_op<cudaq::matrix_handler> &rhs) {
                               // Compare the degrees of the product terms.
                               return lhs.degrees() < rhs.degrees();
                             });
  }
  return batches;
}

cudaq::dynamics::CuDensityMatOpConverter::CuDensityMatOpConverter(
    cudensitymatHandle_t handle)
    : m_handle(handle) {
  const auto getIntEnvVarIfPresent =
      [](const char *envName) -> std::optional<int> {
    if (auto *envVal = std::getenv(envName)) {
      const std::string envValStr(envVal);
      const char *nptr = envValStr.data();
      char *endptr = nullptr;
      errno = 0; // reset errno to 0 before call
      auto envIntVal = strtol(nptr, &endptr, 10);

      if (nptr == endptr || errno != 0 || envIntVal < 0)
        throw std::runtime_error(fmt::format(
            "Invalid {} setting. Expected a non-negative number. Got: '{}'",
            envName, envValStr));

      return envIntVal;
    }
    // The environment variable is not set.
    return std::nullopt;
  };

  {
    const auto minDim =
        getIntEnvVarIfPresent("CUDAQ_DYNAMICS_MIN_MULTIDIAGONAL_DIMENSION");
    if (minDim.has_value()) {
      CUDAQ_INFO("Setting multi-diagonal min dimension to {}.", minDim.value());
      m_minDimensionDiag = minDim.value();
    }
  }

  {
    const auto maxDiags = getIntEnvVarIfPresent(
        "CUDAQ_DYNAMICS_MAX_DIAGONAL_COUNT_FOR_MULTIDIAGONAL");
    if (maxDiags.has_value()) {
      CUDAQ_INFO("Setting multi-diagonal max number of diagonals to {}.",
                 maxDiags.value());
      m_maxDiagonalsDiag = maxDiags.value();
    }
  }
}

void cudaq::dynamics::CuDensityMatOpConverter::clearCallbackContext() {
  m_scalarCallbacks.clear();
  m_tensorCallbacks.clear();
}

cudaq::dynamics::CuDensityMatOpConverter::~CuDensityMatOpConverter() {
  for (auto term : m_operatorTerms)
    cudensitymatDestroyOperatorTerm(term);

  for (auto op : m_elementaryOperators)
    cudensitymatDestroyElementaryOperator(op);

  for (auto *buffer : m_deviceBuffers) {
    cudaq::dynamics::DeviceAllocator::free(buffer);
  }
}

cudensitymatElementaryOperator_t
cudaq::dynamics::CuDensityMatOpConverter::createElementaryOperator(
    const std::vector<cudaq::matrix_handler> &elemOps,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const std::vector<int64_t> &modeExtents) {
  if (elemOps.empty()) {
    throw std::invalid_argument(
        "Elementary operator cannot be created from an empty vector of "
        "elementary operators.");
  }
  const bool allSame = [&]() {
    const auto &firstOp = elemOps[0];
    return std::all_of(elemOps.begin(), elemOps.end(),
                       [&firstOp](const auto &op) { return op == firstOp; });
  }();

  const bool isBatched = elemOps.size() > 1 && !allSame;
  // We should have validated the batching compatibility.
  assert(!isBatched ||
         cudaq::__internal__::checkBatchingCompatibility(elemOps));

  auto subspaceExtents = getSubspaceExtents(modeExtents, elemOps[0].degrees());
  cudaq::dimension_map dimensions = convertDimensions(modeExtents);
  cudensitymatWrappedTensorCallback_t wrappedTensorCallback =
      cudensitymatTensorCallbackNone;

  static const std::vector<std::string> g_knownNonParametricOps = []() {
    std::vector<std::string> opNames;
    opNames.emplace_back(
        cudaq::boson_op::identity(0).begin()->to_string(false));
    // These are ops that we created during lindblad generation
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::boson_op::create(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(
        cudaq::boson_op::annihilate(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::boson_op::number(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::spin_op::i(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::spin_op::x(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::spin_op::y(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::spin_op::z(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    return opNames;
  }();

  const bool isCallbackTensor = [&]() {
    const auto checkIfCanEvaluateWithoutParam =
        [](const cudaq::matrix_handler &op,
           std::unordered_map<std::size_t, std::int64_t> &dimensions) {
          try {
            op.to_matrix(dimensions, {});
            return true;
          } catch (const std::exception &e) {
            return false;
          }
        };

    for (const auto &elemOp : elemOps) {
      if (std::find(g_knownNonParametricOps.begin(),
                    g_knownNonParametricOps.end(),
                    elemOp.to_string(false)) == g_knownNonParametricOps.end() &&
          !checkIfCanEvaluateWithoutParam(elemOp, dimensions)) {
        return true;
      }
    }
    return false;
  }();

  // This is a callback
  if (!parameters.empty() && isCallbackTensor) {
    const std::map<std::string, std::complex<double>> sortedParameters(
        parameters.begin(), parameters.end());
    auto ks = std::views::keys(sortedParameters);
    const std::vector<std::string> keys{ks.begin(), ks.end()};
    wrappedTensorCallback = wrapTensorCallback(elemOps, keys, dimensions);
  }

  const auto batchSize = elemOps.size();
  bool shouldUseDia = false;
  std::vector<int32_t> diagonalOffsets;
  auto *elementaryMat_d = [&]() {
    if (!isBatched) {
      auto &elemOp = elemOps[0];
      const auto [diags, offsets] =
          elemOp.to_diagonal_matrix(dimensions, parameters);
      shouldUseDia = [&]() {
        if (diags.empty())
          return false;
        const auto dim = std::accumulate(
            subspaceExtents.begin(), subspaceExtents.end(), 1,
            std::multiplies<decltype(subspaceExtents)::value_type>());
        if (dim < m_minDimensionDiag)
          return false;
        return offsets.size() <= m_maxDiagonalsDiag;
      }();

      if (shouldUseDia) {
        diagonalOffsets.assign(offsets.begin(), offsets.end());
        return cudaq::dynamics::createArrayGpu(diags);
      }
      auto flatMatrix =
          flattenMatrixColumnMajor(elemOp.to_matrix(dimensions, parameters));

      if (flatMatrix.empty())
        throw std::invalid_argument(
            "Input matrix (flat matrix) cannot be empty.");

      return cudaq::dynamics::createArrayGpu(flatMatrix);
    }

    const int64_t totalDim =
        std::accumulate(subspaceExtents.begin(), subspaceExtents.end(), 1,
                        std::multiplies<int64_t>());
    std::vector<std::complex<double>> tensorData;
    tensorData.reserve(totalDim * totalDim * batchSize);
    for (const auto &elementaryOp : elemOps) {
      const auto flatMatrix = flattenMatrixColumnMajor(
          elementaryOp.to_matrix(dimensions, parameters));
      tensorData.insert(tensorData.end(), flatMatrix.begin(), flatMatrix.end());
    }
    return cudaq::dynamics::createArrayGpu(tensorData);
  }();

  cudensitymatElementaryOperator_t cudmElemOp = nullptr;
  if (shouldUseDia) {
    assert(!isBatched);
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
        m_handle, static_cast<int32_t>(subspaceExtents.size()),
        subspaceExtents.data(), CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL,
        diagonalOffsets.size(), diagonalOffsets.data(), CUDA_C_64F,
        elementaryMat_d, wrappedTensorCallback,
        cudensitymatTensorGradientCallbackNone, &cudmElemOp));
  } else {
    if (isBatched) {
      HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperatorBatch(
          m_handle, static_cast<int32_t>(subspaceExtents.size()),
          subspaceExtents.data(), batchSize,
          CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr, CUDA_C_64F,
          elementaryMat_d, wrappedTensorCallback,
          cudensitymatTensorGradientCallbackNone, &cudmElemOp));
    } else {
      HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
          m_handle, static_cast<int32_t>(subspaceExtents.size()),
          subspaceExtents.data(), CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0,
          nullptr, CUDA_C_64F, elementaryMat_d, wrappedTensorCallback,
          cudensitymatTensorGradientCallbackNone, &cudmElemOp));
    }
  }

  if (!cudmElemOp) {
    std::cerr << "[ERROR] cudmElemOp is NULL in createElementaryOperator !"
              << std::endl;
    cudaq::dynamics::destroyArrayGpu(elementaryMat_d);
    throw std::runtime_error("Failed to create elementary operator.");
  }
  m_elementaryOperators.emplace(cudmElemOp);
  m_deviceBuffers.emplace(elementaryMat_d);
  return cudmElemOp;
}

cudensitymatOperatorTerm_t
cudaq::dynamics::CuDensityMatOpConverter::createProductOperatorTerm(
    const std::vector<cudensitymatElementaryOperator_t> &elemOps,
    const std::vector<int64_t> &modeExtents,
    const std::vector<std::vector<std::size_t>> &degrees,
    const std::vector<std::vector<int>> &dualModalities) {

  cudensitymatOperatorTerm_t term;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
      m_handle, static_cast<int32_t>(modeExtents.size()), modeExtents.data(),
      &term));
  m_operatorTerms.emplace(term);
  if (degrees.empty())
    throw std::invalid_argument("Degrees vector cannot be empty.");

  if (elemOps.empty())
    throw std::invalid_argument("elemOps cannot be null.");

  if (degrees.size() != elemOps.size())
    throw std::invalid_argument("elemOps and degrees must have the same size.");

  const bool hasDualModalities = !dualModalities.empty();

  if (hasDualModalities && degrees.size() != dualModalities.size())
    throw std::invalid_argument(
        "degrees and dualModalities must have the same size.");

  std::vector<int32_t> allDegrees;
  std::vector<int32_t> allModeActionDuality;
  for (size_t i = 0; i < degrees.size(); i++) {
    const auto &sub_degrees = degrees[i];
    const auto &modalities = hasDualModalities
                                 ? dualModalities[i]
                                 : std::vector<int>(sub_degrees.size(), 0);

    if (sub_degrees.size() != modalities.size())
      throw std::runtime_error(
          "Mismatch between degrees and modalities sizes.");

    for (size_t j = 0; j < sub_degrees.size(); j++) {
      std::size_t degree = sub_degrees[j];
      int modality = modalities[j];

      if (sub_degrees[i] < 0)
        throw std::out_of_range("Degree cannot be negative!");

      allDegrees.emplace_back(degree);
      allModeActionDuality.emplace_back(modality);
    }
  }

  assert(elemOps.size() == degrees.size());
  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
      m_handle, term, static_cast<int32_t>(elemOps.size()), elemOps.data(),
      allDegrees.data(), allModeActionDuality.data(),
      make_cuDoubleComplex(1.0, 0.0), cudensitymatScalarCallbackNone,
      cudensitymatScalarGradientCallbackNone));
  return term;
}

cudensitymatOperator_t
cudaq::dynamics::CuDensityMatOpConverter::convertToCudensitymatOperator(
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const std::vector<sum_op<cudaq::matrix_handler>> &ops,
    const std::vector<int64_t> &modeExtents) {
  if (ops.empty())
    throw std::invalid_argument(
        "Operator sum cannot be empty. At least one operator is required.");

  const auto numberProductTerms = ops[0].num_terms();
  if (numberProductTerms == 0)
    throw std::invalid_argument(
        "Operator sum must have at least one product term.");
  for (const auto &op : ops) {
    if (op.num_terms() != numberProductTerms) {
      throw std::invalid_argument(
          "All operators in the sum must have the same number of terms.");
    }
  }

  cudensitymatOperator_t cudmOperator;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
      m_handle, static_cast<int32_t>(modeExtents.size()), modeExtents.data(),
      &cudmOperator));

  appendToCudensitymatOperator(cudmOperator, parameters, ops, modeExtents,
                               /*duality=*/0);

  return cudmOperator;
}

void cudaq::dynamics::CuDensityMatOpConverter::appendToCudensitymatOperator(
    cudensitymatOperator_t &cudmOperator,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const std::vector<sum_op<cudaq::matrix_handler>> &ops,
    const std::vector<int64_t> &modeExtents, int32_t duality) {
  if (ops.empty())
    throw std::invalid_argument(
        "Operator sum cannot be empty. At least one operator is required.");

  const auto numberProductTerms = ops[0].num_terms();
  if (numberProductTerms == 0)
    throw std::invalid_argument(
        "Operator sum must have at least one product term.");
  for (const auto &op : ops) {
    if (op.num_terms() != numberProductTerms) {
      throw std::invalid_argument(
          "All operators in the sum must have the same number of terms.");
    }
  }

  const std::map<std::string, std::complex<double>> sortedParameters(
      parameters.begin(), parameters.end());
  auto ks = std::views::keys(sortedParameters);
  const std::vector<std::string> keys{ks.begin(), ks.end()};
  const bool isBatched = ops.size() > 1;
  if (!isBatched) {
    auto &op = ops[0];
    for (auto &[coeff, term] :
         convertToCudensitymat(op, parameters, modeExtents)) {
      cudensitymatWrappedScalarCallback_t wrappedCallback =
          cudensitymatScalarCallbackNone;

      if (coeff.is_constant()) {
        const auto coeffVal = coeff.evaluate();
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            m_handle, cudmOperator, term, /*duality=*/duality,
            make_cuDoubleComplex(coeffVal.real(), coeffVal.imag()),
            wrappedCallback, cudensitymatScalarGradientCallbackNone));
      } else {
        wrappedCallback = wrapScalarCallback({coeff}, keys);
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            m_handle, cudmOperator, term, /*duality=*/duality,
            make_cuDoubleComplex(1.0, 0.0), wrappedCallback,
            cudensitymatScalarGradientCallbackNone));
      }
    }
  } else {
    // Split the operators into batches.
    auto batchedProductTerms = splitToBatch(ops);
    const auto numberProductTerms = batchedProductTerms[0].size();
    for (std::size_t termIdx = 0; termIdx < numberProductTerms; ++termIdx) {
      std::vector<cudaq::product_op<cudaq::matrix_handler>> prodTerms;
      std::vector<std::complex<double>> batchedProductTermCoeffs;
      bool allConstant = true;
      for (const auto &productTerms : batchedProductTerms) {
        prodTerms.emplace_back(productTerms[termIdx]);
        const auto coeffVal = productTerms[termIdx].get_coefficient();
        if (!coeffVal.is_constant()) {
          allConstant = false;
        }
        if (allConstant) {
          // If all coefficients are constant, we can evaluate them now
          // and store them in the batchedProductTermCoeffs vector.
          // This avoids the need for a callback.
          batchedProductTermCoeffs.emplace_back(coeffVal.evaluate());
        }
      }

      const auto allSameDegrees = std::all_of(
          prodTerms.begin(), prodTerms.end(), [&](const auto &prodTerm) {
            return prodTerm.degrees() == prodTerms[0].degrees();
          });
      if (!allSameDegrees) {
        throw std::invalid_argument(
            "All product terms must have the same degrees.");
      }

      auto cudmProductTerm = [&]() {
        const auto allSameOp = std::all_of(
            prodTerms.begin(), prodTerms.end(), [&](const auto &prodTerm) {
              return prodTerm.get_term_id() == prodTerms[0].get_term_id();
            });
        if (!allSameOp) {
          return createBatchedProductTerm(prodTerms, parameters, modeExtents);
        }

        auto convertedResults =
            convertToCudensitymat(prodTerms[0], parameters, modeExtents);
        assert(convertedResults.size() == 1);
        return convertedResults[0].second;
      }();

      const auto batchSize = prodTerms.size();

      if (allConstant) {
        cuDoubleComplex *staticCoefficients_d = static_cast<cuDoubleComplex *>(
            cudaq::dynamics::createArrayGpu(batchedProductTermCoeffs));
        m_deviceBuffers.emplace(staticCoefficients_d);
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTermBatch(
            m_handle, cudmOperator, cudmProductTerm, /*duality=*/duality,
            /*batchSize=*/batchSize,
            /*staticCoefficients=*/staticCoefficients_d, nullptr,
            cudensitymatScalarCallbackNone,
            cudensitymatScalarGradientCallbackNone));
      } else {
        cuDoubleComplex *staticCoefficients_d = static_cast<cuDoubleComplex *>(
            cudaq::dynamics::createArrayGpu(std::vector<std::complex<double>>(
                batchSize, std::complex<double>(1.0, 0.0))));
        m_deviceBuffers.emplace(staticCoefficients_d);
        std::vector<cudaq::scalar_operator> coeffs;
        coeffs.reserve(batchSize);
        // Fix: Use sorted batchedProductTerms instead of unsorted ops to get
        // the correct coefficient for each term after sorting by degrees.
        for (const auto &productTerms : batchedProductTerms) {
          coeffs.emplace_back(productTerms[termIdx].get_coefficient());
        }
        cuDoubleComplex *totalCoefficients_d = static_cast<cuDoubleComplex *>(
            cudaq::dynamics::createArrayGpu(std::vector<std::complex<double>>(
                batchSize, std::complex<double>(0.0, 0.0))));
        m_deviceBuffers.emplace(totalCoefficients_d);
        auto wrappedCallback = wrapScalarCallback(coeffs, keys);
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTermBatch(
            m_handle, cudmOperator, cudmProductTerm, /*duality=*/duality,
            /*batchSize=*/batchSize,
            /*staticCoefficients=*/staticCoefficients_d,
            /*totalCoefficients=*/totalCoefficients_d, wrappedCallback,
            cudensitymatScalarGradientCallbackNone));
      }
    }
  }
}

cudensitymatOperatorTerm_t
cudaq::dynamics::CuDensityMatOpConverter::createBatchedProductTerm(
    const std::vector<product_op<cudaq::matrix_handler>> &prodTerms,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const std::vector<int64_t> &modeExtents) {
  if (prodTerms.empty())
    throw std::invalid_argument("Product terms cannot be empty. At least one "
                                "product term is required.");
  // Note: all these product terms must be acting on the same degrees.
  assert(std::all_of(prodTerms.begin(), prodTerms.end(),
                     [&](const product_op<cudaq::matrix_handler> &prodTerm) {
                       return prodTerm.degrees() == prodTerms[0].degrees();
                     }));
  // The number of elementary operators in each product
  // term must be the same.
  assert(std::all_of(prodTerms.begin(), prodTerms.end(),
                     [&](const product_op<cudaq::matrix_handler> &prodTerm) {
                       return prodTerm.num_ops() == prodTerms[0].num_ops();
                     }));
  const auto numOps = prodTerms[0].num_ops();
  const auto batchSize = prodTerms.size();
  std::vector<cudensitymatElementaryOperator_t> elemOps;
  std::vector<std::vector<std::size_t>> allDegrees;
  for (std::size_t i = 0; i < numOps; ++i) {
    std::vector<cudaq::matrix_handler> elementaryOps;
    elementaryOps.reserve(batchSize);
    for (const auto &prodTerm : prodTerms) {
      elementaryOps.emplace_back(prodTerm[i]);
    }
    auto cudmElemOp =
        createElementaryOperator(elementaryOps, parameters, modeExtents);
    elemOps.emplace_back(cudmElemOp);
    allDegrees.emplace_back(elementaryOps[0].degrees());
  }
  std::reverse(elemOps.begin(), elemOps.end());
  std::reverse(allDegrees.begin(), allDegrees.end());
  return createProductOperatorTerm(elemOps, modeExtents, allDegrees, {});
}

std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
cudaq::dynamics::CuDensityMatOpConverter::convertToCudensitymat(
    const sum_op<cudaq::matrix_handler> &op,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const std::vector<int64_t> &modeExtents) {
  if (op.num_terms() == 0)
    throw std::invalid_argument("Operator sum cannot be empty.");

  std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
      result;

  for (const auto &productOp : op) {
    std::vector<cudensitymatElementaryOperator_t> elemOps;
    std::vector<std::vector<std::size_t>> allDegrees;
    for (const auto &component : productOp) {
      // No need to check type
      // just call to_matrix on it
      if (const auto *elemOp =
              dynamic_cast<const cudaq::matrix_handler *>(&component)) {
        auto cudmElemOp =
            createElementaryOperator({*elemOp}, parameters, modeExtents);
        elemOps.emplace_back(cudmElemOp);
        allDegrees.emplace_back(elemOp->degrees());
      } else {
        // Catch anything that we don't know
        throw std::runtime_error("Unhandled type!");
      }
    }
    // Note: the order of operator application is the opposite of the writing:
    // i.e., ABC means C to be applied first.
    std::reverse(elemOps.begin(), elemOps.end());
    std::reverse(allDegrees.begin(), allDegrees.end());
    if (elemOps.empty()) {
      // Constant term (no operator)
      cudaq::product_op<cudaq::matrix_handler> constantTerm =
          cudaq::sum_op<cudaq::matrix_handler>::identity(0);
      cudensitymatElementaryOperator_t cudmElemOp = createElementaryOperator(
          {*constantTerm.begin()}, parameters, modeExtents);
      result.emplace_back(std::make_pair(
          productOp.get_coefficient(),
          createProductOperatorTerm({cudmElemOp}, modeExtents, {{0}}, {})));
    } else {
      result.emplace_back(std::make_pair(
          productOp.get_coefficient(),
          createProductOperatorTerm(elemOps, modeExtents, allDegrees, {})));
    }
  }
  return result;
}

void cudaq::dynamics::CuDensityMatOpConverter::appendBatchedTermToOperator(
    cudensitymatOperator_t op, cudensitymatOperatorTerm_t term,
    const std::vector<scalar_operator> coeffs,
    const std::vector<std::string> &paramNames) {
  const auto batchSize = coeffs.size();
  cudensitymatWrappedScalarCallback_t wrappedCallback =
      cudensitymatScalarCallbackNone;

  if (batchSize == 1) {
    auto &coeff = coeffs[0];
    if (coeff.is_constant()) {
      const auto coeffVal = coeff.evaluate();
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          m_handle, op, term, 0,
          make_cuDoubleComplex(coeffVal.real(), coeffVal.imag()),
          wrappedCallback, cudensitymatScalarGradientCallbackNone));
    } else {
      wrappedCallback = wrapScalarCallback({coeff}, paramNames);
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          m_handle, op, term, 0, make_cuDoubleComplex(1.0, 0.0),
          wrappedCallback, cudensitymatScalarGradientCallbackNone));
    }
  } else {
    const bool allConstant = std::all_of(
        coeffs.begin(), coeffs.end(), [](const cudaq::scalar_operator &scalar) {
          return scalar.is_constant();
        });

    if (allConstant) {
      std::vector<std::complex<double>> batchedCoeffs;
      batchedCoeffs.reserve(batchSize);
      for (const auto &coeff : coeffs) {
        batchedCoeffs.push_back(coeff.evaluate());
      }
      cuDoubleComplex *staticCoefficients_d = static_cast<cuDoubleComplex *>(
          cudaq::dynamics::createArrayGpu(batchedCoeffs));
      m_deviceBuffers.emplace(staticCoefficients_d);
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTermBatch(
          m_handle, op, term, /*duality=*/0,
          /*batchSize=*/batchSize,
          /*staticCoefficients=*/staticCoefficients_d, nullptr,
          cudensitymatScalarCallbackNone,
          cudensitymatScalarGradientCallbackNone));
    } else {
      cuDoubleComplex *staticCoefficients_d = static_cast<cuDoubleComplex *>(
          cudaq::dynamics::createArrayGpu(std::vector<std::complex<double>>(
              batchSize, std::complex<double>(1.0, 0.0))));
      m_deviceBuffers.emplace(staticCoefficients_d);
      cuDoubleComplex *totalCoefficients_d = static_cast<cuDoubleComplex *>(
          cudaq::dynamics::createArrayGpu(std::vector<std::complex<double>>(
              batchSize, std::complex<double>(0.0, 0.0))));
      m_deviceBuffers.emplace(totalCoefficients_d);
      wrappedCallback = wrapScalarCallback(coeffs, paramNames);
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTermBatch(
          m_handle, op, term, /*duality=*/0,
          /*batchSize=*/batchSize,
          /*staticCoefficients=*/staticCoefficients_d,
          /*totalCoefficients=*/totalCoefficients_d, wrappedCallback,
          cudensitymatScalarGradientCallbackNone));
    }
  }
}
