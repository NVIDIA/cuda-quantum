/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudm_helpers.h"
#include "common/Logger.h"
#include "cudm_error_handling.h"
#include <ranges>

using namespace cudaq;

namespace cudaq {
cudm_helper::cudm_helper(cudensitymatHandle_t handle) : handle(handle) {}

cudm_helper::~cudm_helper() {
  cudaDeviceSynchronize();

  for (auto term : m_operatorTerms) {
    cudensitymatDestroyOperatorTerm(term);
  }
  for (auto op : m_elementaryOperators) {
    cudensitymatDestroyElementaryOperator(op);
  }
  for (auto *buffer : m_deviceBuffers) {
    cudaFree(buffer);
  }
}

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
cudm_helper::_wrap_callback(const scalar_operator &scalar_op,
                            const std::vector<std::string> &paramNames) {
  if (scalar_op.is_constant()) {
    throw std::runtime_error(
        "scalar_operator does not have a valid generator function.");
  }

  // FIXME: leak
  auto *stored_callback_context =
      new ScalarCallBackContext(scalar_op, paramNames);
  using WrapperFuncType =
      int32_t (*)(cudensitymatScalarCallback_t, double, int32_t, const double[],
                  cudaDataType_t, void *);

  auto wrapper = [](cudensitymatScalarCallback_t callback, double time,
                    int32_t num_params, const double params[],
                    cudaDataType_t data_type, void *scalar_storage) -> int32_t {
    try {
      ScalarCallBackContext *context =
          reinterpret_cast<ScalarCallBackContext *>(callback);
      scalar_operator &stored_op = context->scalarOp;
      if (num_params != 2 * context->paramNames.size())
        throw std::runtime_error(
            fmt::format("[Internal Error] Invalid number of callback "
                        "parameters encountered. Expected {} double params "
                        "representing {} complex values but received {}.",
                        2 * context->paramNames.size(),
                        context->paramNames.size(), num_params));

      std::unordered_map<std::string, std::complex<double>> param_map;
      for (size_t i = 0; i < context->paramNames.size(); ++i) {
        param_map[context->paramNames[i]] =
            std::complex<double>(params[2 * i], params[2 * i + 1]);
        cudaq::debug("Callback param name {}, value {}", context->paramNames[i],
                     param_map[context->paramNames[i]]);
      }

      std::complex<double> result = stored_op.evaluate(param_map);
      cudaq::debug("Scalar callback evaluated result = {}", result);
      auto *tdCoef = static_cast<std::complex<double> *>(scalar_storage);
      *tdCoef = result;
      return CUDENSITYMAT_STATUS_SUCCESS;
    } catch (const std::exception &e) {
      std::cerr << "Error in scalar callback: " << e.what() << std::endl;
      return CUDENSITYMAT_STATUS_INTERNAL_ERROR;
    }
  };

  cudensitymatWrappedScalarCallback_t wrappedCallback;
  wrappedCallback.callback =
      reinterpret_cast<cudensitymatScalarCallback_t>(stored_callback_context);
  wrappedCallback.wrapper =
      reinterpret_cast<void *>(static_cast<WrapperFuncType>(wrapper));
  return wrappedCallback;
}

cudensitymatWrappedTensorCallback_t
cudm_helper::_wrap_tensor_callback(const matrix_operator &op,
                                   const std::vector<std::string> &paramNames) {
  auto *stored_callback_context = new TensorCallBackContext(op, paramNames);

  using WrapperFuncType = int32_t (*)(
      cudensitymatTensorCallback_t, cudensitymatElementaryOperatorSparsity_t,
      int32_t, const int64_t[], const int32_t[], double, int32_t,
      const double[], cudaDataType_t, void *, cudaStream_t);

  auto wrapper = [](cudensitymatTensorCallback_t callback,
                    cudensitymatElementaryOperatorSparsity_t sparsity,
                    int32_t num_modes, const int64_t mode_extents[],
                    const int32_t diagonal_offsets[], double time,
                    int32_t num_params, const double params[],
                    cudaDataType_t data_type, void *tensor_storage,
                    cudaStream_t stream) -> int32_t {
    try {
      auto *context = reinterpret_cast<TensorCallBackContext *>(callback);
      matrix_operator &stored_op = context->tensorOp;

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

      std::unordered_map<std::string, std::complex<double>> param_map;
      for (size_t i = 0; i < context->paramNames.size(); ++i) {
        param_map[context->paramNames[i]] =
            std::complex<double>(params[2 * i], params[2 * i + 1]);
        cudaq::debug("Tensor callback param name {}, value {}",
                     context->paramNames[i], param_map[context->paramNames[i]]);
      }

      std::unordered_map<int, int> dimensions;
      for (int i = 0; i < num_modes; ++i) {
        dimensions[i] = static_cast<int>(mode_extents[i]);
      }

      if (dimensions.empty()) {
        std::cerr << "Dimension map is empty!" << std::endl;
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }

      matrix_2 matrix_data = stored_op.to_matrix(dimensions, param_map);

      std::size_t rows = matrix_data.get_rows();
      std::size_t cols = matrix_data.get_columns();

      if (rows != cols) {
        std::cerr << "Non-square matrix encountered: " << rows << "x" << cols
                  << std::endl;
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }

      std::vector<std::complex<double>> flat_matrix =
          flatten_matrix(matrix_data);

      if (data_type == CUDA_C_64F) {
        memcpy(tensor_storage, flat_matrix.data(),
               flat_matrix.size() * sizeof(cuDoubleComplex));
      } else if (data_type == CUDA_C_32F) {
        std::vector<cuFloatComplex> flat_matrix_float(flat_matrix.size());
        for (size_t i = 0; i < flat_matrix.size(); i++) {
          flat_matrix_float[i] =
              make_cuFloatComplex(static_cast<float>(flat_matrix[i].real()),
                                  static_cast<float>(flat_matrix[i].imag()));
        }
        memcpy(tensor_storage, flat_matrix_float.data(),
               flat_matrix_float.size() * sizeof(cuFloatComplex));
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

  cudensitymatWrappedTensorCallback_t wrapped_callback;
  wrapped_callback.callback =
      reinterpret_cast<cudensitymatTensorCallback_t>(stored_callback_context);
  wrapped_callback.wrapper =
      reinterpret_cast<void *>(static_cast<WrapperFuncType>(wrapper));

  return wrapped_callback;
}

// Function to flatten a matrix into a 1D array (column major)
std::vector<std::complex<double>>
cudm_helper::flatten_matrix(const matrix_2 &matrix) {
  std::vector<std::complex<double>> flat_matrix;
  flat_matrix.reserve(matrix.get_size());
  for (size_t col = 0; col < matrix.get_columns(); col++) {
    for (size_t row = 0; row < matrix.get_rows(); row++) {
      flat_matrix.push_back(matrix[{row, col}]);
    }
  }

  return flat_matrix;
}

// Function to extract sub-space extents based on degrees
std::vector<int64_t>
cudm_helper::get_subspace_extents(const std::vector<int64_t> &mode_extents,
                                  const std::vector<int> &degrees) {
  std::vector<int64_t> subspace_extents;

  for (int degree : degrees) {
    if (degree >= mode_extents.size()) {
      throw std::out_of_range("Degree exceeds mode_extents size.");
    }
    subspace_extents.push_back(mode_extents[degree]);
  }

  return subspace_extents;
}

void cudm_helper::print_complex_vector(
    const std::vector<std::complex<double>> &vec) {
  size_t n = static_cast<size_t>(std::sqrt(vec.size()));

  std::cout << "Vector contents: [\n";
  for (size_t i = 0; i < n; i++) {
    std::cout << "[";
    for (size_t j = 0; j < n; j++) {
      size_t index = i * n + j;
      std::cout << " (" << vec[index].real() << ", " << vec[index].imag()
                << "i) ";
    }
    std::cout << "]\n";
  }
}

// Function to create a cudensitymat elementary operator
// Need to use std::variant
cudensitymatElementaryOperator_t cudm_helper::create_elementary_operator(
    const cudaq::matrix_operator &elem_op,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const std::vector<int64_t> &mode_extents) {
  auto subspace_extents = get_subspace_extents(mode_extents, elem_op.degrees());
  std::unordered_map<int, int> dimensions = convert_dimensions(mode_extents);
  auto flat_matrix = flatten_matrix(elem_op.to_matrix(dimensions, parameters));

  if (flat_matrix.empty()) {
    throw std::invalid_argument("Input matrix (flat matrix) cannot be empty.");
  }

  if (subspace_extents.empty()) {
    throw std::invalid_argument("subspace_extents cannot be empty.");
  }

  cudensitymatWrappedTensorCallback_t wrapped_tensor_callback = {nullptr,
                                                                 nullptr};

  if (!parameters.empty()) {
    const std::map<std::string, std::complex<double>> sortedParameters(
        parameters.begin(), parameters.end());
    auto ks = std::views::keys(sortedParameters);
    const std::vector<std::string> keys{ks.begin(), ks.end()};

    wrapped_tensor_callback = _wrap_tensor_callback(elem_op, keys);
  }

  auto *elementaryMat_d = create_array_gpu(flat_matrix);
  cudensitymatElementaryOperator_t cudm_elem_op = nullptr;

  HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
      handle, static_cast<int32_t>(subspace_extents.size()),
      subspace_extents.data(), CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr,
      CUDA_C_64F, elementaryMat_d, wrapped_tensor_callback, &cudm_elem_op));

  if (!cudm_elem_op) {
    std::cerr << "[ERROR] cudm_elem_op is NULL in create_elementary_operator!"
              << std::endl;
    destroy_array_gpu(elementaryMat_d);
    throw std::runtime_error("Failed to create elementary operator.");
  }
  m_elementaryOperators.emplace(cudm_elem_op);
  m_deviceBuffers.emplace(elementaryMat_d);
  return cudm_elem_op;
}

// Function to append an elementary operator to a term
void cudm_helper::append_elementary_operator_to_term(
    cudensitymatOperatorTerm_t term,
    const std::vector<cudensitymatElementaryOperator_t> &elem_ops,
    const std::vector<std::vector<int>> &degrees,
    const std::vector<std::vector<int>> &all_action_dual_modalities) {

  if (degrees.empty()) {
    throw std::invalid_argument("Degrees vector cannot be empty.");
  }

  if (elem_ops.empty()) {
    throw std::invalid_argument("elem_ops cannot be null.");
  }

  if (degrees.size() != elem_ops.size()) {
    throw std::invalid_argument(
        "elem_ops and degrees must have the same size.");
  }

  bool has_dual_modalities = !all_action_dual_modalities.empty();

  if (has_dual_modalities &&
      degrees.size() != all_action_dual_modalities.size()) {
    throw std::invalid_argument(
        "degrees and all_action_dual_modalities must have the same size.");
  }

  std::vector<int32_t> allDegrees;
  std::vector<int32_t> allModeActionDuality;
  for (size_t i = 0; i < degrees.size(); i++) {
    const auto &sub_degrees = degrees[i];
    const auto &modalities = has_dual_modalities
                                 ? all_action_dual_modalities[i]
                                 : std::vector<int>(sub_degrees.size(), 0);

    if (sub_degrees.size() != modalities.size()) {
      throw std::runtime_error(
          "Mismatch between degrees and modalities sizes.");
    }
    if (sub_degrees.size() != 1) {
      throw std::runtime_error(
          "Elementary operator must act on a single degree.");
    }

    for (size_t j = 0; j < sub_degrees.size(); j++) {
      int degree = sub_degrees[j];
      int modality = modalities[j];

      if (sub_degrees[i] < 0) {
        throw std::out_of_range("Degree cannot be negative!");
      }
      allDegrees.emplace_back(degree);
      allModeActionDuality.emplace_back(modality);
    }
  }

  assert(elem_ops.size() == degrees.size());
  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
      handle, term, static_cast<int32_t>(elem_ops.size()), elem_ops.data(),
      allDegrees.data(), allModeActionDuality.data(),
      make_cuDoubleComplex(1.0, 0.0), {nullptr, nullptr}));
}

void cudm_helper::scale_state(cudensitymatState_t state, double scale_factor,
                              cudaStream_t stream) {
  if (!state) {
    throw std::invalid_argument("Invalid state provided to scale_state.");
  }

  HANDLE_CUDM_ERROR(
      cudensitymatStateComputeScaling(handle, state, &scale_factor, stream));

  HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
}

std::pair<cudensitymatOperatorTerm_t, cudensitymatOperatorTerm_t>
cudm_helper::compute_lindblad_operator_terms(
    operator_sum<cudaq::matrix_operator> &collapseOp,
    const std::vector<int64_t> &mode_extents,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  std::unordered_map<int, int> dimensions;
  for (int i = 0; i < mode_extents.size(); ++i)
    dimensions[i] = mode_extents[i];
  auto c_op = collapseOp.to_matrix(dimensions);
  auto degrees = collapseOp.degrees();
  auto adjointMat = matrix_2::adjoint(c_op);
  cudensitymatElementaryOperator_t LOp, LOpDagger, LdaggerLOp;
  auto *LOp_d = create_array_gpu(flatten_matrix(c_op));
  auto *LOpDagger_d = create_array_gpu(flatten_matrix(adjointMat));
  auto *LdaggerL_d = create_array_gpu(flatten_matrix(adjointMat * c_op));
  m_deviceBuffers.emplace(LOp_d);
  m_deviceBuffers.emplace(LOpDagger_d);
  m_deviceBuffers.emplace(LdaggerL_d);
  auto subspace_extents = get_subspace_extents(mode_extents, degrees);
  HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
      handle, subspace_extents.size(), subspace_extents.data(),
      CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr, CUDA_C_64F, LOp_d,
      {nullptr, nullptr}, &LOp));

  HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
      handle, subspace_extents.size(), subspace_extents.data(),
      CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr, CUDA_C_64F, LOpDagger_d,
      {nullptr, nullptr}, &LOpDagger));

  HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
      handle, subspace_extents.size(), subspace_extents.data(),
      CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr, CUDA_C_64F, LdaggerL_d,
      {nullptr, nullptr}, &LdaggerLOp));
  m_elementaryOperators.emplace(LOp);
  m_elementaryOperators.emplace(LOpDagger);
  m_elementaryOperators.emplace(LdaggerLOp);

  cudensitymatOperatorTerm_t D1_term;
  //  Create an empty operator term
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
      handle,
      mode_extents.size(), // Hilbert space rank (number of dimensions)
      mode_extents.data(), // Hilbert space shape
      &D1_term));          // the created empty operator term
  m_operatorTerms.emplace(D1_term);
  //  Define the operator term
  std::vector<int32_t> d1Degree; // stacked degrees
  d1Degree.insert(d1Degree.end(), degrees.begin(), degrees.end());
  d1Degree.insert(d1Degree.end(), degrees.begin(), degrees.end());
  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
      handle, D1_term,
      2, // number of elementary tensor operators in the product
      std::vector<cudensitymatElementaryOperator_t>({LOp, LOpDagger})
          .data(),     // elementary tensor operators forming the product
      d1Degree.data(), // space modes acted on by the operator product (from
                       // different sides)
      std::vector<int32_t>({0, 1}).data(), // space mode action duality (0: from
                                           // the left; 1: from the right)
      make_cuDoubleComplex(1.0, 0.0),      // default coefficient: Always
                                           // 64-bit-precision complex number
      {nullptr, nullptr})); // no time-dependent coefficient associated with
                            // the operator product

  cudensitymatOperatorTerm_t D2_term;
  //  Create an empty operator term
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
      handle,
      mode_extents.size(), // Hilbert space rank (number of dimensions)
      mode_extents.data(), // Hilbert space shape
      &D2_term));          // the created empty operator term
  m_operatorTerms.emplace(D2_term);
  //  Define the operator term
  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
      handle, D2_term,
      1, // number of elementary tensor operators in the product
      std::vector<cudensitymatElementaryOperator_t>({LdaggerLOp})
          .data(),    // elementary tensor operators forming the product
      degrees.data(), // space modes acted on by the operator
                      // product (from different sides)
      std::vector<int32_t>({0}).data(), // space mode action duality (0:
                                        // from the left; 1: from the right)
      make_cuDoubleComplex(-0.5, 0.0),  // default coefficient: Always
                                        // 64-bit-precision complex number
      {nullptr, nullptr})); // no time-dependent coefficient associated with
                            // the operator product

  return std::make_pair(D1_term, D2_term);
}

// TODO: fix the signature
// c_ops: std::vector<operator_sum>
cudensitymatOperator_t cudm_helper::compute_lindblad_operator(
    const std::vector<matrix_2> &c_ops,
    const std::vector<int64_t> &mode_extents) {
  if (c_ops.empty()) {
    throw std::invalid_argument("Collapse operators cannot be empty.");
  }

  cudensitymatOperator_t liouvillian;

  //  Create an empty operator (super-operator)
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
      handle,
      mode_extents.size(), // Hilbert space rank (number of dimensions)
      mode_extents.data(), // Hilbert space shape
      &liouvillian));      // the created empty operator (super-operator)

  for (auto &c_op : c_ops) {

    cudensitymatElementaryOperator_t LOp, LOpDagger, LdaggerLOp;
    auto adjointMat = matrix_2::adjoint(c_op);
    auto *LOp_d = create_array_gpu(flatten_matrix(c_op));
    auto *LOpDagger_d = create_array_gpu(flatten_matrix(adjointMat));
    auto *LdaggerL_d = create_array_gpu(flatten_matrix(adjointMat * c_op));
    m_deviceBuffers.emplace(LOp_d);
    m_deviceBuffers.emplace(LOpDagger_d);
    m_deviceBuffers.emplace(LdaggerL_d);
    // FIXME: assume degree 0 as we don't have that info here
    auto subspace_extents = get_subspace_extents(mode_extents, {0});
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
        handle, 1, subspace_extents.data(), CUDENSITYMAT_OPERATOR_SPARSITY_NONE,
        0, nullptr, CUDA_C_64F, LOp_d, {nullptr, nullptr}, &LOp));

    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
        handle, 1, subspace_extents.data(), CUDENSITYMAT_OPERATOR_SPARSITY_NONE,
        0, nullptr, CUDA_C_64F, LOpDagger_d, {nullptr, nullptr}, &LOpDagger));

    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
        handle, 1, subspace_extents.data(), CUDENSITYMAT_OPERATOR_SPARSITY_NONE,
        0, nullptr, CUDA_C_64F, LdaggerL_d, {nullptr, nullptr}, &LdaggerLOp));
    m_elementaryOperators.emplace(LOp);
    m_elementaryOperators.emplace(LOpDagger);
    m_elementaryOperators.emplace(LdaggerLOp);
    {
      cudensitymatOperatorTerm_t D1_term;
      //  Create an empty operator term
      HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
          handle,
          mode_extents.size(), // Hilbert space rank (number of dimensions)
          mode_extents.data(), // Hilbert space shape
          &D1_term));          // the created empty operator term
      m_operatorTerms.emplace(D1_term);
      //  Define the operator term
      HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
          handle, D1_term,
          2, // number of elementary tensor operators in the product
          std::vector<cudensitymatElementaryOperator_t>({LOp, LOpDagger})
              .data(), // elementary tensor operators forming the product
          std::vector<int32_t>({0, 0})
              .data(), // space modes acted on by the operator product (from
                       // different sides)
          std::vector<int32_t>({0, 1})
              .data(),                    // space mode action duality (0: from
                                          // the left; 1: from the right)
          make_cuDoubleComplex(1.0, 0.0), // default coefficient: Always
                                          // 64-bit-precision complex number
          {nullptr, nullptr})); // no time-dependent coefficient associated with
                                // the operator product

      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, liouvillian,
          D1_term, // appended operator term
          0, // operator term action duality as a whole (no duality reversing in
             // this case)
          make_cuDoubleComplex(1, 0.0), // constant coefficient associated with
                                        // the operator term as a whole
          {nullptr,
           nullptr})); // no time-dependent coefficient associated with the
    }
    {
      cudensitymatOperatorTerm_t D2_term;
      //  Create an empty operator term
      HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
          handle,
          mode_extents.size(), // Hilbert space rank (number of dimensions)
          mode_extents.data(), // Hilbert space shape
          &D2_term));          // the created empty operator term
      m_operatorTerms.emplace(D2_term);
      //  Define the operator term
      HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
          handle, D2_term,
          1, // number of elementary tensor operators in the product
          std::vector<cudensitymatElementaryOperator_t>({LdaggerLOp})
              .data(), // elementary tensor operators forming the product
          std::vector<int32_t>({0})
              .data(), // space modes acted on by the operator product (from
                       // different sides)
          std::vector<int32_t>({0}).data(), // space mode action duality (0:
                                            // from the left; 1: from the right)
          make_cuDoubleComplex(-0.5, 0.0),  // default coefficient: Always
                                            // 64-bit-precision complex number
          {nullptr, nullptr})); // no time-dependent coefficient associated with
                                // the operator product

      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, liouvillian,
          D2_term, // appended operator term
          0, // operator term action duality as a whole (no duality reversing in
             // this case)
          make_cuDoubleComplex(1, 0.0), // constant coefficient associated with
                                        // the operator term as a whole
          {nullptr,
           nullptr})); // no time-dependent coefficient associated with the

      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, liouvillian,
          D2_term, // appended operator term
          1, // operator term action duality as a whole (no duality reversing in
             // this case)
          make_cuDoubleComplex(1, 0.0), // constant coefficient associated with
                                        // the operator term as a whole
          {nullptr,
           nullptr})); // no time-dependent coefficient associated with the
    }
  }
  // operator term as a whole
  std::cout << "Constructed the Liouvillian operator\n";
  return liouvillian;
}

std::unordered_map<int, int>
cudm_helper::convert_dimensions(const std::vector<int64_t> &mode_extents) {

  std::unordered_map<int, int> dimensions;
  for (size_t i = 0; i < mode_extents.size(); i++) {
    dimensions[static_cast<int>(i)] = static_cast<int>(mode_extents[i]);
  }

  return dimensions;
}

std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
cudm_helper::convert_to_cudensitymat(
    const operator_sum<cudaq::matrix_operator> &op,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const std::vector<int64_t> &mode_extents) {
  if (op.get_terms().empty()) {
    throw std::invalid_argument("Operator sum cannot be empty.");
  }

  std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
      result;

  for (const auto &product_op : op.get_terms()) {
    cudensitymatOperatorTerm_t term;
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
        handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
        &term));
    m_operatorTerms.emplace(term);
    std::vector<cudensitymatElementaryOperator_t> elem_ops;
    std::vector<std::vector<int>> all_degrees;
    for (const auto &component : product_op.get_terms()) {
      // No need to check type
      // just call to_matrix on it
      if (const auto *elem_op =
              dynamic_cast<const cudaq::matrix_operator *>(&component)) {
        auto cudm_elem_op =
            create_elementary_operator(*elem_op, parameters, mode_extents);
        elem_ops.emplace_back(cudm_elem_op);
        all_degrees.emplace_back(elem_op->degrees());
      } else {
        // Catch anything that we don't know
        throw std::runtime_error("Unhandled type!");
      }
    }
    append_elementary_operator_to_term(term, elem_ops, all_degrees, {});
    result.emplace_back(std::make_pair(product_op.get_coefficient(), term));
  }
  return result;
}

template <typename HandlerTy>
cudensitymatOperator_t cudm_helper::convert_to_cudensitymat_operator(
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const operator_sum<HandlerTy> &op,
    const std::vector<int64_t> &mode_extents) {
  if (op.get_terms().empty()) {
    throw std::invalid_argument("Operator sum cannot be empty.");
  }

  cudensitymatOperator_t operator_handle;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
      handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
      &operator_handle));

  const std::map<std::string, std::complex<double>> sortedParameters(
      parameters.begin(), parameters.end());
  auto ks = std::views::keys(sortedParameters);
  const std::vector<std::string> keys{ks.begin(), ks.end()};
  for (auto &[coeff, term] :
       convert_to_cudensitymat(op, parameters, mode_extents)) {
    cudensitymatWrappedScalarCallback_t wrapped_callback = {nullptr, nullptr};

    if (coeff.is_constant()) {
      const auto coeffVal = coeff.evaluate();
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, operator_handle, term, 0,
          make_cuDoubleComplex(coeffVal.real(), coeffVal.imag()),
          wrapped_callback));
    } else {
      wrapped_callback = _wrap_callback(coeff, keys);
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, operator_handle, term, 0, make_cuDoubleComplex(1.0, 0.0),
          wrapped_callback));
    }
  }

  return operator_handle;
}

cudensitymatOperator_t cudm_helper::construct_liouvillian(
    const operator_sum<cudaq::matrix_operator> &op,
    const std::vector<operator_sum<cudaq::matrix_operator> *>
        &collapse_operators,
    const std::vector<int64_t> &mode_extents,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool is_master_equation) {
  if (!is_master_equation && collapse_operators.empty()) {
    cudaq::info("Construct state vector Liouvillian");
    auto liouvillian = op * std::complex<double>(0.0, -1.0);
    return convert_to_cudensitymat_operator(parameters, liouvillian,
                                            mode_extents);
  } else {
    cudaq::info("Construct density matrix Liouvillian");
    cudensitymatOperator_t liouvillian;
    HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
        handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
        &liouvillian));
    // Append an operator term to the operator (super-operator)
    // Handle the Hamiltonian
    const std::map<std::string, std::complex<double>> sortedParameters(
        parameters.begin(), parameters.end());
    auto ks = std::views::keys(sortedParameters);
    const std::vector<std::string> keys{ks.begin(), ks.end()};
    for (auto &[coeff, term] :
         convert_to_cudensitymat(op, parameters, mode_extents)) {
      cudensitymatWrappedScalarCallback_t wrapped_callback = {nullptr, nullptr};
      if (coeff.is_constant()) {
        const auto coeffVal = coeff.evaluate();
        const auto leftCoeff = std::complex<double>(0.0, -1.0) * coeffVal;
        // -i constant (left multiplication)
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            handle, liouvillian, term, 0,
            make_cuDoubleComplex(leftCoeff.real(), leftCoeff.imag()),
            wrapped_callback));

        // +i constant (right multiplication, i.e., dual)
        const auto rightCoeff = std::complex<double>(0.0, 1.0) * coeffVal;
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            handle, liouvillian, term, 1,
            make_cuDoubleComplex(rightCoeff.real(), rightCoeff.imag()),
            wrapped_callback));
      } else {
        wrapped_callback = _wrap_callback(coeff, keys);
        // -i constant (left multiplication)
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            handle, liouvillian, term, 0, make_cuDoubleComplex(0.0, -1.0),
            wrapped_callback));

        // +i constant (right multiplication, i.e., dual)
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            handle, liouvillian, term, 1, make_cuDoubleComplex(0.0, 1.0),
            wrapped_callback));
      }
    }

    // Handle collapsed operators
    for (auto &collapse_operators : collapse_operators) {
      auto [d1Term, d2Term] = compute_lindblad_operator_terms(
          *collapse_operators, mode_extents, parameters);

      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, liouvillian,
          d1Term, // appended operator term
          0, // operator term action duality as a whole (no duality reversing in
             // this case)
          make_cuDoubleComplex(1, 0.0), // constant coefficient associated with
                                        // the operator term as a whole
          {nullptr,
           nullptr})); // no time-dependent coefficient associated with the

      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, liouvillian,
          d2Term, // appended operator term
          0, // operator term action duality as a whole (no duality reversing in
             // this case)
          make_cuDoubleComplex(1, 0.0), // constant coefficient associated with
                                        // the operator term as a whole
          {nullptr,
           nullptr})); // no time-dependent coefficient associated with the

      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, liouvillian,
          d2Term, // appended operator term
          1, // operator term action duality as a whole (no duality reversing in
             // this case)
          make_cuDoubleComplex(1, 0.0), // constant coefficient associated with
                                        // the operator term as a whole
          {nullptr,
           nullptr})); // no time-dependent coefficient associated with the
    }

    return liouvillian;
  }
}

cudensitymatOperator_t cudm_helper::construct_liouvillian(
    const cudensitymatOperator_t &hamiltonian,
    const std::vector<cudensitymatOperator_t> &collapse_operators,
    double gamma) {
  try {
    cudensitymatOperator_t liouvillian;
    HANDLE_CUDM_ERROR(
        cudensitymatCreateOperator(handle, 0, nullptr, &liouvillian));

    cudensitymatWrappedScalarCallback_t scalarCallback = {nullptr, nullptr};
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
        handle, liouvillian, hamiltonian, 0, {1.0, 0.0}, scalarCallback));

    // Collapse operator scaled by gamma
    cuDoubleComplex coefficient = make_cuDoubleComplex(gamma, 0.0);
    for (const auto &c_op : collapse_operators) {
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, liouvillian, c_op, 0, coefficient, scalarCallback));
    }

    return liouvillian;
  } catch (const std::exception &e) {
    std::cerr << "Error in construct_liouvillian: " << e.what() << std::endl;
    throw;
  }
}

// Function for creating an array copy in GPU memory
void *cudm_helper::create_array_gpu(
    const std::vector<std::complex<double>> &cpu_array) {
  void *gpu_array{nullptr};
  const std::size_t array_size =
      cpu_array.size() * sizeof(std::complex<double>);
  if (array_size > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&gpu_array, array_size));
    HANDLE_CUDA_ERROR(cudaMemcpy(gpu_array,
                                 static_cast<const void *>(cpu_array.data()),
                                 array_size, cudaMemcpyHostToDevice));
  }
  return gpu_array;
}

// Function to detsroy a previously created array copy in GPU memory
void cudm_helper::destroy_array_gpu(void *gpu_array) {
  if (gpu_array) {
    HANDLE_CUDA_ERROR(cudaFree(gpu_array));
  }
}

template cudensitymatOperator_t
cudm_helper::convert_to_cudensitymat_operator<cudaq::matrix_operator>(
    const std::unordered_map<std::string, std::complex<double>> &,
    const operator_sum<cudaq::matrix_operator> &, const std::vector<int64_t> &);

} // namespace cudaq
