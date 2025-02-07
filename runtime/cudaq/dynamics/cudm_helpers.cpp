/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/cudm_helpers.h"
#include "cudaq/cudm_error_handling.h"

namespace cudaq {
// Function to flatten a matrix into a 1D array (column major)
std::vector<std::complex<double>> flatten_matrix(const matrix_2 &matrix) {
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
get_subspace_extents(const std::vector<int64_t> &mode_extents,
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

// Function to create a cudensitymat elementary operator
cudensitymatElementaryOperator_t create_elementary_operator(
    cudensitymatHandle_t handle, const std::vector<int64_t> &subspace_extents,
    const std::vector<std::complex<double>> &flat_matrix) {
  if (flat_matrix.empty()) {
    throw std::invalid_argument("Input matrix (flat matrix) cannot be empty.");
  }

  if (subspace_extents.empty()) {
    throw std::invalid_argument("subspace_extents cannot be empty.");
  }

  cudensitymatElementaryOperator_t cudm_elem_op = nullptr;

  // FIXME: leak (need to track this buffer somewhere and delete **after** the
  // whole evolve)
  auto *elementaryMat_d = create_array_gpu(flat_matrix);

  cudensitymatStatus_t status = cudensitymatCreateElementaryOperator(
      handle, static_cast<int32_t>(subspace_extents.size()),
      subspace_extents.data(), CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr,
      CUDA_C_64F, elementaryMat_d, {nullptr, nullptr}, &cudm_elem_op);

  if (status != CUDENSITYMAT_STATUS_SUCCESS) {
    std::cerr << "Error: Failed to create elementary operator. Status: "
              << status << std::endl;
    return nullptr;
  }

  return cudm_elem_op;
}

// Function to append an elementary operator to a term
void append_elementary_operator_to_term(
    cudensitymatHandle_t handle, cudensitymatOperatorTerm_t term,
    const cudensitymatElementaryOperator_t &elem_op,
    const std::vector<int> &degrees) {
  if (degrees.empty()) {
    throw std::invalid_argument("Degrees vector cannot be empty.");
  }

  if (!elem_op) {
    throw std::invalid_argument("elem_op cannot be null.");
  }

  std::vector<cudensitymatElementaryOperator_t> elem_ops = {elem_op};

  std::vector<int32_t> modeActionDuality(degrees.size(), 0);
  assert(elem_ops.size() == degrees.size());
  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
      handle, term, static_cast<int32_t>(degrees.size()), elem_ops.data(),
      degrees.data(), modeActionDuality.data(), make_cuDoubleComplex(1.0, 0.0),
      {nullptr, nullptr}));
}

// Function to create and append a scalar to a term
void append_scalar_to_term(cudensitymatHandle_t handle,
                           cudensitymatOperatorTerm_t term,
                           const std::complex<double> &coeff) {
  // TODO: Implement handling for time-dependent scalars using
  // cudensitymatScalarCallback_t
  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
      handle, term, 0, nullptr, nullptr, nullptr,
      {make_cuDoubleComplex(coeff.real(), coeff.imag())}, {nullptr, nullptr}));
}

void scale_state(cudensitymatHandle_t handle, cudensitymatState_t state,
                 double scale_factor, cudaStream_t stream) {
  if (!state) {
    throw std::invalid_argument("Invalid state provided to scale_state.");
  }

  HANDLE_CUDM_ERROR(
      cudensitymatStateComputeScaling(handle, state, &scale_factor, stream));
}

cudensitymatOperator_t
compute_lindblad_operator(cudensitymatHandle_t handle,
                          const std::vector<matrix_2> &c_ops,
                          const std::vector<int64_t> &mode_extents) {
  if (c_ops.empty()) {
    throw std::invalid_argument("Collapse operators cannot be empty.");
  }

  cudensitymatOperator_t lindblad_op;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
      handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
      &lindblad_op));

  for (const auto &c_op : c_ops) {
    size_t dim = c_op.get_rows();
    if (dim == 0 || c_op.get_columns() != dim) {
      throw std::invalid_argument("Collapse operator must be a square matrix");
    }

    auto flat_matrix = flatten_matrix(c_op);

    // Create Operator term for LtL and add to lindblad_op
    cudensitymatOperatorTerm_t term;
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
        handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
        &term));

    // Create elementary operator from c_op
    cudensitymatElementaryOperator_t cudm_elem_op =
        create_elementary_operator(handle, mode_extents, flat_matrix);

    if (!cudm_elem_op) {
      throw std::runtime_error(
          "Failed to create elementary operator in compute_lindblad_operator.");
    }

    // Append the elementary operator to the term
    std::vector<int> degrees = {0, 1};
    append_elementary_operator_to_term(handle, term, cudm_elem_op, degrees);

    // Add term to lindblad operator
    cudensitymatWrappedScalarCallback_t scalarCallback = {nullptr, nullptr};
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle, lindblad_op, term,
                                                     0, {1.0}, scalarCallback));

    // Destroy intermediate resources
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(term));
    HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(cudm_elem_op));
  }

  return lindblad_op;
}

std::map<int, int>
convert_dimensions(const std::vector<int64_t> &mode_extents) {
  std::map<int, int> dimensions;
  for (size_t i = 0; i < mode_extents.size(); i++) {
    dimensions[static_cast<int>(i)] = static_cast<int>(mode_extents[i]);
  }
  return dimensions;
}

template <typename HandlerTy>
cudensitymatOperator_t convert_to_cudensitymat_operator(
    cudensitymatHandle_t handle,
    const std::map<std::string, std::complex<double>> &parameters,
    const operator_sum<HandlerTy> &op,
    const std::vector<int64_t> &mode_extents) {
  if (op.get_terms().empty()) {
    throw std::invalid_argument("Operator sum cannot be empty.");
  }

  try {
    cudensitymatOperator_t operator_handle;
    HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
        handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
        &operator_handle));

    std::vector<cudensitymatElementaryOperator_t> elementary_operators;

    for (const auto &product_op : op.get_terms()) {
      cudensitymatOperatorTerm_t term;

      HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
          handle, static_cast<int32_t>(mode_extents.size()),
          mode_extents.data(), &term));

      for (const auto &component : product_op.get_terms()) {
        if (const auto *elem_op =
                dynamic_cast<const cudaq::matrix_operator *>(&component)) {
          auto subspace_extents =
              get_subspace_extents(mode_extents, elem_op->degrees);
          auto flat_matrix = flatten_matrix(
              elem_op->to_matrix(convert_dimensions(mode_extents), parameters));
          auto cudm_elem_op =
              create_elementary_operator(handle, subspace_extents, flat_matrix);

          elementary_operators.push_back(cudm_elem_op);
          append_elementary_operator_to_term(handle, term, cudm_elem_op,
                                             elem_op->degrees);
        } else if (const auto *scalar_op =
                       dynamic_cast<const cudaq::scalar_operator *>(
                           &component)) {
          auto coeff = scalar_op->evaluate(parameters);
          append_scalar_to_term(handle, term, coeff);
        }
      }

      // Append the product operator term to the top-level operator
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, operator_handle, term, 0, make_cuDoubleComplex(1.0, 0.0),
          {nullptr, nullptr}));

      // FIXME: leak
      // We must track these handles and destroy **after** evolve finishes
      // Destroy the term
      // HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(term));

      // // Cleanup
      // for (auto &elem_op : elementary_operators) {
      //   HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(elem_op));
      // }
    }

    return operator_handle;
  } catch (const std::exception &e) {
    std::cerr << "Error in convert_to_cudensitymat_operator: " << e.what()
              << std::endl;
    throw;
  }
}

cudensitymatOperator_t construct_liouvillian(
    cudensitymatHandle_t handle, const cudensitymatOperator_t &hamiltonian,
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
void *create_array_gpu(const std::vector<std::complex<double>> &cpu_array) {
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
void destroy_array_gpu(void *gpu_array) {
  if (gpu_array) {
    HANDLE_CUDA_ERROR(cudaFree(gpu_array));
  }
}

template cudensitymatOperator_t
convert_to_cudensitymat_operator<cudaq::matrix_operator>(
    cudensitymatHandle_t, const std::map<std::string, std::complex<double>> &,
    const operator_sum<cudaq::matrix_operator> &, const std::vector<int64_t> &);

} // namespace cudaq
