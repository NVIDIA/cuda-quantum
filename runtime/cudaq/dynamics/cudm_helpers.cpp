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
cudensitymatState_t initialize_state(cudensitymatHandle_t handle,
                                     cudensitymatStatePurity_t purity,
                                     const std::vector<int64_t> &mode_extents) {
  cudensitymatState_t state;
  cudensitymatStatus_t status =
      cudensitymatCreateState(handle, purity, mode_extents.size(),
                              mode_extents.data(), 1, CUDA_C_64F, &state);
  if (status != CUDENSITYMAT_STATUS_SUCCESS) {
    std::cerr << "Error in cudensitymatCreateState: " << status << std::endl;
  }
  return state;
}

void scale_state(cudensitymatHandle_t handle, cudensitymatState_t state,
                 double scale_factor, cudaStream_t stream) {
  if (!state) {
    throw std::invalid_argument("Invalid state provided to scale_state.");
  }

  HANDLE_CUDM_ERROR(
      cudensitymatStateComputeScaling(handle, state, &scale_factor, stream));
}

void destroy_state(cudensitymatState_t state) {
  cudensitymatDestroyState(state);
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
      throw std::invalid_argument("Collapse operator must be a square matrix.");
    }

    std::vector<std::complex<double>> flat_matrix(dim * dim);
    for (size_t i = 0; i < dim; i++) {
      for (size_t j = 0; j < dim; j++) {
        flat_matrix[i * dim + j] = c_op[{i, j}];
      }
    }

    // Create Operator term for LtL and add to lindblad_op
    cudensitymatOperatorTerm_t term;
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
        handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
        &term));
    cudensitymatDestroyOperator(lindblad_op);

    // Attach terms and cleanup
    cudensitymatWrappedScalarCallback_t scalarCallback = {nullptr, nullptr};
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle, lindblad_op, term,
                                                     0, {1.0}, scalarCallback));
    cudensitymatDestroyOperatorTerm(term);
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

cudensitymatOperator_t convert_to_cudensitymat_operator(
    cudensitymatHandle_t handle,
    const std::map<std::string, std::complex<double>> &parameters,
    const operator_sum &op, const std::vector<int64_t> &mode_extents) {
  try {
    cudensitymatOperator_t operator_handle;
    auto status = cudensitymatCreateOperator(
        handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
        &operator_handle);
    if (status != CUDENSITYMAT_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create operator.");
    }

    for (const auto &product_op : op.get_terms()) {
      cudensitymatOperatorTerm_t term;

      HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
          handle, static_cast<int32_t>(mode_extents.size()),
          mode_extents.data(), &term));

      for (const auto &component : product_op.get_terms()) {
        if (std::holds_alternative<cudaq::elementary_operator>(component)) {
          const auto &elem_op = std::get<cudaq::elementary_operator>(component);

          // Create a cudensitymat elementary operator
          cudensitymatElementaryOperator_t cudm_elem_op;

          // Get the matrix representation of elementary operator
          auto dimensions = convert_dimensions(mode_extents);
          auto matrix = elem_op.to_matrix(dimensions, parameters);

          // Flatten the matrix into a single-dimensional array
          std::vector<std::complex<double>> flat_matrix;
          for (size_t i = 0; i < matrix.get_rows(); i++) {
            for (size_t j = 0; j < matrix.get_columns(); j++) {
              flat_matrix.push_back(matrix[{i, j}]);
            }
          }

          // Create a cudensitymat elementary operator
          HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
              handle, 1, mode_extents.data(),
              CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr, CUDA_C_64F,
              flat_matrix.data(), {nullptr, nullptr}, &cudm_elem_op));

          // Append the elementary operator to the term
          HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
              handle, term, 1, &cudm_elem_op, &elem_op.degrees[0], nullptr,
              make_cuDoubleComplex(1.0, 0.0), {nullptr, nullptr}));

          // Destroy the elementary operator after appending
          HANDLE_CUDM_ERROR(
              cudensitymatDestroyElementaryOperator(cudm_elem_op));
        } else if (std::holds_alternative<cudaq::scalar_operator>(component)) {
          const auto &scalar_op = std::get<cudaq::scalar_operator>(component);

          // Use the scalar coefficient
          auto coeff = scalar_op.evaluate(parameters);
          HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
              handle, term, 0, nullptr, nullptr, nullptr,
              {make_cuDoubleComplex(coeff.real(), coeff.imag())},
              {nullptr, nullptr}));
        }
      }

      // Append the product operator term to the top-level operator
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, operator_handle, term, 0, make_cuDoubleComplex(1.0, 0.0),
          {nullptr, nullptr}));

      // Destroy the term
      HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(term));
    }

    return operator_handle;
  } catch (const std::exception &e) {
    std::cerr << "Error in convert_to_cudensitymat_operator: " << e.what()
              << std::endl;
    throw;
  }
}

cudensitymatOperator_t construct_liovillian(
    cudensitymatHandle_t handle, const cudensitymatOperator_t &hamiltonian,
    const std::vector<cudensitymatOperator_t> &collapse_operators,
    double gamma) {
  try {
    cudensitymatOperator_t liouvillian;
    auto status = cudensitymatCreateOperator(handle, 0, nullptr, &liouvillian);
    if (status != CUDENSITYMAT_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create Liouvillian operator.");
    }

    cudensitymatWrappedScalarCallback_t scalarCallback = {nullptr, nullptr};
    status = cudensitymatOperatorAppendTerm(handle, liouvillian, hamiltonian, 0,
                                            {1.0, 0.0}, scalarCallback);
    if (status != CUDENSITYMAT_STATUS_SUCCESS) {
      cudensitymatDestroyOperator(liouvillian);
      throw std::runtime_error("Failed to add hamiltonian term.");
    }

    cuDoubleComplex coefficient = make_cuDoubleComplex(gamma, 0.0);
    for (const auto &c_op : collapse_operators) {
      status = cudensitymatOperatorAppendTerm(handle, liouvillian, c_op, 0,
                                              coefficient, scalarCallback);
      if (status != CUDENSITYMAT_STATUS_SUCCESS) {
        cudensitymatDestroyOperator(liouvillian);
        throw std::runtime_error("Failed to add collapse operator term.");
      }
    }

    return liouvillian;
  } catch (const std::exception &e) {
    std::cerr << "Error in construct_liovillian: " << e.what() << std::endl;
    throw;
  }
}
} // namespace cudaq
