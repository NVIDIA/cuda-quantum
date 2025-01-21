/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/cudm_helpers.h"

namespace cudaq {
cudensitymatState_t initialize_state(cudensitymatHandle_t handle,
                                     cudensitymatStatePurity_t purity,
                                     int num_modes,
                                     const std::vector<int64_t> &mode_extents) {
  try {
    cudensitymatState_t state;
    auto status = cudensitymatCreateState(
        handle, purity, num_modes, mode_extents.data(), 1, CUDA_R_64F, &state);
    if (status != CUDENSITYMAT_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to initialize quantum state.");
    }
    return state;
  } catch (const std::exception &e) {
    std::cerr << "Error in initialize_state: " << e.what() << std::endl;
    throw;
  }
}

void scale_state(cudensitymatHandle_t handle, cudensitymatState_t state,
                 double scale_factor, cudaStream_t stream) {
  try {
    auto status =
        cudensitymatStateComputeScaling(handle, state, &scale_factor, stream);
    if (status != CUDENSITYMAT_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to scale quantum state.");
    }
  } catch (const std::exception &e) {
    std::cerr << "Error in scale_state: " << e.what() << std::endl;
    throw;
  }
}

void destroy_state(cudensitymatState_t state) {
  try {
    cudensitymatDestroyState(state);
  } catch (const std::exception &e) {
    std::cerr << "Error in destroy_state: " << e.what() << std::endl;
  }
}

cudensitymatOperator_t
compute_lindblad_operator(cudensitymatHandle_t handle,
                          const std::vector<matrix_2> &c_ops,
                          const std::vector<int64_t> &mode_extents) {
  try {
    if (c_ops.empty()) {
      throw std::invalid_argument("Collapse operators cannot be empty.");
    }

    cudensitymatOperator_t lindblad_op;
    auto status = cudensitymatCreateOperator(
        handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
        &lindblad_op);
    if (status != CUDENSITYMAT_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create lindblad operator.");
    }

    for (const auto &c_op : c_ops) {
      size_t dim = c_op.get_rows();
      if (dim == 0 || c_op.get_columns() != dim) {
        throw std::invalid_argument(
            "Collapse operator must be a square matrix.");
      }

      std::vector<std::complex<double>> flat_matrix(dim * dim);
      for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
          flat_matrix[i * dim + j] = c_op[{i, j}];
        }
      }

      // Create Operator term for LtL and add to lindblad_op
      cudensitymatOperatorTerm_t term;
      status = cudensitymatCreateOperatorTerm(
          handle, static_cast<int32_t>(mode_extents.size()),
          mode_extents.data(), &term);
      if (status != CUDENSITYMAT_STATUS_SUCCESS) {
        cudensitymatDestroyOperator(lindblad_op);
        throw std::runtime_error("Failed to create operator term.");
      }

      // Attach terms and cleanup
      cudensitymatWrappedScalarCallback_t scalarCallback = {nullptr, nullptr};
      status = cudensitymatOperatorAppendTerm(handle, lindblad_op, term, 0,
                                              {1.0}, scalarCallback);
      if (status != CUDENSITYMAT_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to append operator term.");
      }

      cudensitymatDestroyOperatorTerm(term);
    }

    return lindblad_op;
  } catch (const std::exception &e) {
    std::cerr << "Error in compute_lindblad_op: " << e.what() << std::endl;
    throw;
  }
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

    // Define dimensions for the operator
    std::map<int, int> dimensions;
    for (size_t i = 0; i < mode_extents.size(); i++) {
      dimensions[static_cast<int>(i)] = static_cast<int>(mode_extents[i]);
    }

    auto matrix = op.to_matrix(dimensions, parameters);
    size_t dim = matrix.get_rows();
    if (matrix.get_columns() != dim) {
      throw std::invalid_argument("Matrix must be a square.");
    }

    std::vector<std::complex<double>> flat_matrix;
    for (size_t i = 0; i < matrix.get_rows(); i++) {
      for (size_t j = 0; j < matrix.get_columns(); j++) {
        flat_matrix.push_back(matrix[{i, j}]);
      }
    }

    cudensitymatOperatorTerm_t term;
    status = cudensitymatCreateOperatorTerm(
        handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
        &term);
    if (status != CUDENSITYMAT_STATUS_SUCCESS) {
      cudensitymatDestroyOperator(operator_handle);
      throw std::runtime_error("Failed to create operator term.");
    }

    // Attach flat_matrix to the term
    int32_t num_elem_operators = 1;
    int32_t num_operator_modes = static_cast<int32_t>(mode_extents.size());
    const int64_t *operator_mode_extents = mode_extents.data();
    const int64_t *operator_mode_strides = nullptr;
    int32_t state_modes_acted_on[static_cast<int32_t>(mode_extents.size())];
    for (int32_t i = 0; i < num_operator_modes; i++) {
      state_modes_acted_on[i] = i;
    }

    cudensitymatWrappedTensorCallback_t tensorCallback = {nullptr, nullptr};
    cudensitymatWrappedScalarCallback_t scalarCallback = {nullptr, nullptr};

    void *tensor_data = flat_matrix.data();
    cuDoubleComplex coefficient = make_cuDoubleComplex(1.0, 0.0);

    status = cudensitymatOperatorTermAppendGeneralProduct(
        handle, term, num_elem_operators, &num_operator_modes,
        &operator_mode_extents, &operator_mode_strides, state_modes_acted_on,
        nullptr, CUDA_C_64F, &tensor_data, &tensorCallback, coefficient,
        scalarCallback);
    if (status != CUDENSITYMAT_STATUS_SUCCESS) {
      cudensitymatDestroyOperatorTerm(term);
      cudensitymatDestroyOperator(operator_handle);
      throw std::runtime_error(
          "Failed to attach flat_matrix to operator term.");
    }

    status = cudensitymatOperatorAppendTerm(handle, operator_handle, term, 0,
                                            coefficient, scalarCallback);
    if (status != CUDENSITYMAT_STATUS_SUCCESS) {
      cudensitymatDestroyOperatorTerm(term);
      cudensitymatDestroyOperator(operator_handle);
      throw std::runtime_error("Failed to attach term to operator.");
    }

    cudensitymatDestroyOperatorTerm(term);
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
