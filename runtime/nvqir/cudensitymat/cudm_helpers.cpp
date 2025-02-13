/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudm_helpers.h"
#include "cudm_error_handling.h"

using namespace cudaq;

namespace cudaq {
cudm_helper::cudm_helper(cudensitymatHandle_t handle) : handle(handle) {}

cudm_helper::~cudm_helper() {
  cudaDeviceSynchronize();
}

cudensitymatWrappedScalarCallback_t
cudm_helper::_wrap_callback(const scalar_operator &scalar_op) {
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
cudm_helper::_wrap_tensor_callback(const matrix_operator &op) {
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
    const std::map<std::string, std::complex<double>> &parameters,
    const std::vector<int64_t> &mode_extents) {
  auto subspace_extents = get_subspace_extents(mode_extents, elem_op.degrees);
  auto flat_matrix = flatten_matrix(
      elem_op.to_matrix(convert_dimensions(mode_extents), parameters));

  if (flat_matrix.empty()) {
    throw std::invalid_argument("Input matrix (flat matrix) cannot be empty.");
  }

  if (subspace_extents.empty()) {
    throw std::invalid_argument("subspace_extents cannot be empty.");
  }

  cudensitymatWrappedTensorCallback_t wrapped_tensor_callback = {nullptr,
                                                                 nullptr};

  if (!parameters.empty()) {
    wrapped_tensor_callback = _wrap_tensor_callback(elem_op);
  }

  // FIXME: leak (need to track this buffer somewhere and delete **after** the
  // whole evolve)
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

// Function to create and append a scalar to a term
void cudm_helper::append_scalar_to_term(cudensitymatOperatorTerm_t term,
                                        const scalar_operator &scalar_op) {
  cudensitymatWrappedScalarCallback_t wrapped_callback = {nullptr, nullptr};

  if (!scalar_op.get_generator()) {
    std::complex<double> coeff = scalar_op.evaluate({});
    HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
        handle, term, 0, nullptr, nullptr, nullptr,
        {make_cuDoubleComplex(coeff.real(), coeff.imag())}, wrapped_callback));
  } else {
    wrapped_callback = _wrap_callback(scalar_op);
    HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
        handle, term, 0, nullptr, nullptr, nullptr,
        {make_cuDoubleComplex(1.0, 0.0)}, wrapped_callback));
  }
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

// c_ops: std::vector<operator_sum>
cudensitymatOperator_t cudm_helper::compute_lindblad_operator(
    const std::vector<matrix_2> &c_ops,
    const std::vector<int64_t> &mode_extents) {
  if (c_ops.empty()) {
    throw std::invalid_argument("Collapse operators cannot be empty.");
  }

  cudensitymatOperator_t lindblad_op;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
      handle, static_cast<int32_t>(mode_extents.size()), mode_extents.data(),
      &lindblad_op));

  std::vector<cudensitymatOperatorTerm_t> terms;
  std::vector<cudensitymatElementaryOperator_t> elem_ops;
  std::vector<std::vector<int>> all_degrees;
  std::vector<std::vector<int>> all_action_dual_modalities;

  try {
    for (const auto &c_op : c_ops) {
      size_t dim = c_op.get_rows();

      if (dim == 0 || c_op.get_columns() != dim) {
        throw std::invalid_argument(
            "Collapse operator must be a square matrix.");
      }

      matrix_2 L_dagger_op_matrix = matrix_2::adjoint(c_op);

      const std::string L_id = "L_op";
      const std::string L_dagger_id = "L_dagger_op";

      cudaq::matrix_operator::define(
          L_id, {-1},
          [c_op](std::vector<int> dims,
                 std::map<std::string, std::complex<double>>) { return c_op; });

      cudaq::matrix_operator::define(
          L_dagger_id, {-1},
          [L_dagger_op_matrix](std::vector<int> dims,
                               std::map<std::string, std::complex<double>>) {
            return L_dagger_op_matrix;
          });

      matrix_operator L_op(L_id, {0});
      matrix_operator L_dagger_op(L_dagger_id, {0});

      cudensitymatElementaryOperator_t L_elem_op =
          create_elementary_operator(L_op, {}, mode_extents);

      cudensitymatElementaryOperator_t L_dagger_elem_op =
          create_elementary_operator(L_dagger_op, {}, mode_extents);

      if (!L_elem_op || !L_dagger_elem_op) {
        throw std::runtime_error("Failed to create elementary operators in "
                                 "compute_lindblad_operator.");
      }

      elem_ops.emplace_back(L_elem_op);
      all_degrees.emplace_back(L_op.degrees);

      std::vector<int> mod_vec(L_op.degrees.size(), 1);
      all_action_dual_modalities.emplace_back(mod_vec);

      elem_ops.emplace_back(L_dagger_elem_op);
      all_degrees.emplace_back(L_dagger_op.degrees);

      mod_vec = std::vector<int>(L_dagger_op.degrees.size(), 0);
      all_action_dual_modalities.emplace_back(mod_vec);

      // D1 = L * Lt
      cudensitymatOperatorTerm_t term_D1;
      HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
          handle, static_cast<int32_t>(mode_extents.size()),
          mode_extents.data(), &term_D1));

      append_elementary_operator_to_term(term_D1, elem_ops, all_degrees,
                                         all_action_dual_modalities);

      elem_ops.clear();
      all_degrees.clear();
      all_action_dual_modalities.clear();

      // Add term D1 to the Lindblad operator
      cudensitymatWrappedScalarCallback_t scalar_callback = {nullptr, nullptr};
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          handle, lindblad_op, term_D1, 0, make_cuDoubleComplex(1.0, 0.0),
          scalar_callback));

      // elem_ops.emplace_back(L_dagger_elem_op);
      // all_degrees.emplace_back(L_dagger_op.degrees);

      // elem_ops.emplace_back(L_elem_op);
      // all_degrees.emplace_back(L_op.degrees);

      // mod_vec = std::vector<int>(L_dagger_op.degrees.size(), 0);
      // all_action_dual_modalities.emplace_back(mod_vec);

      // mod_vec = std::vector<int>(L_op.degrees.size(), 0);
      // all_action_dual_modalities.emplace_back(mod_vec);

      // // D2 = -0.5 * (Lt * L)
      // cudensitymatOperatorTerm_t term_D2;
      // HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
      //     handle, static_cast<int32_t>(mode_extents.size()),
      //     mode_extents.data(), &term_D2));

      // append_elementary_operator_to_term(term_D2, elem_ops, all_degrees,
      // all_action_dual_modalities);

      // // Clear vectors
      // elem_ops.clear();
      // all_degrees.clear();
      // all_action_dual_modalities.clear();

      // Add term D2 to the Lindblad operator
      // HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
      //     handle, lindblad_op, term_D2, 0, make_cuDoubleComplex(-0.5, 0.0),
      //     scalar_callback));

      // elem_ops.emplace_back(L_elem_op);
      // all_degrees.emplace_back(L_op.degrees);

      // elem_ops.emplace_back(L_dagger_elem_op);
      // all_degrees.emplace_back(L_dagger_op.degrees);

      // mod_vec = std::vector<int>(L_op.degrees.size(), 1);
      // all_action_dual_modalities.emplace_back(mod_vec);

      // mod_vec = std::vector<int>(L_dagger_op.degrees.size(), 1);
      // all_action_dual_modalities.emplace_back(mod_vec);

      // // D3 = -0.5 * (L * Lt)
      // cudensitymatOperatorTerm_t term_D3;
      // HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
      //     handle, static_cast<int32_t>(mode_extents.size()),
      //     mode_extents.data(), &term_D3));
      // append_elementary_operator_to_term(term_D3, elem_ops, all_degrees,
      // all_action_dual_modalities);

      // // Clear vectors
      // elem_ops.clear();
      // all_degrees.clear();
      // all_action_dual_modalities.clear();

      // // Add term D3 to the Lindblad operator
      // HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
      //     handle, lindblad_op, term_D3, 0, make_cuDoubleComplex(-0.5, 0.0),
      //     scalar_callback));
    }

    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  } catch (const std::exception &e) {
    std::cerr << "Exception in compute_lindblad_operator: " << e.what()
              << std::endl;

    for (auto term : terms) {
      HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(term));
    }

    for (auto elem_op : elem_ops) {
      HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(elem_op));
    }

    cudensitymatDestroyOperator(lindblad_op);
    return nullptr;
  }

  // for (auto term : terms) {
  //   HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(term));
  // }

  // for (auto elem_op : elem_ops) {
  //   HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(elem_op));
  // }

  return lindblad_op;
}

std::map<int, int>
cudm_helper::convert_dimensions(const std::vector<int64_t> &mode_extents) {

  std::map<int, int> dimensions;
  for (size_t i = 0; i < mode_extents.size(); i++) {
    dimensions[static_cast<int>(i)] = static_cast<int>(mode_extents[i]);
  }

  return dimensions;
}

template <typename HandlerTy>
cudensitymatOperator_t cudm_helper::convert_to_cudensitymat_operator(
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

    // std::vector<cudensitymatElementaryOperator_t> elementary_operators;

    for (const auto &product_op : op.get_terms()) {
      cudensitymatOperatorTerm_t term;
      HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
          handle, static_cast<int32_t>(mode_extents.size()),
          mode_extents.data(), &term));

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
          all_degrees.emplace_back(elem_op->degrees);
        } else {
          // Catch anything that we don't know
          throw std::runtime_error("Unhandled type!");
        }
      }
      append_elementary_operator_to_term(term, elem_ops, all_degrees, {});
      auto coeff = product_op.get_coefficient();
      cudensitymatWrappedScalarCallback_t wrapped_callback = {nullptr, nullptr};

      if (!coeff.get_generator()) {
        const auto coeffVal = coeff.evaluate();
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            handle, operator_handle, term, 0,
            make_cuDoubleComplex(coeffVal.real(), coeffVal.imag()),
            wrapped_callback));
      } else {
        wrapped_callback = _wrap_callback(coeff);
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            handle, operator_handle, term, 0, make_cuDoubleComplex(1.0, 0.0),
            wrapped_callback));
      }

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

cudensitymatOperator_t cudm_helper::construct_liouvillian(
    const operator_sum<cudaq::matrix_operator> &op,
    const std::vector<operator_sum<cudaq::matrix_operator> *>
        &collapse_operators,
    const std::vector<int64_t> &mode_extents,
    const std::map<std::string, std::complex<double>> &parameters,
    bool is_master_equation) {
  if (!is_master_equation && collapse_operators.empty()) {
    auto liouvillian = op * std::complex<double>(0.0, -1.0);
    return convert_to_cudensitymat_operator(parameters, liouvillian,
                                            mode_extents);
  } else {
    throw std::runtime_error("TODO: handle Lindblad equation");
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
    const std::map<std::string, std::complex<double>> &,
    const operator_sum<cudaq::matrix_operator> &, const std::vector<int64_t> &);

} // namespace cudaq
