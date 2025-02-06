/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/cudm_error_handling.h"
#include "cudaq/operators.h"
#include "cudaq/utils/tensor.h"
#include <cudensitymat.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace cudaq {
void scale_state(cudensitymatHandle_t handle, cudensitymatState_t state,
                 double scale_factor, cudaStream_t stream);

cudensitymatOperator_t
compute_lindblad_operator(cudensitymatHandle_t handle,
                          const std::vector<matrix_2> &c_ops,
                          const std::vector<int64_t> &mode_extents);

// std::map<int, int>
// convert_dimensions(const std::vector<int64_t> &mode_extents) {
//   std::map<int, int> dimensions;
//   for (size_t i = 0; i < mode_extents.size(); i++) {
//     dimensions[static_cast<int>(i)] = static_cast<int>(mode_extents[i]);
//   }
//   return dimensions;
// }

// template <typename HandlerTy>
// cudensitymatOperator_t convert_to_cudensitymat_operator(
//     cudensitymatHandle_t handle,
//     const std::map<std::string, std::complex<double>> &parameters,
//     const operator_sum<HandlerTy> &op,
//     const std::vector<int64_t> &mode_extents) {
//   if (op.get_terms().empty()) {
//     throw std::invalid_argument("Operator sum cannot be empty.");
//   }

//   try {
//     cudensitymatOperator_t operator_handle;
//     HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
//         handle, static_cast<int32_t>(mode_extents.size()),
//         mode_extents.data(), &operator_handle));

//     std::vector<cudensitymatElementaryOperator_t> elementary_operators;

//     for (const auto &product_op : op.get_terms()) {
//       cudensitymatOperatorTerm_t term;

//       HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
//           handle, static_cast<int32_t>(mode_extents.size()),
//           mode_extents.data(), &term));

//       for (const auto &component : product_op.get_terms()) {
//         if (std::holds_alternative<HandlerTy>(component)) {
//           const auto &elem_op = std::get<cudaq::matrix_operator>(component);

//           auto subspace_extents =
//               get_subspace_extents(mode_extents, elem_op.degrees);
//           auto flat_matrix = flatten_matrix(
//               elem_op.to_matrix(convert_dimensions(mode_extents),
//               parameters));
//           auto cudm_elem_op =
//               create_elementary_operator(handle, subspace_extents,
//               flat_matrix);

//           elementary_operators.push_back(cudm_elem_op);
//           append_elementary_operator_to_term(handle, term, cudm_elem_op,
//                                              elem_op.degrees);
//         } else if (std::holds_alternative<cudaq::scalar_operator>(component))
//         {
//           auto coeff =
//               std::get<cudaq::scalar_operator>(component).evaluate(parameters);
//           append_scalar_to_term(handle, term, coeff);
//         }
//       }

//       // Append the product operator term to the top-level operator
//       HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
//           handle, operator_handle, term, 0, make_cuDoubleComplex(1.0, 0.0),
//           {nullptr, nullptr}));

//       // Destroy the term
//       HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(term));

//       // Cleanup
//       for (auto &elem_op : elementary_operators) {
//         HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(elem_op));
//       }
//     }

//     return operator_handle;
//   } catch (const std::exception &e) {
//     throw std::runtime_error("Error in convert_to_cudensitymat_operator!");
//   }
// }

cudensitymatOperator_t construct_liovillian(
    cudensitymatHandle_t handle, const cudensitymatOperator_t &hamiltonian,
    const std::vector<cudensitymatOperator_t> &collapse_operators,
    double gamma);

// Function for creating an array copy in GPU memory
void *create_array_gpu(const std::vector<std::complex<double>> &cpu_array);

// Function to detsroy a previously created array copy in GPU memory
void destroy_array_gpu(void *gpu_array);
} // namespace cudaq