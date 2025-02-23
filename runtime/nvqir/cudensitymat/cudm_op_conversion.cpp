/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudm_op_conversion.h"
#include "cudm_error_handling.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace cudaq;

namespace cudaq {
// cudm_op_conversion::cudm_op_conversion(const cudensitymatHandle_t handle,
//                                        const std::map<int, int> &dimensions,
//                                        std::shared_ptr<Schedule> schedule)
//     : handle_(handle), dimensions_(dimensions), schedule_(schedule) {
//   if (handle_ == nullptr) {
//     throw std::runtime_error("Handle cannot be null.");
//   }

//   if (dimensions_.empty()) {
//     throw std::invalid_argument("Dimensions map must not be empty.");
//   }
// }

// std::vector<std::complex<double>> cudm_op_conversion::get_identity_matrix() {
//   size_t dim = 1;
//   for (const auto &entry : dimensions_) {
//     dim *= entry.second;
//   }

//   std::vector<std::complex<double>> identity_matrix(dim * dim, {0.0, 0.0});
//   for (size_t i = 0; i < dim; i++) {
//     identity_matrix[i * dim + i] = {1.0, 0.0};
//   }

//   return identity_matrix;
// }

// std::vector<int64_t> cudm_op_conversion::get_space_mode_extents() {
//   std::vector<int64_t> space_mode_extents;
//   for (const auto &dim : dimensions_) {
//     space_mode_extents.push_back(dim.second);
//   }

//   return space_mode_extents;
// }

// cudensitymatOperatorTerm_t cudm_op_conversion::_scalar_to_op(
//     const cudensitymatWrappedScalarCallback_t &scalar) {
//   std::vector<int64_t> space_mode_extents = get_space_mode_extents();

//   cudensitymatOperatorTerm_t op_term;
//   HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
//       handle_, dimensions_.size(), space_mode_extents.data(), &op_term));

//   void *tensor_data = create_array_gpu(get_identity_matrix());
//   if (!tensor_data) {
//     throw std::runtime_error("Failed to allocate GPU memory for
//     tensor_data.");
//   }

//   std::vector<int32_t> mode_action_duality(dimensions_.size(),
//                                            CUDENSITYMAT_OPERATOR_SPARSITY_NONE);

//   cudensitymatElementaryOperator_t identity;
//   HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
//       handle_, dimensions_.size(), space_mode_extents.data(),
//       CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, mode_action_duality.data(),
//       CUDA_C_64F, tensor_data, {nullptr, nullptr}, &identity));

//   std::vector<int32_t> states_modes_acted_on(dimensions_.size());
//   std::iota(states_modes_acted_on.begin(), states_modes_acted_on.end(), 0);

//   HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
//       handle_, op_term, 1, &identity, states_modes_acted_on.data(),
//       mode_action_duality.data(), {1.0, 0.0}, scalar));

//   return op_term;
// }

// cudensitymatOperator_t cudm_op_conversion::_callback_mult_op(
//     const cudensitymatWrappedScalarCallback_t &scalar,
//     const cudensitymatOperatorTerm_t &op) {
//   if (!op) {
//     throw std::invalid_argument("Invalid operator term (nullptr).");
//   }

//   std::vector<int64_t> space_mode_extents = get_space_mode_extents();

//   cudensitymatOperatorTerm_t scalar_op = _scalar_to_op(scalar);

//   if (!scalar_op) {
//     throw std::runtime_error("scalar_op is NULL.");
//   }

//   cudensitymatOperator_t new_op;
//   HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
//       handle_, static_cast<int32_t>(dimensions_.size()),
//       space_mode_extents.data(), &new_op));

//   std::vector<int32_t> mode_action_duality(dimensions_.size(),
//                                            CUDENSITYMAT_OPERATOR_SPARSITY_NONE);

//   HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle_, new_op,
//   scalar_op,
//                                                    mode_action_duality.size(),
//                                                    {1.0, 0.0}, scalar));

//   HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
//       handle_, new_op, op, mode_action_duality.size(), {1.0, 0.0},
//       {nullptr, nullptr}));

//   return new_op;
// }

// std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
//              std::complex<double>>
// cudm_op_conversion::tensor(
//     const std::variant<cudensitymatOperatorTerm_t,
//                        cudensitymatWrappedScalarCallback_t,
//                        std::complex<double>> &op1,
//     const std::variant<cudensitymatOperatorTerm_t,
//                        cudensitymatWrappedScalarCallback_t,
//                        std::complex<double>> &op2) {
//   if (std::holds_alternative<std::complex<double>>(op1) &&
//       std::holds_alternative<std::complex<double>>(op2)) {
//     return std::get<std::complex<double>>(op1) *
//            std::get<std::complex<double>>(op2);
//   }

//   if (std::holds_alternative<std::complex<double>>(op1)) {
//     return _callback_mult_op(
//         _wrap_callback(scalar_operator(std::get<std::complex<double>>(op1))),
//         std::get<cudensitymatOperatorTerm_t>(op2));
//   }

//   if (std::holds_alternative<std::complex<double>>(op2)) {
//     return _callback_mult_op(
//         _wrap_callback(scalar_operator(std::get<std::complex<double>>(op2))),
//         std::get<cudensitymatOperatorTerm_t>(op1));
//   }

//   if (std::holds_alternative<cudensitymatWrappedScalarCallback_t>(op1)) {
//     return tensor(
//         _scalar_to_op(std::get<cudensitymatWrappedScalarCallback_t>(op1)),
//         std::get<cudensitymatOperatorTerm_t>(op2));
//   }

//   if (std::holds_alternative<cudensitymatWrappedScalarCallback_t>(op2)) {
//     return tensor(
//         _scalar_to_op(std::get<cudensitymatWrappedScalarCallback_t>(op2)),
//         std::get<cudensitymatOperatorTerm_t>(op1));
//   }

//   std::vector<int64_t> space_mode_extents = get_space_mode_extents();

//   cudensitymatOperator_t result;
//   HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
//       handle_, dimensions_.size(), space_mode_extents.data(), &result));

//   HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
//       handle_, result, std::get<cudensitymatOperatorTerm_t>(op1), 0, {1.0,
//       0.0}, {nullptr, nullptr}));
//   HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
//       handle_, result, std::get<cudensitymatOperatorTerm_t>(op2), 0, {1.0,
//       0.0}, {nullptr, nullptr}));

//   return result;
// }

// std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
//              std::complex<double>>
// cudm_op_conversion::mul(const std::variant<cudensitymatOperatorTerm_t,
//                                            cudensitymatWrappedScalarCallback_t,
//                                            std::complex<double>> &op1,
//                         const std::variant<cudensitymatOperatorTerm_t,
//                                            cudensitymatWrappedScalarCallback_t,
//                                            std::complex<double>> &op2) {
//   return tensor(op1, op2);
// }

// std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
//              std::complex<double>>
// cudm_op_conversion::add(const std::variant<cudensitymatOperatorTerm_t,
//                                            cudensitymatWrappedScalarCallback_t,
//                                            std::complex<double>> &op1,
//                         const std::variant<cudensitymatOperatorTerm_t,
//                                            cudensitymatWrappedScalarCallback_t,
//                                            std::complex<double>> &op2) {
//   if (std::holds_alternative<std::complex<double>>(op1) &&
//       std::holds_alternative<std::complex<double>>(op2)) {
//     return std::get<std::complex<double>>(op1) +
//            std::get<std::complex<double>>(op2);
//   }

//   if (std::holds_alternative<std::complex<double>>(op1)) {
//     return _callback_mult_op(
//         _wrap_callback(scalar_operator(std::get<std::complex<double>>(op1))),
//         std::get<cudensitymatOperatorTerm_t>(op2));
//   }

//   if (std::holds_alternative<std::complex<double>>(op2)) {
//     return _callback_mult_op(
//         _wrap_callback(scalar_operator(std::get<std::complex<double>>(op2))),
//         std::get<cudensitymatOperatorTerm_t>(op1));
//   }

//   // FIXME: Need to check later
//   int32_t num_space_modes =
//       std::max(static_cast<int32_t>(dimensions_.size()), 1);
//   std::vector<int64_t> space_mode_extents = get_space_mode_extents();

//   cudensitymatOperator_t result;
//   HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
//       handle_, num_space_modes, space_mode_extents.data(), &result));

//   HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
//       handle_, result, std::get<cudensitymatOperatorTerm_t>(op1), 0, {1.0,
//       0.0}, {nullptr, nullptr}));
//   HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
//       handle_, result, std::get<cudensitymatOperatorTerm_t>(op2), 0, {1.0,
//       0.0}, {nullptr, nullptr}));

//   return result;
// }

// std::variant<cudensitymatOperatorTerm_t, cudensitymatWrappedScalarCallback_t,
//              std::complex<double>>
// cudm_op_conversion::evaluate(
//     const std::variant<scalar_operator, matrix_operator,
//                        product_operator<matrix_operator>> &op) {
//   if (std::holds_alternative<scalar_operator>(op)) {
//     const scalar_operator &scalar_op = std::get<scalar_operator>(op);

//     ScalarCallbackFunction generator = scalar_op.get_generator();

//     if (!generator) {
//       return scalar_op.evaluate({});
//     } else {
//       return _wrap_callback(scalar_op);
//     }
//   }

//   if (std::holds_alternative<matrix_operator>(op)) {
//     const matrix_operator &mat_op = std::get<matrix_operator>(op);

//     std::vector<int64_t> space_mode_extents = get_space_mode_extents();

//     cudensitymatOperatorTerm_t opterm;
//     HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
//         handle_, dimensions_.size(), space_mode_extents.data(), &opterm));

//     cudensitymatElementaryOperator_t elem_op;
//     // Need to check if it is a static, use nullptr
//     // or a callback and then only use callback
//     cudensitymatWrappedTensorCallback_t callback =
//         _wrap_callback_tensor(mat_op);

//     auto flat_matrix = flatten_matrix(mat_op.to_matrix(dimensions_, {}));

//     void *tensor_data = create_array_gpu(flat_matrix);
//     if (!tensor_data) {
//       throw std::runtime_error(
//           "Failed to allocate GPU memory for tensor_data.");
//     }

//     std::vector<int32_t> mode_action_duality(
//         mat_op.degrees.size(), CUDENSITYMAT_OPERATOR_SPARSITY_NONE);

//     HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
//         handle_, mat_op.degrees.size(), space_mode_extents.data(),
//         CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, mode_action_duality.data(),
//         CUDA_C_64F, tensor_data, callback, &elem_op));

//     HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
//         handle_, opterm, 1, &elem_op, mat_op.degrees.data(),
//         mode_action_duality.data(), {1.0, 0.0}, {nullptr, nullptr}));

//     return opterm;
//   }

//   if (std::holds_alternative<product_operator<matrix_operator>>(op)) {
//     throw std::runtime_error(
//         "Handling of product_operator<matrix_operator> is not implemented.");
//   }

//   throw std::runtime_error(
//       "Unknown operator type in cudm_op_conversion::evaluate.");
// }

} // namespace cudaq