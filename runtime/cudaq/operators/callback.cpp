/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "callback.h"

#include <complex>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq {

// scalar_callback

std::complex<double> scalar_callback::operator()(
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  return this->callback_func(parameters);
}

// matrix_callback

complex_matrix matrix_callback::operator()(
    const std::vector<std::int64_t> &relevant_dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  return this->callback_func(relevant_dimensions, parameters);
}

// diag_matrix_callback

mdiag_sparse_matrix diag_matrix_callback::operator()(
    const std::vector<std::int64_t> &relevant_dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  return this->callback_func(relevant_dimensions, parameters);
}

// Definition

Definition::Definition(
    std::string operator_id,
    const std::vector<std::int64_t> &expected_dimensions,
    matrix_callback &&create,
    std::unordered_map<std::string, std::string> &&parameter_descriptions)
    : id(operator_id), generator(std::move(create)),
      parameter_descriptions(std::move(parameter_descriptions)),
      required_dimensions(expected_dimensions) {}

Definition::Definition(Definition &&def)
    : id(def.id), generator(std::move(def.generator)),
      diag_generator(std::move(def.diag_generator)),
      parameter_descriptions(std::move(def.parameter_descriptions)),
      required_dimensions(std::move(def.expected_dimensions)) {}

Definition::Definition(
    std::string operator_id, const std::vector<int64_t> &expected_dimensions,
    matrix_callback &&create, diag_matrix_callback &&diag_create,
    std::unordered_map<std::string, std::string> &&parameter_descriptions)
    : id(operator_id), generator(std::move(create)),
      diag_generator(std::move(diag_create)),
      parameter_descriptions(std::move(parameter_descriptions)),
      required_dimensions(std::move(expected_dimensions)) {}

complex_matrix Definition::generate_matrix(
    const std::vector<std::int64_t> &relevant_dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  return generator(relevant_dimensions, parameters);
}

mdiag_sparse_matrix Definition::generate_dia_matrix(
    const std::vector<std::int64_t> &relevant_dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  return diag_generator.value()(relevant_dimensions, parameters);
}
Definition::~Definition() = default;
} // namespace cudaq
