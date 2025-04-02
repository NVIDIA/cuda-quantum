/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
    const std::vector<int> &relevant_dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  return this->callback_func(relevant_dimensions, parameters);
}

// Definition

Definition::Definition(std::string operator_id,
                       const std::vector<int> &expected_dimensions,
                       matrix_callback &&create)
    : id(operator_id), generator(std::move(create)),
      required_dimensions(expected_dimensions) {}

Definition::Definition(Definition &&def)
    : id(def.id), generator(std::move(def.generator)),
      required_dimensions(std::move(def.expected_dimensions)) {}

complex_matrix Definition::generate_matrix(
    const std::vector<int> &relevant_dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  return generator(relevant_dimensions, parameters);
}

Definition::~Definition() = default;
} // namespace cudaq
