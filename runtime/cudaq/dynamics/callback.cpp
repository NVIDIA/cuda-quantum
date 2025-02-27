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
#include <unordered_map>
#include <string>
#include <vector>

namespace cudaq {

// ScalarCallbackFunction

std::complex<double>
ScalarCallbackFunction::operator()(const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  return this->callback_func(parameters);
}

// MatrixCallbackFunction

matrix_2
MatrixCallbackFunction::operator()(const std::vector<int> &relevant_dimensions,
                                   const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  return this->callback_func(relevant_dimensions, parameters);
}

// Definition

Definition::Definition(std::string operator_id, const std::vector<int> &expected_dimensions, MatrixCallbackFunction &&create) 
  : id(operator_id), generator(std::move(create)), required_dimensions(expected_dimensions) {}

Definition::Definition(Definition &&def) 
  : id(def.id), generator(std::move(def.generator)), required_dimensions(std::move(def.expected_dimensions)) {}

matrix_2 Definition::generate_matrix(
    const std::vector<int> &relevant_dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  return generator(relevant_dimensions, parameters);
}

Definition::~Definition() = default;
} // namespace cudaq
