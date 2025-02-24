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

ScalarCallbackFunction::ScalarCallbackFunction(const ScalarCallbackFunction &other) {
  _callback_func = other._callback_func;
}

ScalarCallbackFunction::ScalarCallbackFunction(ScalarCallbackFunction &&other) {
  _callback_func = std::move(other._callback_func);
}

ScalarCallbackFunction& ScalarCallbackFunction::operator=(const ScalarCallbackFunction &other) {
  if (this != &other) {
    _callback_func = other._callback_func;
  }
  return *this;
}

ScalarCallbackFunction& ScalarCallbackFunction::operator=(ScalarCallbackFunction &&other) {
  if (this != &other) {
    _callback_func = std::move(other._callback_func);
  }
  return *this;
}

std::complex<double>
ScalarCallbackFunction::operator()(const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  return _callback_func(parameters);
}

// MatrixCallbackFunction

MatrixCallbackFunction::MatrixCallbackFunction(const MatrixCallbackFunction &other) {
  _callback_func = other._callback_func;
}

MatrixCallbackFunction::MatrixCallbackFunction(MatrixCallbackFunction &&other) {
  _callback_func = std::move(other._callback_func);
}

MatrixCallbackFunction& MatrixCallbackFunction::operator=(const MatrixCallbackFunction &other) {
  if (this != &other) {
    _callback_func = other._callback_func;
  }
  return *this;
}

MatrixCallbackFunction& MatrixCallbackFunction::operator=(MatrixCallbackFunction &&other) {
  if (this != &other) {
    _callback_func = std::move(other._callback_func);
  }
  return *this;
}

matrix_2
MatrixCallbackFunction::operator()(const std::vector<int> &relevant_dimensions,
                                   const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  return _callback_func(relevant_dimensions, parameters);
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
