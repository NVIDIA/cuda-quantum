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
ScalarCallbackFunction::operator()(std::map<std::string, std::complex<double>> parameters) const {
  return _callback_func(std::move(parameters));
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
MatrixCallbackFunction::operator()(std::vector<int> relevant_dimensions,
            std::map<std::string, std::complex<double>> parameters) const {
  return _callback_func(std::move(relevant_dimensions), std::move(parameters));
}

// Definition

Definition::Definition(const std::string &operator_id, std::vector<int> expected_dimensions, MatrixCallbackFunction &&create) 
  : id(operator_id), generator(std::move(create)), m_expected_dimensions(std::move(expected_dimensions)) {}

Definition::Definition(Definition &&def) 
  : id(def.id), generator(std::move(def.generator)), m_expected_dimensions(std::move(def.m_expected_dimensions)) {}

matrix_2 Definition::generate_matrix(
    const std::vector<int> &relevant_dimensions,
    const std::map<std::string, std::complex<double>> &parameters) const {
  return generator(relevant_dimensions, parameters);
}

Definition::~Definition() = default;
} // namespace cudaq
