/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/definition.h"
#include "cudaq/qis/state.h"

#include <complex>
#include <functional>
#include <string>
#include <vector>

namespace cudaq {

Definition::Definition(const std::string &operator_id, std::vector<int> expected_dimensions, CallbackFunction &&create) 
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
