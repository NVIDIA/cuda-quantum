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

Definition::Definition() = default;

// Convenience setter
void Definition::create_definition(const std::string &operator_id,
                                   std::map<int, int> expected_dimensions,
                                   CallbackFunction &&create) {
  id = operator_id;
  generator = std::move(create);
  m_expected_dimensions = std::move(expected_dimensions);
}

matrix_2 Definition::generate_matrix(
    const std::map<int, int> &degrees,
    const std::map<std::string, std::complex<double>> &parameters) const {
  return generator(degrees, parameters);
}

Definition::~Definition() = default;
} // namespace cudaq
