/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/matrix.h"
#include "cudaq/utils/tensor.h"
#include <map>
#include <string>
#include <complex>
#include <vector>

namespace cudaq {
/// @brief Base class for all operator types.
class base_operator {
public:
    virtual ~base_operator() = default;

    /// @brief Evaluate the operator with given parameters
    virtual std::complex<double> evaluate(const std::map<std::string, std::complex<double>> &parameters) const = 0;

    /// @brief Convert the operator to a matrix representation.
    virtual matrix_2 to_matrix(const std::map<int, int> &dimensions, const std::map<std::string, std::complex<double>> &parameters = {}) const = 0;

    /// @brief Convert the operator to a string representation.
    virtual std::string to_string() const = 0;

    /// @brief Return the degrees of freedom that the operator acts on.
    virtual std::vector<int> degrees() const = 0;
};
}
