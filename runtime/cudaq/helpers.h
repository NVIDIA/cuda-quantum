/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <map>
#include <vector>
#include <string>
#include <complex>
#include <functional>
#include <regex>
#include <numeric>
#include <Eigen/Dense>

namespace cudaq {
class OperatorHelpers {
public:
    // Aggregate parameters from multiple mappings.
    static std::map<std::string, std::string> aggregate_parameters(const std::vector<std::map<std::string, std::string>> &parameter_mappings);

    // Extract documentation for a specific parameter from docstring.
    static std::string parameter_docs(const std::string &param_name, const std::string &docs);

    // Extract positional arguments and keyword-only arguments.
    static std::pair<std::vector<std::string>, std::map<std::string, std::string>> args_from_kwargs(const std::map<std::string, std::string> &kwargs,
    const std::vector<std::string> &required_args, const std::vector<std::string> &kwonly_args);

    // Generate all possible quantum states for given degrees and dimensions.
    static std::vector<std::string> generate_all_states(const std::vector<int> &degrees, const std::map<int, int> &dimensions);

    // Permute a given Eigen matrix.
    static void permute_matrix(Eigen::MatrixXcd &matrix, const std::vector<int> &permutation);

    // Canonicalize degrees by sorting in descending order.
    static std::vector<int> canonicalize_degrees(const std::vector<int> &degrees);
};
}