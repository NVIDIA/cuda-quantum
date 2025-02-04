/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/tensor.h"
#include <map>
#include <vector>

namespace cudaq {
namespace detail {

// Aggregate parameters from multiple mappings.
std::map<std::string, std::string> aggregate_parameters(
    const std::vector<std::map<std::string, std::string>> &parameter_mappings);

// Extract documentation for a specific parameter from docstring.
std::string parameter_docs(const std::string &param_name,
                           const std::string &docs);

// Extract positional arguments and keyword-only arguments.
std::pair<std::vector<std::string>, std::map<std::string, std::string>>
args_from_kwargs(const std::map<std::string, std::string> &kwargs,
                 const std::vector<std::string> &required_args,
                 const std::vector<std::string> &kwonly_args);

/// Generates all possible states for the given dimensions ordered according
/// to the sequence of degrees (ordering is relevant if dimensions differ).
std::vector<std::string> generate_all_states(std::vector<int> degrees,
                                             std::map<int, int> dimensions);

// Permutes the given matrix according to the given permutation.
// If states is the current order of vector entries on which the given matrix
// acts, and permuted_states is the desired order of an array on which the
// permuted matrix should act, then the permutation is defined such that
// [states[i] for i in permutation] produces permuted_states.
cudaq::matrix_2 permute_matrix(cudaq::matrix_2 matrix,
                               std::vector<int> permutation);

// Returns the degrees sorted in canonical order.
std::vector<int> canonicalize_degrees(std::vector<int> degrees);
} // namespace detail
} // namespace cudaq
