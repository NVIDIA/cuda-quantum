/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/matrix.h"
#include <unordered_map>
#include <vector>

namespace cudaq {
namespace detail {

/// Generates all possible states for the given dimensions ordered according
/// to the sequence of degrees (ordering is relevant if dimensions differ).
std::vector<std::string>
generate_all_states(const std::vector<int> &degrees,
                    const std::unordered_map<int, int> &dimensions);

/// Computes a vector describing the permutation to reorder a matrix that is
/// ordered according to `op_degrees` to apply to `canon_degrees` instead.
/// The dimensions define the number of levels for each degree of freedom.
/// The degrees of freedom in `op_degrees` and `canon_degrees` have to match.
std::vector<int>
compute_permutation(const std::vector<int> &op_degrees,
                    const std::vector<int> &canon_degrees,
                    const std::unordered_map<int, int> dimensions);

// Permutes the given matrix according to the given permutation.
// If states is the current order of vector entries on which the given matrix
// acts, and permuted_states is the desired order of an array on which the
// permuted matrix should act, then the permutation is defined such that
// [states[i] for i in permutation] produces permuted_states.
void permute_matrix(cudaq::complex_matrix &matrix,
                    const std::vector<int> &permutation);

} // namespace detail
} // namespace cudaq
