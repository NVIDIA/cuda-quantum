/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/matrix.h"
#include <functional>
#include <unordered_map>
#include <vector>

namespace Eigen {
// forward declared here so that this header can be used even if the Eigen is
// not used/found
template <typename Scalar_, int Options_, typename StorageIndex_>
class SparseMatrix;
} // namespace Eigen

namespace cudaq {
using csr_spmatrix =
    std::tuple<std::vector<std::complex<double>>, std::vector<std::size_t>,
               std::vector<std::size_t>>;

namespace detail {
struct states_hash {
  int operator()(const std::vector<std::int64_t> &vect) const;
};

// SparseMatrix really wants a *signed* type
using EigenSparseMatrix =
    Eigen::SparseMatrix<std::complex<double>, 0x1, long>; // row major

/// Generates all possible states for the given dimensions.
std::vector<std::vector<std::int64_t>>
generate_all_states(const std::vector<std::int64_t> &dimensions);

/// Generates all possible states for the given dimensions ordered according
/// to the sequence of degrees (ordering is relevant if dimensions differ).
std::vector<std::vector<std::int64_t>> generate_all_states(
    const std::vector<std::size_t> &degrees,
    const std::unordered_map<std::size_t, std::int64_t> &dimensions);

/// Computes a vector describing the permutation to reorder a matrix that is
/// ordered according to `op_degrees` to apply to `canon_degrees` instead.
/// The dimensions define the number of levels for each degree of freedom.
/// The degrees of freedom in `op_degrees` and `canon_degrees` have to match.
std::vector<std::size_t> compute_permutation(
    const std::vector<std::size_t> &op_degrees,
    const std::vector<std::size_t> &canon_degrees,
    const std::unordered_map<std::size_t, std::int64_t> dimensions);

/// Permutes the given matrix according to the given permutation.
/// If states is the current order of vector entries on which the given matrix
/// acts, and permuted_states is the desired order of an array on which the
/// permuted matrix should act, then the permutation is defined such that
/// [states[i] for i in permutation] produces permuted_states.
void permute_matrix(cudaq::complex_matrix &matrix,
                    const std::vector<std::size_t> &permutation);

/// Helper function for matrix creation.
/// The matrix creation function should call its function argument for
/// every entry in the matrix.
complex_matrix create_matrix(
    std::size_t dim, std::complex<double> coeff,
    const std::function<void(const std::function<void(std::size_t, std::size_t,
                                                      std::complex<double>)> &)>
        &create);

/// Helper function for sparse matrix creation.
/// The matrix creation function should call its function argument for
/// every entry in the matrix.
EigenSparseMatrix create_sparse_matrix(
    std::size_t dim, std::complex<double> coeff,
    const std::function<void(const std::function<void(std::size_t, std::size_t,
                                                      std::complex<double>)> &)>
        &create);

/// Converts and Eigen sparse matrix to the `csr_spmatrix` format used in
/// CUDA-Q.
csr_spmatrix to_csr_spmatrix(const EigenSparseMatrix &matrix,
                             std::size_t estimated_num_entries);

} // namespace detail
} // namespace cudaq
