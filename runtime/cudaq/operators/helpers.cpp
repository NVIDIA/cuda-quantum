/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "helpers.h"
#include "common/EigenSparse.h"
#include <algorithm>
#include <set>
#include <unordered_map>

namespace cudaq {
namespace detail {

int states_hash::operator()(const std::vector<int64_t> &vect) const {
  int hash = vect.size();
  for (auto &i : vect)
    hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  return hash;
}

std::vector<std::vector<int64_t>>
generate_all_states(const std::vector<int64_t> &dimensions) {
  if (dimensions.size() == 0)
    return {};
  auto dim_it = dimensions.cbegin();
  auto tot_nr_states = 1;
  for (auto d : dimensions)
    tot_nr_states *= d;

  std::vector<std::vector<int64_t>> states;
  states.reserve(tot_nr_states);
  for (int64_t state = 0; state < *dim_it; state++) {
    std::vector<int64_t> expanded_state;
    expanded_state.reserve(dimensions.size());
    expanded_state.push_back(state);
    states.push_back(expanded_state);
  }

  while (++dim_it != dimensions.cend()) {
    std::size_t current_size = states.size();
    for (int64_t state = 1; state < *dim_it; state++) {
      for (std::size_t idx = 0; idx < current_size; ++idx) {
        std::vector<int64_t> expanded_state;
        expanded_state.reserve(dimensions.size());
        expanded_state.insert(expanded_state.end(), states[idx].cbegin(),
                              states[idx].cend());
        expanded_state.push_back(state);
        states.push_back(expanded_state);
      }
    }
    for (std::size_t idx = 0; idx < current_size; ++idx)
      states[idx].push_back(0);
  }

  return states;
}

std::vector<std::vector<int64_t>> generate_all_states(
    const std::vector<std::size_t> &degrees,
    const std::unordered_map<std::size_t, int64_t> &dimensions) {
  std::vector<int64_t> relevant_dimensions;
  relevant_dimensions.reserve(degrees.size());
  for (auto d : degrees) {
    auto it = dimensions.find(d);
    assert(it != dimensions.cend());
    relevant_dimensions.push_back(it->second);
  }
  return generate_all_states(relevant_dimensions);
}

std::vector<std::size_t>
compute_permutation(const std::vector<std::size_t> &op_degrees,
                    const std::vector<std::size_t> &canon_degrees,
                    const std::unordered_map<std::size_t, int64_t> dimensions) {
  // canon_degrees and op_degrees should be the same up to permutation
  assert(op_degrees.size() == canon_degrees.size());
  assert(std::set<std::size_t>(op_degrees.cbegin(), op_degrees.cend()).size() ==
         op_degrees.size());
  assert(std::set<std::size_t>(canon_degrees.cbegin(), canon_degrees.cend())
             .size() == canon_degrees.size());
  assert(std::set<std::size_t>(op_degrees.cbegin(), op_degrees.cend()) ==
         std::set<std::size_t>(canon_degrees.cbegin(), canon_degrees.cend()));

  auto states = cudaq::detail::generate_all_states(canon_degrees, dimensions);

  std::vector<int64_t> reordering;
  reordering.reserve(op_degrees.size());
  for (auto degree : op_degrees) {
    auto it = std::find(canon_degrees.cbegin(), canon_degrees.cend(), degree);
    reordering.push_back(it - canon_degrees.cbegin());
  }

  auto op_states = cudaq::detail::generate_all_states(op_degrees, dimensions);
  // probably worth creating a hashmap for faster lookup
  std::unordered_map<std::vector<int64_t>, std::size_t, states_hash>
      op_states_map;
  op_states_map.reserve(op_states.size());
  for (std::size_t idx = 0; idx < op_states.size(); ++idx)
    op_states_map[std::move(op_states[idx])] = idx;

  std::vector<std::size_t> permutation;
  permutation.reserve(states.size());
  for (const auto &state : states) {
    std::vector<int64_t> term;
    term.reserve(reordering.size());
    for (auto i : reordering) {
      term.push_back(state[i]);
    }
    auto it = op_states_map.find(term);
    assert(it != op_states_map.cend());
    permutation.push_back(it->second);
  }

  return permutation;
}

void permute_matrix(cudaq::complex_matrix &matrix,
                    const std::vector<std::size_t> &permutation) {
  assert(matrix.rows() == matrix.cols());
  assert(permutation.size() == 0 || permutation.size() == matrix.rows());
  if (permutation.size() == 0) {
    assert(matrix.rows() == 1);
    return;
  }

  // We do an in-place permutation of rows and columns.
  // Since we don't (can't?) do it simultaneously, the number of operators is
  // roughly 2*N*N. The same number of operators would be required if we first
  // copy the matrix, and then populate a new matrix directly with the correct
  // entries. So, runtime-wise both should amount to the same, but we safe a
  // factor 2 of memory this way.
  // The outermost for-loop (the one with the iterator) walks the cycles in the
  // permutation, the second for-loop (the one in process_cycle) performs the
  // permutations for that cycle, and the innermost for-loop (the one in
  // swap_row/swap_col) iterates over the columns/rows to grab the entire
  // row/column.
  std::vector<bool> processed;
  processed.reserve(matrix.rows());
  auto process_cycle = [&processed, &permutation](
                           std::size_t start,
                           std::function<void(std::size_t, std::size_t)> swap) {
    processed[start] = true;
    for (std::size_t row = start; permutation[row] != start;
         row = permutation[row]) {
      processed[permutation[row]] = true;
      swap(row, permutation[row]);
    }
  };

  // in-place permutation of rows
  auto swap_row = [&matrix](std::size_t row1, std::size_t row2) {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
      std::swap(matrix[{row1, col}], matrix[{row2, col}]);
  };
  processed.assign(matrix.rows(), false);
  for (auto it = processed.begin(); it != processed.end();
       it = std::find(it, processed.end(), false))
    process_cycle(it - processed.cbegin(), swap_row);

  // in-place permutation of columns
  auto swap_col = [&matrix](std::size_t col1, std::size_t col2) {
    for (std::size_t row = 0; row < matrix.rows(); ++row)
      std::swap(matrix[{row, col1}], matrix[{row, col2}]);
  };
  processed.assign(matrix.cols(), false);
  for (auto it = processed.begin(); it != processed.end();
       it = std::find(it, processed.end(), false))
    process_cycle(it - processed.cbegin(), swap_col);
}

complex_matrix create_matrix(
    std::size_t dim, std::complex<double> coeff,
    const std::function<void(const std::function<void(std::size_t, std::size_t,
                                                      std::complex<double>)> &)>
        &create) {
  complex_matrix matrix(dim, dim);
  auto process_entry = [&matrix, &coeff](std::size_t new_state,
                                         std::size_t old_state,
                                         std::complex<double> entry) {
    matrix[{new_state, old_state}] = coeff * entry;
  };
  create(process_entry);
  return matrix;
}

EigenSparseMatrix create_sparse_matrix(
    std::size_t dim, std::complex<double> coeff,
    const std::function<void(const std::function<void(std::size_t, std::size_t,
                                                      std::complex<double>)> &)>
        &create) {
  using Triplet = Eigen::Triplet<std::complex<double>>;
  std::vector<Triplet> triplets;
  triplets.reserve(dim);
  auto process_entry = [&triplets, &coeff](std::size_t new_state,
                                           std::size_t old_state,
                                           std::complex<double> entry) {
    triplets.push_back(Triplet(new_state, old_state, coeff * entry));
  };
  create(process_entry);
  cudaq::detail::EigenSparseMatrix matrix(dim, dim);
  matrix.setFromTriplets(triplets.begin(), triplets.end());
  return matrix;
}

cudaq::csr_spmatrix to_csr_spmatrix(const EigenSparseMatrix &matrix,
                                    std::size_t estimated_num_entries) {
  std::vector<std::complex<double>> values;
  std::vector<std::size_t> rows, cols;
  values.reserve(estimated_num_entries);
  rows.reserve(estimated_num_entries);
  cols.reserve(estimated_num_entries);
  for (std::size_t k = 0; k < matrix.outerSize(); ++k)
    for (EigenSparseMatrix::InnerIterator it(matrix, k); it; ++it) {
      values.emplace_back(it.value());
      rows.emplace_back(it.row());
      cols.emplace_back(it.col());
    }
  return std::make_tuple(values, rows, cols);
}

} // namespace detail
} // namespace cudaq
