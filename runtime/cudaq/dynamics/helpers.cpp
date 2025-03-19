/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "helpers.h"
#include <algorithm>
#include <unordered_map>

namespace cudaq {
namespace detail {

std::vector<std::string>
generate_all_states(const std::vector<int> &degrees,
                    const std::unordered_map<int, int> &dimensions) {
  if (degrees.size() == 0)
    return {};
  auto dit = degrees.crbegin();

  std::vector<std::string> states;
  auto entry = dimensions.find(*dit);
  assert(entry != dimensions.end());
  for (auto state = 0; state < entry->second; state++) {
    states.push_back(std::to_string(state));
  }

  while (++dit != degrees.crend()) {
    auto entry = dimensions.find(*dit);
    assert(entry != dimensions.end());
    std::vector<std::string> result;
    for (auto current : states) {
      for (auto state = 0; state < entry->second; state++) {
        result.push_back(std::to_string(state) + current);
      }
    }
    states = result;
  }

  return states;
}

std::vector<int>
compute_permutation(const std::vector<int> &op_degrees,
                    const std::vector<int> &canon_degrees,
                    const std::unordered_map<int, int> dimensions) {
  assert(op_degrees.size() == canon_degrees.size());
  auto states = cudaq::detail::generate_all_states(canon_degrees, dimensions);

  std::vector<int> reordering;
  for (auto degree : op_degrees) {
    auto it = std::find(canon_degrees.cbegin(), canon_degrees.cend(), degree);
    reordering.push_back(it - canon_degrees.cbegin());
  }

  std::vector<std::string> op_states =
      cudaq::detail::generate_all_states(op_degrees, dimensions);

  std::vector<int> permutation;
  for (const auto &state : states) {
    std::string term;
    for (auto i : reordering) {
      term += state[i];
    }
    auto it = std::find(op_states.cbegin(), op_states.cend(), term);
    permutation.push_back(it - op_states.cbegin());
  }

  return std::move(permutation);
}

void permute_matrix(cudaq::complex_matrix &matrix,
                    const std::vector<int> &permutation) {
  if (permutation.size() == 0) {
    assert(matrix.rows() == matrix.cols() == 1);
    return;
  }

  std::vector<std::complex<double>> sorted_values;
  for (std::size_t permuted : permutation) {
    for (std::size_t permuted_again : permutation) {
      sorted_values.push_back(matrix[{permuted, permuted_again}]);
    }
  }
  int idx = 0;
  for (std::size_t row = 0; row < matrix.rows(); row++) {
    for (std::size_t col = 0; col < matrix.cols(); col++) {
      matrix[{row, col}] = sorted_values[idx];
      idx++;
    }
  }
}

cudaq::csr_spmatrix to_csr_matrix(const Eigen::SparseMatrix<std::complex<double>> &matrix, 
                                  std::size_t estimated_num_entries) {
  std::vector<std::complex<double>> values;
  std::vector<std::size_t> rows, cols;
  values.reserve(estimated_num_entries);
  rows.reserve(estimated_num_entries);
  cols.reserve(estimated_num_entries);
  using SparseMatrix = Eigen::SparseMatrix<std::complex<double>>;
  for (int k = 0; k < matrix.outerSize(); ++k)
    for (SparseMatrix::InnerIterator it(matrix, k); it; ++it) {
      values.emplace_back(it.value());
      rows.emplace_back(it.row());
      cols.emplace_back(it.col());
    }
  return std::make_tuple(values, rows, cols);                                      
}

} // namespace detail
} // namespace cudaq
