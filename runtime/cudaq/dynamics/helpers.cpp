/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "helpers.h"
#include <unordered_map>

namespace cudaq {
namespace detail {

std::vector<std::string>
generate_all_states(const std::vector<int> &degrees,
                    const std::unordered_map<int, int> &dimensions) {
  if (degrees.size() == 0)
    return {};

  std::vector<std::string> states;
  auto entry = dimensions.find(degrees[0]);
  assert(entry != dimensions.end());
  for (auto state = 0; state < entry->second; state++) {
    states.push_back(std::to_string(state));
  }

  for (auto idx = 1; idx < degrees.size(); ++idx) {
    auto entry = dimensions.find(degrees[idx]);
    assert(entry != dimensions.end());
    std::vector<std::string> result;
    for (auto current : states) {
      for (auto state = 0; state < entry->second; state++) {
        result.push_back(current + std::to_string(state));
      }
    }
    states = result;
  }

  return states;
}

void permute_matrix(cudaq::matrix_2 &matrix,
                    const std::vector<int> &permutation) {
  std::vector<std::complex<double>> sorted_values;
  for (std::size_t permuted : permutation) {
    for (std::size_t permuted_again : permutation) {
      sorted_values.push_back(matrix[{permuted, permuted_again}]);
    }
  }
  int idx = 0;
  for (std::size_t row = 0; row < matrix.get_rows(); row++) {
    for (std::size_t col = 0; col < matrix.get_columns(); col++) {
      matrix[{row, col}] = sorted_values[idx];
      idx++;
    }
  }
}

void canonicalize_degrees(std::vector<int> &degrees) {
  std::sort(degrees.begin(), degrees.end(), std::greater<int>());
}
} // namespace detail
} // namespace cudaq
