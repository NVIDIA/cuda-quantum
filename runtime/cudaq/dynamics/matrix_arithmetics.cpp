/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cassert>
#include <matrix_arithmetics.h>
#include <operators.h>
#include <unsupported/Eigen/KroneckerProduct>

namespace cudaq {
Evaluated::Evaluated(const std::vector<int> &degrees,
                     const cudaq::complex_matrix &matrix)
    : _degrees(degrees), _matrix(matrix) {}

const std::vector<int> &Evaluated::get_degrees() const { return _degrees; }

const cudaq::complex_matrix &Evaluated::get_matrix() const { return _matrix; }

matrix_arithmetics::matrix_arithmetics(const std::map<int, int> &dimensions)
    : dimensions(dimensions) {}

std::vector<std::string>
generate_all_states(const std::vector<int> &degrees,
                    const std::map<int, int> &dimensions) {
  // Determine total combinations based on degrees dimensions
  int total_states = 1;
  for (int deg : degrees) {
    total_states *= dimensions.at(deg);
  }

  std::vector<std::string> states(total_states);
  int num_states = total_states;

  for (size_t i = 0; i < degrees.size(); i++) {
    int dim = dimensions.at(degrees[i]);
    int repeats = num_states / dim;
    int index = 0;

    for (int j = 0; j < total_states; j++) {
      states[j] += std::to_string(index);
      if ((j + 1) % repeats == 0) {
        index = (index + 1) % dim;
      }
    }
  }

  return states;
}

std::vector<int>
matrix_arithmetics::compute_permutation(const std::vector<int> &op_degrees,
                                        const std::vector<int> &canon_degrees) {
  // Generate all states for canonical degrees
  std::vector<std::string> states =
      generate_all_states(canon_degrees, dimensions);

  // Determine the reordering indices
  std::vector<int> reordering;
  for (int deg : op_degrees) {
    auto it = std::find(canon_degrees.begin(), canon_degrees.end(), deg);
    if (it != canon_degrees.end()) {
      reordering.push_back(std::distance(canon_degrees.begin(), it));
    }
  }

  // Generate the permutation vector
  std::vector<int> permutation;
  for (const auto &state : states) {
    std::string reordered_state;
    for (int index : reordering) {
      reordered_state += state[index];
    }

    auto perm_index = std::find(states.begin(), states.end(), reordered_state);
    permutation.push_back(std::distance(states.begin(), perm_index));
  }
  return permutation;
}

std::pair<cudaq::complex_matrix, std::vector<int>>
matrix_arithmetics::canonicalize(const cudaq::complex_matrix &op_matrix,
                                 const std::vector<int> &op_degrees) {
  std::vector<int> canon_degrees = op_degrees;
  std::sort(canon_degrees.begin(), canon_degrees.end());

  if (op_degrees == canon_degrees) {
    return std::make_pair(op_matrix, canon_degrees);
  }

  // Compute permutation
  std::vector<int> permutation = compute_permutation(op_degrees, canon_degrees);

  cudaq::complex_matrix permuted_matrix = op_matrix;
  for (int i = 0; i < permutation.size(); i++) {
    auto row = op_matrix.get_row(permutation[i]);
    permuted_matrix.set_row(i, row);
  }

  return std::make_pair(permuted_matrix, canon_degrees);
}

Evaluated matrix_arithmetics::evaluate(const Operator &op) const {
  // cudaq::complex_matrix matrix = op.to_matrix(dimensions, op.parameters);
  // // Replace with degrees from operator
  // std::vector<int> degrees = op.degrees;
  // return Evaluated(degrees, matrix);

  return std::visit(
      [this](auto &&arg) -> Evaluated {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, cudaq::elementary_operator>) {
          cudaq::complex_matrix matrix = arg.to_matrix(dimensions, {});
          std::vector<int> degrees = arg.degrees;
          return Evaluated(degrees, matrix);
        } else if constexpr (std::is_same_v<T, cudaq::scalar_operator>) {
          // Handle the scalar operator
          if (arg.has_val()) {
            // double scalar_value = arg.get_val();
            int dim = dimensions.at(arg.degrees[0]);
            cudaq::complex_matrix matrix =
                cudaq::complex_matrix::identity(dim, dim);
            return Evaluated(arg.degrees, matrix);
          } else {
            throw std::runtime_error(
                "Dynamic scalar operator evaluation not implemented.");
          }
        } else {
          throw std::runtime_error(
              "Unknown operator type in CudaqOperatorVariant.");
        }
      },
      op);
}

// std::shared_ptr<matrix_arithmetics::Evaluated> matrix_arithmetics::add(const
// std::shared_ptr<Evaluated> &op1, const std::shared_ptr<Evaluated> &op2) {
//     assert(op1.degrees() == op2.degrees());
//     Matrix result = op1.matrix() + op2.matrix();
//     return std::make_shared<Evaluated>(op1.degrees(), result);
// }

// matrix_arithmetics::Evaluated matrix_arithmetics::mul(const Evaluated &op1,
// const Evaluated &op2) {
//     assert(op1.degrees() == op2.degrees());
//     Matrix result = op1.matrix() * op2.matrix();
//     return Evaluated(op1.degrees(), result);
// }

// matrix_arithmetics::Evaluated matrix_arithmetics::tensor(const Evaluated
// &op1, const Evaluated &op2) {
//     std::vector<int> new_degrees(op1.degrees());
//     new_degrees.insert(new_degrees.end(), op2.degrees().begin(),
//     op2.degrees().end());

//     Matrix result = Eigen::kroneckerProduct(op1.matrix(),
//     op2.matrix()).eval(); auto [canonical_matrix, canonical_degrees] =
//     canonicalize(result, new_degrees); return Evaluated(new_degrees, result);
// }

} // namespace cudaq