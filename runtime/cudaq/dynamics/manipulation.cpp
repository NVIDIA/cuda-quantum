/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "helpers.cpp"

namespace cudaq {

std::vector<int>
MatrixArithmetics::_compute_permutation(std::vector<int> op_degrees,
                                        std::vector<int> canon_degrees) {
  auto states =
      _OperatorHelpers::generate_all_states(canon_degrees, m_dimensions);

  std::vector<int> reordering;
  for (auto degree : op_degrees)
    reordering.push_back(canon_degrees[degree]);

  std::vector<int> result;
  for (auto state : states) {
    int index;
    std::string term;
    for (auto i : reordering) {
      term += state[i];
    }
    auto it = std::find(states.begin(), states.end(), term);
    if (it != states.end())
      index = std::distance(states.begin(), it);
    result.push_back(index);
  }

  return result;
}

// Given a matrix representation that acts on the given degrees or freedom,
// sorts the degrees and permutes the matrix to match that canonical order.
// Returns:
//     A tuple consisting of the permuted matrix as well as the sequence of
//     degrees of freedom in canonical order.
std::tuple<matrix_2, std::vector<int>>
MatrixArithmetics::_canonicalize(matrix_2 &op_matrix,
                                 std::vector<int> op_degrees) {
  auto canon_degrees = _OperatorHelpers::canonicalize_degrees(op_degrees);
  if (op_degrees == canon_degrees)
    return std::tuple<matrix_2, std::vector<int>>{op_matrix, canon_degrees};

  auto permutation = this->_compute_permutation(op_degrees, canon_degrees);
  auto result = _OperatorHelpers::permute_matrix(op_matrix, permutation);
  return std::tuple<matrix_2, std::vector<int>>{result, canon_degrees};
}

EvaluatedMatrix MatrixArithmetics::tensor(EvaluatedMatrix op1,
                                          EvaluatedMatrix op2) {
  /// FIXME: do this check:
  // assert len(frozenset(op1.degrees).intersection(op2.degrees)) == 0, \
  //     "Operators should not have common degrees of freedom."

  auto op_degrees = op1.m_degrees;
  std::copy(op2.m_degrees.begin(), op2.m_degrees.end(),
            back_inserter(op_degrees));
  auto op_matrix = cudaq::kronecker(op1.m_matrix, op2.m_matrix);
  auto [new_matrix, new_degrees] = this->_canonicalize(op_matrix, op_degrees);
  return EvaluatedMatrix(new_degrees, new_matrix);
}

EvaluatedMatrix MatrixArithmetics::mul(EvaluatedMatrix op1,
                                       EvaluatedMatrix op2) {
  // Elementary operators have sorted degrees such that we have a unique
  // convention for how to define the matrix. Tensor products permute the
  // computed matrix if necessary to guarantee that all operators always have
  // sorted degrees.
  if (op1.m_degrees != op2.m_degrees)
    throw std::runtime_error(
        "Operators should have the same order of degrees.");
  return EvaluatedMatrix(op1.m_degrees, (op1.m_matrix * op2.m_matrix));
}

EvaluatedMatrix MatrixArithmetics::add(EvaluatedMatrix op1,
                                       EvaluatedMatrix op2) {
  // Elementary operators have sorted degrees such that we have a unique
  // convention for how to define the matrix. Tensor products permute the
  // computed matrix if necessary to guarantee that all operators always have
  // sorted degrees.
  if (op1.m_degrees != op2.m_degrees)
    throw std::runtime_error(
        "Operators should have the same order of degrees.");
  return EvaluatedMatrix(op1.m_degrees, (op1.m_matrix + op2.m_matrix));
}

EvaluatedMatrix MatrixArithmetics::evaluate(
    std::variant<scalar_operator, matrix_operator, product_operator<matrix_operator>> op) {
  // auto getDegrees = [](auto &&t) { return t.degrees; };
  // auto toMatrix = [&](auto &&t) {
  //   return t.to_matrix(this->m_dimensions, this->m_parameters);
  // };
  // auto degrees = std::visit(getDegrees, op);
  // auto matrix = std::visit(toMatrix, op);
  // return EvaluatedMatrix(degrees, matrix);
  throw std::runtime_error("implementation broken.");
}

} // namespace cudaq