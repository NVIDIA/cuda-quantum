/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "manipulation.h"
#include "helpers.h"
#include <set>

namespace cudaq {

// EvaluatedMatrix class

const std::vector<int>& EvaluatedMatrix::degrees() const {
  return this->targets;
}

const matrix_2& EvaluatedMatrix::matrix() const {
  return this->value;
}

EvaluatedMatrix::EvaluatedMatrix(const std::vector<int> &degrees, const matrix_2 &matrix)
  : targets(degrees), value(matrix) {
#if !defined(NDEBUG)
    std::set<int> unique_degrees;
    for (auto d : degrees)
      unique_degrees.insert(d);
    assert(unique_degrees.size() == degrees.size());
#endif
  }

EvaluatedMatrix::EvaluatedMatrix(EvaluatedMatrix &&other)
  : targets(std::move(other.targets)), value(std::move(other.value)) {}

EvaluatedMatrix& EvaluatedMatrix::operator=(EvaluatedMatrix &&other) {
  if (this != &other) {
    this->targets = std::move(other.targets);
    this->value = std::move(other.value);
  }
  return *this;
}

// MatrixArithmetics

MatrixArithmetics::MatrixArithmetics(std::map<int, int> dimensions,
                  std::map<std::string, std::complex<double>> parameters)
  : m_dimensions(dimensions), m_parameters(parameters) {}

std::vector<int>
MatrixArithmetics::_compute_permutation(std::vector<int> op_degrees,
                                        std::vector<int> canon_degrees) {
  auto states = cudaq::detail::generate_all_states(canon_degrees, m_dimensions);

  std::vector<int> reordering;
  for (auto degree : op_degrees) {
    auto it = std::find(canon_degrees.begin(), canon_degrees.end(), degree);
    reordering.push_back(it - canon_degrees.begin());
  }

  std::vector<std::string> op_states =
      cudaq::detail::generate_all_states(op_degrees, m_dimensions);

  std::vector<int> permutation;
  for (auto state : states) {
    std::string term;
    for (auto i : reordering) {
      term += state[i];
    }
    auto it = std::find(op_states.begin(), op_states.end(), term);
    permutation.push_back(it - op_states.begin());
  }

  return permutation;
}

// Given a matrix representation that acts on the given degrees or freedom,
// sorts the degrees and permutes the matrix to match that canonical order.
// Returns:
//     A tuple consisting of the permuted matrix as well as the sequence of
//     degrees of freedom in canonical order.
std::tuple<matrix_2, std::vector<int>>
MatrixArithmetics::_canonicalize(matrix_2 &op_matrix,
                                 std::vector<int> op_degrees) {
  auto canon_degrees = cudaq::detail::canonicalize_degrees(op_degrees);
  if (op_degrees == canon_degrees)
    return std::tuple<matrix_2, std::vector<int>>{op_matrix, canon_degrees};

  auto permutation = this->_compute_permutation(op_degrees, canon_degrees);
  auto result = cudaq::detail::permute_matrix(op_matrix, permutation);
  return std::tuple<matrix_2, std::vector<int>>{result, canon_degrees};
}

EvaluatedMatrix MatrixArithmetics::tensor(EvaluatedMatrix op1,
                                          EvaluatedMatrix op2) {
  std::vector<int> op_degrees;
  auto op1_degrees = op1.degrees();
  auto op2_degrees = op2.degrees();
  op_degrees.reserve(op1_degrees.size() + op2_degrees.size());
  for (auto d : op1_degrees)
    op_degrees.push_back(d);
  for (auto d : op2_degrees) {
    assert(std::find(op_degrees.begin(), op_degrees.end(), d) == op_degrees.end());
    op_degrees.push_back(d);
  }
  auto op_matrix = cudaq::kronecker(op1.matrix(), op2.matrix());
  auto [new_matrix, new_degrees] = this->_canonicalize(op_matrix, op_degrees);
  return EvaluatedMatrix(new_degrees, new_matrix);
}

EvaluatedMatrix MatrixArithmetics::mul(EvaluatedMatrix op1,
                                       EvaluatedMatrix op2) {
  // Elementary operators have sorted degrees such that we have a unique
  // convention for how to define the matrix. Tensor products permute the
  // computed matrix if necessary to guarantee that all operators always have
  // sorted degrees.
  auto degrees = op1.degrees();
  assert(degrees == op2.degrees());
  return EvaluatedMatrix(std::move(degrees), (op1.matrix() * op2.matrix()));
}

EvaluatedMatrix MatrixArithmetics::add(EvaluatedMatrix op1,
                                       EvaluatedMatrix op2) {
  // Elementary operators have sorted degrees such that we have a unique
  // convention for how to define the matrix. Tensor products permute the
  // computed matrix if necessary to guarantee that all operators always have
  // sorted degrees.
  auto degrees = op1.degrees();
  assert(degrees == op2.degrees());
  return EvaluatedMatrix(std::move(degrees), op1.matrix() + op2.matrix());
}

} // namespace cudaq