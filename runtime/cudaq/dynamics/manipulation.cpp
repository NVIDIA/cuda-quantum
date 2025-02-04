/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

<<<<<<< HEAD
#include "cudaq/helpers.h"
#include "cudaq/operators.h"
=======
#include "manipulation.h"
#include "helpers.h"
>>>>>>> 558b2496f (general clean up)

namespace cudaq {

// EvaluatedMatrix class

const std::vector<int>& EvaluatedMatrix::degrees() const {
  return this->targets;
}

const matrix_2& EvaluatedMatrix::matrix() const {
  return this->value;
}

EvaluatedMatrix::EvaluatedMatrix(const std::vector<int> &degrees, const matrix_2 &matrix)
  : targets(degrees), value(matrix) {}

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
  /// FIXME: do this check:
  // assert len(frozenset(op1.degrees).intersection(op2.degrees)) == 0, \
  //     "Operators should not have common degrees of freedom."

  std::vector<int> op_degrees;
  op_degrees.reserve(op1.degrees().size() + op2.degrees().size());
  for (auto d : op1.degrees())
    op_degrees.push_back(d);
  for (auto d : op2.degrees())
    op_degrees.push_back(d);
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
  if (op1.degrees() != op2.degrees())
    throw std::runtime_error(
        "Operators should have the same order of degrees.");
  return EvaluatedMatrix(op1.degrees(), (op1.matrix() * op2.matrix()));
}

EvaluatedMatrix MatrixArithmetics::add(EvaluatedMatrix op1,
                                       EvaluatedMatrix op2) {
  // Elementary operators have sorted degrees such that we have a unique
  // convention for how to define the matrix. Tensor products permute the
  // computed matrix if necessary to guarantee that all operators always have
  // sorted degrees.
  if (op1.degrees() != op2.degrees())
    throw std::runtime_error(
        "Operators should have the same order of degrees.");
  return EvaluatedMatrix(op1.degrees(), (op1.matrix() + op2.matrix()));
}

} // namespace cudaq