/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <set>
#include <unordered_map>

#include "helpers.h"
#include "manipulation.h"

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

MatrixArithmetics::MatrixArithmetics(
                  std::unordered_map<int, int> &dimensions,
                  const std::unordered_map<std::string, std::complex<double>> &parameters)
  : m_dimensions(dimensions), m_parameters(parameters) {}

std::vector<int>
MatrixArithmetics::compute_permutation(const std::vector<int> &op_degrees,
                                        const std::vector<int> &canon_degrees) {
  assert(op_degrees.size() == canon_degrees.size());
  auto states = cudaq::detail::generate_all_states(canon_degrees, m_dimensions);

  std::vector<int> reordering;
  for (auto degree : op_degrees) {
    auto it = std::find(canon_degrees.cbegin(), canon_degrees.cend(), degree);
    reordering.push_back(it - canon_degrees.cbegin());
  }

  std::vector<std::string> op_states =
      cudaq::detail::generate_all_states(op_degrees, m_dimensions);

  std::vector<int> permutation;
  for (auto state : states) {
    std::string term;
    for (auto i : reordering) {
      term += state[i];
    }
    auto it = std::find(op_states.cbegin(), op_states.cend(), term);
    permutation.push_back(it - op_states.cbegin());
  }

  return permutation;
}

// Given a matrix representation that acts on the given degrees or freedom,
// sorts the degrees and permutes the matrix to match that canonical order.
// Returns:
//     A tuple consisting of the permuted matrix as well as the sequence of
//     degrees of freedom in canonical order.
void MatrixArithmetics::canonicalize(matrix_2 &matrix, std::vector<int> &degrees) {
  auto current_degrees = degrees;
  cudaq::detail::canonicalize_degrees(degrees);
  if (current_degrees != degrees) {
    auto permutation = this->compute_permutation(current_degrees, degrees);
    cudaq::detail::permute_matrix(matrix, permutation);   
  }
}

EvaluatedMatrix MatrixArithmetics::tensor(EvaluatedMatrix op1,
                                          EvaluatedMatrix op2) {
  std::vector<int> degrees;
  auto op1_degrees = op1.degrees();
  auto op2_degrees = op2.degrees();
  degrees.reserve(op1_degrees.size() + op2_degrees.size());
  for (auto d : op1_degrees)
    degrees.push_back(d);
  for (auto d : op2_degrees) {
    assert(std::find(degrees.cbegin(), degrees.cend(), d) == degrees.cend());
    degrees.push_back(d);
  }
  auto matrix = cudaq::kronecker(op1.matrix(), op2.matrix());
  this->canonicalize(matrix, degrees);
  return EvaluatedMatrix(std::move(degrees), std::move(matrix));
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