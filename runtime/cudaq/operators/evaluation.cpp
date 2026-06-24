/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "evaluation.h"

namespace cudaq {

// operator_arithmetics<matrix_evaluation>

void operator_arithmetics<operator_handler::matrix_evaluation>::canonicalize(
    complex_matrix &matrix, std::vector<std::size_t> &degrees) const {
  auto current_degrees = degrees;
  std::sort(degrees.begin(), degrees.end(), operator_handler::canonical_order);
  if (current_degrees != degrees) {
    auto permutation = cudaq::detail::compute_permutation(
        current_degrees, degrees, this->dimensions);
    cudaq::detail::permute_matrix(matrix, permutation);
  }
}

operator_handler::matrix_evaluation
operator_arithmetics<operator_handler::matrix_evaluation>::evaluate(
    const operator_handler &op) {
  return operator_handler::matrix_evaluation(
      op.degrees(), op.to_matrix(this->dimensions, this->parameters));
}

operator_handler::matrix_evaluation
operator_arithmetics<operator_handler::matrix_evaluation>::evaluate(
    const scalar_operator &op) const {
  return operator_handler::matrix_evaluation({},
                                             op.to_matrix(this->parameters));
}

operator_handler::matrix_evaluation
operator_arithmetics<operator_handler::matrix_evaluation>::tensor(
    operator_handler::matrix_evaluation &&op1,
    operator_handler::matrix_evaluation &&op2) const {
  op1.degrees.reserve(op1.degrees.size() + op2.degrees.size());
  for (auto d : op2.degrees) {
    assert(std::find(op1.degrees.cbegin(), op1.degrees.cend(), d) ==
           op1.degrees.cend());
    op1.degrees.push_back(d);
  }

  auto matrix = // matrix order needs to be reversed to be consistent
      cudaq::kronecker(std::move(op2.matrix), std::move(op1.matrix));
  this->canonicalize(matrix, op1.degrees);
  return operator_handler::matrix_evaluation(std::move(op1.degrees),
                                             std::move(matrix));
}

operator_handler::matrix_evaluation
operator_arithmetics<operator_handler::matrix_evaluation>::mul(
    const scalar_operator &scalar,
    operator_handler::matrix_evaluation &&op) const {
  auto matrix = scalar.evaluate(this->parameters) * std::move(op.matrix);
  return operator_handler::matrix_evaluation(std::move(op.degrees),
                                             std::move(matrix));
}

operator_handler::matrix_evaluation
operator_arithmetics<operator_handler::matrix_evaluation>::mul(
    operator_handler::matrix_evaluation &&op1,
    operator_handler::matrix_evaluation &&op2) const {
  // Elementary operators have sorted degrees such that we have a unique
  // convention for how to define the matrix. Tensor products permute the
  // computed matrix if necessary to guarantee that all operators always have
  // sorted degrees.
  assert(op1.degrees == op2.degrees);
  op1.matrix *= std::move(op2.matrix);
  return operator_handler::matrix_evaluation(std::move(op1.degrees),
                                             std::move(op1.matrix));
}

operator_handler::matrix_evaluation
operator_arithmetics<operator_handler::matrix_evaluation>::add(
    operator_handler::matrix_evaluation &&op1,
    operator_handler::matrix_evaluation &&op2) const {
  // Elementary operators have sorted degrees such that we have a unique
  // convention for how to define the matrix. Tensor products permute the
  // computed matrix if necessary to guarantee that all operators always have
  // sorted degrees.
  assert(op1.degrees == op2.degrees);
  op1.matrix += std::move(op2.matrix);
  return operator_handler::matrix_evaluation(std::move(op1.degrees),
                                             std::move(op1.matrix));
}

// operator_arithmetics<canonical_evaluation>

operator_handler::canonical_evaluation
operator_arithmetics<operator_handler::canonical_evaluation>::evaluate(
    const operator_handler &op) {
  std::vector<int64_t> relevant_dims;
  auto canon_str = op.canonical_form(this->dimensions, relevant_dims);
  return operator_handler::canonical_evaluation(std::move(canon_str),
                                                std::move(relevant_dims));
}

operator_handler::canonical_evaluation
operator_arithmetics<operator_handler::canonical_evaluation>::evaluate(
    const scalar_operator &scalar) const {
  return operator_handler::canonical_evaluation(
      scalar.evaluate(this->parameters));
}

operator_handler::canonical_evaluation
operator_arithmetics<operator_handler::canonical_evaluation>::tensor(
    operator_handler::canonical_evaluation &&val1,
    operator_handler::canonical_evaluation &&val2) const {
  assert(val1.terms.size() == 1 && val2.terms.size() == 1);
  assert(val2.terms[0].coefficient ==
         std::complex<double>(1.)); // should be trivial
  val1.push_back(val2.terms[0].encoding, val2.terms[0].relevant_dimensions);
  return std::move(val1);
}

operator_handler::canonical_evaluation
operator_arithmetics<operator_handler::canonical_evaluation>::mul(
    const scalar_operator &scalar,
    operator_handler::canonical_evaluation &&op) const {
  throw std::runtime_error(
      "multiplication should never be called on canonicalized operator - "
      "product padding is disabled");
}

operator_handler::canonical_evaluation
operator_arithmetics<operator_handler::canonical_evaluation>::mul(
    operator_handler::canonical_evaluation &&val1,
    operator_handler::canonical_evaluation &&val2) const {
  throw std::runtime_error(
      "multiplication should never be called on canonicalized operator - "
      "product padding is disabled");
}

operator_handler::canonical_evaluation
operator_arithmetics<operator_handler::canonical_evaluation>::add(
    operator_handler::canonical_evaluation &&val1,
    operator_handler::canonical_evaluation &&val2) const {
  assert(val2.terms.size() == 1);
  val1.terms.push_back(std::move(val2.terms[0]));
  return std::move(val1);
}

} // namespace cudaq
