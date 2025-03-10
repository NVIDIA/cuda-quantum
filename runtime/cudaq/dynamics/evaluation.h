/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/matrix.h"
#include <algorithm>
#include <set>
#include <unordered_map>
#include <vector>

#include "helpers.h"
#include "operator_leafs.h"

namespace cudaq {

template <typename EvalTy>
class operator_arithmetics {
public:
  operator_arithmetics(
      std::unordered_map<int, int> &dimensions,
      const std::unordered_map<std::string, std::complex<double>> &parameters);

  /// Whether to inject tensor products with identity to each term in the
  /// sum to ensure that all terms are acting on the same degrees of freedom
  /// by the time they are added.
  const bool pad_sum_terms;
  /// Whether to inject tensor products with identity to each term in the
  /// product to ensure that each term has its full size by the time they
  /// are multiplied.
  const bool pad_product_terms;

  /// @brief Accesses the relevant data to evaluate an operator expression
  /// in the leaf nodes, that is in elementary and scalar operators.
  EvalTy evaluate(const operator_handler &op);
  EvalTy evaluate(const scalar_operator &op);

  /// @brief Computes the tensor product of two operators that act on different
  /// degrees of freedom.
  EvalTy tensor(EvalTy &&val1, EvalTy &&val2);

  /// @brief Multiplies two operators that act on the same degrees of freedom.
  EvalTy mul(EvalTy &&val1, EvalTy &&val2);

  /// @brief Multiplies an evaluated operator with a scalar.
  EvalTy mul(const scalar_operator &scalar, EvalTy &&op);

  /// @brief Adds two operators that act on the same degrees of freedom.
  EvalTy add(EvalTy &&val1, EvalTy &&val2);
};

template <>
class operator_arithmetics<operator_handler::matrix_evaluation> {

private:
  std::unordered_map<int, int> &dimensions; // may be updated during evaluation
  const std::unordered_map<std::string, std::complex<double>> &parameters;

  // Given a matrix representation that acts on the given degrees or freedom,
  // sorts the degrees and permutes the matrix to match that canonical order.
  void canonicalize(complex_matrix &matrix, std::vector<int> &degrees) {
    auto current_degrees = degrees;
    std::sort(degrees.begin(), degrees.end(),
              operator_handler::canonical_order);
    if (current_degrees != degrees) {
      auto permutation = cudaq::detail::compute_permutation(
          current_degrees, degrees, this->dimensions);
      cudaq::detail::permute_matrix(matrix, permutation);
    }
  }

public:
  const bool pad_sum_terms = true;
  const bool pad_product_terms = true;

  operator_arithmetics(
      std::unordered_map<int, int> &dimensions,
      const std::unordered_map<std::string, std::complex<double>> &parameters)
      : dimensions(dimensions), parameters(parameters) {}

  operator_handler::matrix_evaluation evaluate(const operator_handler &op) {
    return operator_handler::matrix_evaluation(
        op.degrees(), op.to_matrix(this->dimensions, this->parameters));
  }

  operator_handler::matrix_evaluation evaluate(const scalar_operator &op) {
    return operator_handler::matrix_evaluation({},
                                               op.to_matrix(this->parameters));
  }

  operator_handler::matrix_evaluation
  tensor(operator_handler::matrix_evaluation &&op1,
         operator_handler::matrix_evaluation &&op2) {
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
  mul(const scalar_operator &scalar, operator_handler::matrix_evaluation &&op) {
    auto matrix = scalar.evaluate(this->parameters) * std::move(op.matrix);
    return operator_handler::matrix_evaluation(std::move(op.degrees),
                                               std::move(matrix));
  }

  operator_handler::matrix_evaluation
  mul(operator_handler::matrix_evaluation &&op1,
      operator_handler::matrix_evaluation &&op2) {
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
  add(operator_handler::matrix_evaluation &&op1,
      operator_handler::matrix_evaluation &&op2) {
    // Elementary operators have sorted degrees such that we have a unique
    // convention for how to define the matrix. Tensor products permute the
    // computed matrix if necessary to guarantee that all operators always have
    // sorted degrees.
    assert(op1.degrees == op2.degrees);
    op1.matrix += std::move(op2.matrix);
    return operator_handler::matrix_evaluation(std::move(op1.degrees),
                                               std::move(op1.matrix));
  }
};

template <>
class operator_arithmetics<operator_handler::canonical_evaluation> {

private:
  std::unordered_map<int, int> &dimensions; // may be updated during evaluation
  const std::unordered_map<std::string, std::complex<double>> &parameters;

public:
  const bool pad_sum_terms = true;
  const bool pad_product_terms = false;

  operator_arithmetics(
      std::unordered_map<int, int> &dimensions,
      const std::unordered_map<std::string, std::complex<double>> &parameters)
      : dimensions(dimensions), parameters(parameters) {}

  operator_handler::canonical_evaluation evaluate(const operator_handler &op) {
    auto canon_str = op.op_code_to_string(this->dimensions);
    operator_handler::canonical_evaluation eval;
    eval.push_back(
        std::make_pair(std::complex<double>(1.), std::move(canon_str)));
    return eval;
  }

  operator_handler::canonical_evaluation
  evaluate(const scalar_operator &scalar) {
    operator_handler::canonical_evaluation eval;
    eval.push_back(std::make_pair(scalar.evaluate(this->parameters), ""));
    return eval;
  }

  operator_handler::canonical_evaluation
  tensor(operator_handler::canonical_evaluation &&val1,
         operator_handler::canonical_evaluation &&val2) {
    assert(val1.terms.size() == 1 && val2.terms.size() == 1);
    assert(val2.terms[0].first ==
           std::complex<double>(1.)); // should be trivial
    val1.push_back(val2.terms[0].second);
    return std::move(val1);
  }

  operator_handler::canonical_evaluation
  mul(const scalar_operator &scalar,
      operator_handler::canonical_evaluation &&op) {
    throw std::runtime_error(
        "multiplication should never be called on canonicalized operator - "
        "product padding is disabled");
  }

  operator_handler::canonical_evaluation
  mul(operator_handler::canonical_evaluation &&val1,
      operator_handler::canonical_evaluation &&val2) {
    throw std::runtime_error(
        "multiplication should never be called on canonicalized operator - "
        "product padding is disabled");
  }

  operator_handler::canonical_evaluation
  add(operator_handler::canonical_evaluation &&val1,
      operator_handler::canonical_evaluation &&val2) {
    assert(val2.terms.size() == 1);
    val1.push_back(std::move(val2.terms[0]));
    return std::move(val1);
  }
};

} // namespace cudaq