/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
      std::unordered_map<std::size_t, std::int64_t> &dimensions,
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
  std::unordered_map<std::size_t, std::int64_t>
      &dimensions; // may be updated during evaluation
  const std::unordered_map<std::string, std::complex<double>> &parameters;

  // Given a matrix representation that acts on the given degrees or freedom,
  // sorts the degrees and permutes the matrix to match that canonical order.
  void canonicalize(complex_matrix &matrix,
                    std::vector<std::size_t> &degrees) const;

public:
  const bool pad_sum_terms = true;
  const bool pad_product_terms = true;

  constexpr operator_arithmetics(
      std::unordered_map<std::size_t, std::int64_t> &dimensions,
      const std::unordered_map<std::string, std::complex<double>> &parameters)
      : dimensions(dimensions), parameters(parameters) {}

  operator_handler::matrix_evaluation evaluate(const operator_handler &op);
  operator_handler::matrix_evaluation evaluate(const scalar_operator &op) const;

  operator_handler::matrix_evaluation
  tensor(operator_handler::matrix_evaluation &&op1,
         operator_handler::matrix_evaluation &&op2) const;

  operator_handler::matrix_evaluation
  mul(const scalar_operator &scalar,
      operator_handler::matrix_evaluation &&op) const;

  operator_handler::matrix_evaluation
  mul(operator_handler::matrix_evaluation &&op1,
      operator_handler::matrix_evaluation &&op2) const;

  operator_handler::matrix_evaluation
  add(operator_handler::matrix_evaluation &&op1,
      operator_handler::matrix_evaluation &&op2) const;
};

template <>
class operator_arithmetics<operator_handler::canonical_evaluation> {

private:
  std::unordered_map<std::size_t, std::int64_t>
      &dimensions; // may be updated during evaluation
  const std::unordered_map<std::string, std::complex<double>> &parameters;

public:
  const bool pad_sum_terms = true;
  const bool pad_product_terms = false;

  constexpr operator_arithmetics(
      std::unordered_map<std::size_t, std::int64_t> &dimensions,
      const std::unordered_map<std::string, std::complex<double>> &parameters)
      : dimensions(dimensions), parameters(parameters) {}

  operator_handler::canonical_evaluation evaluate(const operator_handler &op);

  operator_handler::canonical_evaluation
  evaluate(const scalar_operator &scalar) const;

  operator_handler::canonical_evaluation
  tensor(operator_handler::canonical_evaluation &&val1,
         operator_handler::canonical_evaluation &&val2) const;

  operator_handler::canonical_evaluation
  mul(const scalar_operator &scalar,
      operator_handler::canonical_evaluation &&op) const;

  operator_handler::canonical_evaluation
  mul(operator_handler::canonical_evaluation &&val1,
      operator_handler::canonical_evaluation &&val2) const;

  operator_handler::canonical_evaluation
  add(operator_handler::canonical_evaluation &&val1,
      operator_handler::canonical_evaluation &&val2) const;
};

} // namespace cudaq
