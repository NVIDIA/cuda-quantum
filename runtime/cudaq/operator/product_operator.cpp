/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "operators.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>

namespace cudaq {

/// Product Operator constructors.
product_operator::product_operator(
    std::vector<std::variant<scalar_operator, elementary_operator>>
        atomic_operators)
    : m_terms(atomic_operators) {}

complex_matrix kroneckerHelper(std::vector<complex_matrix> &matrices) {
  // essentially we pass in the list of elementary operators to
  // this function -- with lowest degree being leftmost -- then it computes the
  // kronecker product of all of them.
  auto kronecker = [](complex_matrix &self, complex_matrix &other) {
    return self.kronecker(other);
  };

  return std::accumulate(begin(matrices), end(matrices),
                         complex_matrix::identity(1, 1), kronecker);
}

/// Convert the product_operator to a matrix representation.
complex_matrix product_operator::to_matrix(
    const std::map<int, int> dimensions,
    const std::map<std::string, std::complex<double>> parameters) const {
  std::vector<complex_matrix> matricesFullVectorSpace;

  for (auto &term : m_terms) {
    std::vector<int> op_degrees;
    complex_matrix operator_matrix;

    // Retrieve degrees and the matrix for the current operator
    std::visit(
        [&](auto &&op) {
          op_degrees = op.degrees;
          operator_matrix = op.to_matrix(dimensions, parameters);
        },
        term);

    std::vector<complex_matrix> matrixWithIdentities;

    // Add identity matrices for missing degrees of freedom
    for (const auto &[degree, level] : dimensions) {
      if (std::find(op_degrees.begin(), op_degrees.end(), degree) ==
          op_degrees.end()) {
        matrixWithIdentities.push_back(complex_matrix::identity(level, level));
      } else {
        matrixWithIdentities.push_back(operator_matrix);
      }
    }

    // Compute the Kronecker product for this term
    matricesFullVectorSpace.push_back(kroneckerHelper(matrixWithIdentities));
  }

  // Sum all Kronecker product to form the final matrix
  if (matricesFullVectorSpace.empty()) {
    throw std::runtime_error(
        "No matrices to sum in product_operator::to_matrix.");
  }

  complex_matrix result(dimensions.begin()->second, dimensions.end()->second);
  result.set_zero();

  for (const auto &matrix : matricesFullVectorSpace) {
    result = result + matrix;
  }

  return result;
}

// Degrees property
std::vector<int> product_operator::degrees() const {
  std::set<int> unique_degrees;
  // The variant type makes it difficult
  auto beginFunc = [](auto &&t) { return t.degrees.begin(); };
  auto endFunc = [](auto &&t) { return t.degrees.end(); };
  for (const auto &term : m_terms) {
    unique_degrees.insert(std::visit(beginFunc, term),
                          std::visit(endFunc, term));
  }
  // Erase any `-1` degree values that may have come from scalar operators.
  auto it = unique_degrees.find(-1);
  if (it != unique_degrees.end()) {
    unique_degrees.erase(it);
  }
  return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
}

template <typename TEval>
TEval elementary_operator::_evaluate(
    operator_arithmetics<TEval> &arithmetics) const {
  std::cout << "In ProductOp _evaluate" << std::endl;
  return arithmetics.evaluate(*this);
}

operator_sum product_operator::operator+(const product_operator &other) const {
  return operator_sum({*this, other});
}

operator_sum product_operator::operator-(const product_operator &other) const {
  return *this + (-1.0 * other);
}

product_operator
product_operator::operator*(const product_operator &other) const {
  std::vector<std::variant<scalar_operator, elementary_operator>>
      combined_terms = m_terms;
  combined_terms.insert(combined_terms.end(), other.m_terms.begin(),
                        other.m_terms.end());
  return product_operator(combined_terms);
}

product_operator
product_operator::operator/(const std::complex<double> scalar) const {
  return *this * (1.0 / scalar);
}

product_operator product_operator::_from_word(const std::string &word) {
  if (word.empty()) {
    throw std::invalid_argument("Empty Pauli word!");
  }

  std::vector<std::variant<scalar_operator, elementary_operator>> ops;

  for (size_t i = 0; i < word.size(); i++) {
    char c = std::tolower(word[i]);
    switch (c) {
    case 'x':
      ops.push_back(elementary_operator("pauli_x", {static_cast<int>(i)}));
      break;
    case 'y':
      ops.push_back(elementary_operator("pauli_y", {static_cast<int>(i)}));
      break;
    case 'z':
      ops.push_back(elementary_operator("pauli_z", {static_cast<int>(i)}));
      break;
    case 'i':
      ops.push_back(elementary_operator::identity(static_cast<int>(i)));
      break;
    default:
      throw std::invalid_argument("Invalid character in Pauli word: " + c);
    }
  }

  return product_operator(ops);
}

operator_sum product_operator::operator+(const scalar_operator &other) const {
  return operator_sum({*this, product_operator({other})});
}

operator_sum product_operator::operator-(const scalar_operator &other) const {
  return *this + (-1.0 * other);
}

product_operator
product_operator::operator*(const scalar_operator &other) const {
  std::vector<std::variant<scalar_operator, elementary_operator>>
      combined_terms = m_terms;
  combined_terms.insert(combined_terms.begin(), other);
  return product_operator(combined_terms);
}

product_operator &product_operator::operator*=(const scalar_operator &other) {
  m_terms.insert(m_terms.begin(), other);
  return *this;
}

// product_operator product_operator::operator*=(std::complex<double> other);
// operator_sum product_operator::operator+(operator_sum other);
// operator_sum product_operator::operator-(operator_sum other);
// product_operator product_operator::operator*(operator_sum other);
// product_operator product_operator::operator*=(operator_sum other);
// operator_sum product_operator::operator-(scalar_operator other);
// product_operator product_operator::operator*(scalar_operator other);
// product_operator product_operator::operator*=(scalar_operator other);
// operator_sum product_operator::operator+(product_operator other);
// operator_sum product_operator::operator-(product_operator other);
// product_operator product_operator::operator*(product_operator other);
// product_operator product_operator::operator*=(product_operator other);
// operator_sum product_operator::operator+(elementary_operator other);
// operator_sum product_operator::operator-(elementary_operator other);
// product_operator product_operator::operator*(elementary_operator other);
// product_operator product_operator::operator*=(elementary_operator other);

} // namespace cudaq