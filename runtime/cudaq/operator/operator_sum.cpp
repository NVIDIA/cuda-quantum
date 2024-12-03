/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "operator_arithmetics.h"
#include "operators.h"

#include <iostream>
#include <set>

namespace cudaq {

/// Operator sum constructor given a vector of product operators.
operator_sum::operator_sum(const std::vector<product_operator> &terms)
    : m_terms(terms) {}

/// Canonicalize the terms of the operator sum.
std::vector<sdt::variant<scalar_operator, elementary_operator>>
operator_sum::canonical_terms() const {
  std::vector<sdt::variant<scalar_operator, elementary_operator>>
      canonicalized_terms;

  for (const auto &term : m_terms) {
    std::vector<sdt::variant<scalar_operator, elementary_operator>> term_ops =
        term.get_terms();
    std::vector<scalar_operator> scalars;
    std::vector<elementary_operator> non_scalars;

    for (const auto &op_variant : term_ops) {
      std::visit(
          [&scalars, &non_scalars](auto &op) {
            if constexpr (std::is_same_v<decltype(op), scalar_operator>) {
              scalars.push_back(op);
            } else if constexpr (std::is_same_v<decltype(op),
                                                elementary_operator>) {
              non_scalars.push_back(op);
            }
          },
          op_variant);
    }

    // Sort scalars and non-scalars for canonical order
    std::sort(non_scalars.begin(), non_scalars.end(),
              [](const elementary_operator &a, const elementary_operator &b) {
                return a.degrees < b.degrees;
              });

    if (scalars.empty()) {
      scalars.push_back(scalar_operator(1.0));
    }

    for (const auto &scalar : scalars) {
      for (const auto &non_scalar : non_scalars) {
        canonicalized_terms.emplace_back(scalar, non_scalar);
      }
    }
  }

  return canonicalized_terms;
}

/// Canonicalize the operator sum into a new instance
operator_sum operator_sum::canonicalize() const {
  std::vector<product_operator> canonical_terms;
  for (const auto &[scalar, elementary] : canonical_terms()) {
    canonical_terms.emplace_back(product_operator({scalar, elementary}));
  }

  return operator_sum(canonical_terms);
}

/// Equality operator to compare canonicalized terms.
bool operator_sum::operator==(const operator_sum &other) const {
  return canonical_terms() == other.canonical_terms;
}

/// Get unique degrees of freedom in canonical order.
std::vector<int> operator_sum::degrees() const {
  std::set<int> unique_degrees;
  for (const auto &term : m_terms) {
    for (const auto &op_variant : term.get_terms()) {
      std::visit(
          [&unique_degrees](auto &op) {
            unique_degrees.insert(op.degrees.begin(), op.degrees.end());
          },
          op_variant);
    }
  }

  return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
}

/// Check if the operator sum acts as a spin operator.
bool operator_sum::_is_spinop() const {
  return std::all_of(
      m_terms.begin(), m_terms.end(), [](const product_operator &term) {
        return std::all_of(term.get_terms().begin(), term.get_terms().end(),
                           [](const auto &op_variant) {
                             return std::visit(
                                 [](auto &op) { return op._is_spinop(); },
                                 op_variant);
                           });
      });
}

/// Evaluate the opetrator sum using provided arithmetic rules.
template <typename TEval>
TEval operator_sum::_evaluate(operator_arithmetics<TEval> &arithmetics) const {
  // Collect all degrees of freedom from the terms
  std::set<int> degrees;
  for (const auto &term : m_terms) {
    for (auto &op_variant : term.get_terms()) {
      std::visit(
          [&degrees](auto &op) {
            degrees.insert(op.degrees.begin(), op.degrees.end());
          },
          op_variant);
    }
  }

  // Function to pad a product operator with identities for missing degrees
  auto padded_term = [&](product_operator term) -> product_operator {
    std::set<int> term_degrees;
    for (const auto &op_variant : term.get_terms()) {
      std::visit(
          [&term_degrees](auto &op) {
            term.degrees.insert(op.degrees.begin(), op.degrees.end());
          },
          op_variant);
    }

    // Add identity operators for missing degrees
    for (int degree : degrees) {
      if (term_degrees.find(degree) == term_degrees.end()) {
        term *= elementary_operator::identity(degree);
      }
    }
    return term;
  };

  // Evaluate the first term and initialize the sum
  TEval sum = padded_term(m_terms[0])._evaluate(arithmetics);

  // Evaluate remaining terms and add them to the sum
  for (size_t i = 1; i < m_terms.size(); i++) {
    sum = arithmetics.add(sum, padded_term(m_terms[i])._evaluate(arithmetics));
  }

  return sum;
}

// Arithmetic operators
operator_sum operator_sum::operator+(const operator_sum &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.insert(combined_terms.end(), other.m_terms.begin(),
                        other.m_terms.end());
  return operator_sum(combined_terms).canonicalize();
}

/// FIXME:
// operator_sum operator_sum::operator-(const operator_sum &other) const {
//   return *this + (-1 * other);
// }

/// FIXME:
// operator_sum operator_sum::operator-=(const operator_sum &other) {
//   *this = *this - other;
//   return *this;
// }

operator_sum operator_sum::operator+=(const operator_sum &other) {
  *this = *this + other;
  return *this;
}

/// FIXME:
// operator_sum operator_sum::operator*(const operator_sum &other) const {
//   std::vector<product_operator> product_terms;
//   for (const auto &self_term : m_terms) {
//     for (const auto &other_term : other.m_terms) {
//       product_terms.push_back(self_term * other_term);
//     }
//   }
// return operator_sum(product_terms);
// }

operator_sum operator_sum::operator+(const scalar_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.push_back(product_operator({other}));
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator-(const scalar_operator &other) const {
  return *this + (-1.0 * other);
}

operator_sum operator_sum::operator+=(const scalar_operator &other) {
  *this = *this + other;
  return *this;
}

operator_sum operator_sum::operator-=(const scalar_operator &other) {
  *this = *this - other;
  return *this;
}

operator_sum operator_sum::operator+(const product_operator &other) const {
  std::vector<product_operator> combined_terms = m_terms;
  combined_terms.push_back(other);
  return operator_sum(combined_terms);
}

operator_sum operator_sum::operator+=(const product_operator &other) {
  *this = *this + other;
  return *this;
}

/// FIXME:
// operator_sum operator_sum::operator-(const product_operator &other) const {
//   return *this + (-1. * other);
// }

/// FIXME:
// operator_sum operator_sum::operator-=(const product_operator &other) {
//   *this = *this - other;
//   return *this;
// }

complex_matrix
operator_sum::to_matrix(const std::map<int, int> &dimensions,
                        const std::map<std::string, double> &params) const {
  complex_matrix result = m_terms[0].to_matrix(dimensions, params);
  for (size_t i = 1; i < m_terms.size(); i++) {
    result += m_terms[i].to_matrix(dimensions, params);
  }
  return result;
}

/// Convert the operator sum to a string representation.
std::string operator_sum::to_string() const {
  std::string result;
  for (const auto &term : m_terms) {
    result += term.to_string() + " + ";
  }
  // Remove last " + "
  if (!result.empty())
    result.erase(result.size() - 3);
  return result;
}

} // namespace cudaq