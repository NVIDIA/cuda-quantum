/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "helpers.h"

#include <iostream>
#include <set>
#include <concepts>
#include <type_traits>

namespace cudaq {

// private methods

template <typename HandlerTy>
cudaq::matrix_2 operator_sum<HandlerTy>::m_evaluate(
    MatrixArithmetics arithmetics, std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters, bool pad_terms) const {

  auto terms = this->get_terms();

  std::set<int> degrees_set;
  for (auto op : terms) {
    for (auto degree : op.degrees()) {
      degrees_set.insert(degree);
    }
  }
  std::vector<int> degrees(degrees_set.begin(), degrees_set.end());

  // We need to make sure all matrices are of the same size to sum them up.
  auto paddedTerm = [&](auto &&term) {
    std::vector<int> op_degrees;
    for (auto op : term.get_terms()) {
      for (auto degree : op.degrees)
        op_degrees.push_back(degree);
    }
    for (auto degree : degrees) {
      auto it = std::find(op_degrees.begin(), op_degrees.end(), degree);
      if (it == op_degrees.end()) {
        term *= matrix_operator::identity(degree);
      }
    }
    return term;
  };

  auto sum = EvaluatedMatrix();
  if (pad_terms) {

    sum = EvaluatedMatrix(degrees, paddedTerm(terms[0]).m_evaluate(arithmetics, pad_terms));
    for (auto term_idx = 1; term_idx < terms.size(); ++term_idx) {
      auto term = terms[term_idx];

      auto eval = paddedTerm(term).m_evaluate(arithmetics, pad_terms);
      sum = arithmetics.add(sum, EvaluatedMatrix(degrees, eval));
    }
  } else {
    sum =
        EvaluatedMatrix(degrees, terms[0].m_evaluate(arithmetics, pad_terms));
    for (auto term_idx = 1; term_idx < terms.size(); ++term_idx) {
      auto term = terms[term_idx];
      auto eval =
          term.m_evaluate(arithmetics, pad_terms);
      sum = arithmetics.add(sum, EvaluatedMatrix(degrees, eval));
    }
  }
  return sum.matrix();
}

template<typename HandlerTy>
std::tuple<std::vector<scalar_operator>, std::vector<HandlerTy>> operator_sum<HandlerTy>::m_canonicalize_product(product_operator<HandlerTy> &prod) const {
  std::vector<scalar_operator> scalars = {prod.get_coefficient()};
  auto non_scalars = prod.get_terms();

  std::vector<int> all_degrees;
  for (auto op : non_scalars) {
    for (auto degree : op.degrees)
      all_degrees.push_back(degree);
  }

  std::set<int> unique_degrees(all_degrees.begin(), all_degrees.end());

  if (all_degrees.size() == unique_degrees.size()) {
    // Each operator acts on different degrees of freedom; they
    // hence commute and can be reordered arbitrarily.
    /// FIXME: Doing nothing for now
    // std::sort(non_scalars.begin(), non_scalars.end(), [](auto op){ return
    // op.degrees; })
  } else {
    // Some degrees exist multiple times; order the scalars, identities,
    // and zeros, but do not otherwise try to reorder terms.
    std::vector<matrix_operator> zero_ops;
    std::vector<matrix_operator> identity_ops;
    std::vector<matrix_operator> non_commuting;
    for (auto op : non_scalars) {
      if (op.id == "zero")
        zero_ops.push_back(op);
      if (op.id == "identity")
        identity_ops.push_back(op);
      if (op.id != "zero" || op.id != "identity")
        non_commuting.push_back(op);
    }

    /// FIXME: Not doing the same sorting we do in python yet
    std::vector<matrix_operator> sorted_non_scalars;
    sorted_non_scalars.insert(sorted_non_scalars.end(), zero_ops.begin(),
                              zero_ops.end());
    sorted_non_scalars.insert(sorted_non_scalars.end(), identity_ops.begin(),
                              identity_ops.end());
    sorted_non_scalars.insert(sorted_non_scalars.end(), non_commuting.begin(),
                              non_commuting.end());
    non_scalars = sorted_non_scalars;
  }
  return std::make_tuple(scalars, non_scalars);
}

template<typename HandlerTy>
std::tuple<std::vector<scalar_operator>, std::vector<HandlerTy>> operator_sum<HandlerTy>::m_canonical_terms() const {
  /// FIXME: Not doing the same sorting we do in python yet
  std::tuple<std::vector<scalar_operator>, std::vector<matrix_operator>> result;
  std::vector<scalar_operator> scalars;
  std::vector<matrix_operator> matrix_ops;
  for (auto term : this->get_terms()) {
    auto canon_term = m_canonicalize_product(term);
    auto canon_scalars = std::get<0>(canon_term);
    auto canon_elementary = std::get<1>(canon_term);
    scalars.insert(scalars.end(), canon_scalars.begin(), canon_scalars.end());
    canon_elementary.insert(canon_elementary.end(), canon_elementary.begin(), canon_elementary.end());
  }
  return std::make_tuple(scalars, matrix_ops);
}

template<typename HandlerTy>
void operator_sum<HandlerTy>::aggregate_terms() {}

template<typename HandlerTy>
template <typename ... Args>
void operator_sum<HandlerTy>::aggregate_terms(const product_operator<HandlerTy> &head, Args&& ... args) {
    this->terms.push_back(head.terms[0]);
    this->coefficients.push_back(head.coefficients[0]);
    aggregate_terms(std::forward<Args>(args)...);
}

template
cudaq::matrix_2 operator_sum<matrix_operator>::m_evaluate(
    MatrixArithmetics arithmetics, std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters, bool pad_terms) const;

template
std::tuple<std::vector<scalar_operator>, std::vector<matrix_operator>> operator_sum<matrix_operator>::m_canonicalize_product(product_operator<matrix_operator> &prod) const;

template
std::tuple<std::vector<scalar_operator>, std::vector<matrix_operator>> operator_sum<matrix_operator>::m_canonical_terms() const;

// no overload for a single product, since we don't want a constructor for a single term

template
void operator_sum<matrix_operator>::aggregate_terms(const product_operator<matrix_operator> &item1, 
                                                        const product_operator<matrix_operator> &item2);

template
void operator_sum<matrix_operator>::aggregate_terms(const product_operator<matrix_operator> &item1, 
                                                        const product_operator<matrix_operator> &item2,
                                                        const product_operator<matrix_operator> &item3);

// read-only properties

template<typename HandlerTy>
std::vector<int> operator_sum<HandlerTy>::degrees() const {
  std::set<int> unsorted_degrees;
  for (const std::vector<HandlerTy> &term : this->terms) {
    for (const HandlerTy &op : term)
      unsorted_degrees.insert(op.degrees.begin(), op.degrees.end());
  }
  auto degrees = std::vector<int>(unsorted_degrees.begin(), unsorted_degrees.end());
  return cudaq::detail::canonicalize_degrees(degrees);
}

template<typename HandlerTy>
int operator_sum<HandlerTy>::n_terms() const { 
    return this->terms.size(); 
}

template<typename HandlerTy>
std::vector<product_operator<HandlerTy>> operator_sum<HandlerTy>::get_terms() const { 
    std::vector<product_operator<HandlerTy>> prods;
    prods.reserve(this->terms.size());
    for (size_t i = 0; i < this->terms.size(); ++i) {
        prods.push_back(product_operator<HandlerTy>(this->coefficients[i], this->terms[i]));
    }
    return prods; 
}

template
std::vector<int> operator_sum<matrix_operator>::degrees() const;

template
int operator_sum<matrix_operator>::n_terms() const;

template
std::vector<product_operator<matrix_operator>> operator_sum<matrix_operator>::get_terms() const;

// constructors

template<typename HandlerTy>
template<class... Args, class>
operator_sum<HandlerTy>::operator_sum(const Args&... args) {
    this->terms.reserve(sizeof...(Args));
    this->coefficients.reserve(sizeof...(Args));
    aggregate_terms(args...);
}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(const std::vector<product_operator<HandlerTy>> &terms) { 
    this->terms.reserve(terms.size());
    this->coefficients.reserve(terms.size());
    for (const product_operator<HandlerTy>& term : terms) {
        this->terms.push_back(term.terms[0]);
        this->coefficients.push_back(term.coefficients[0]);
    }
}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(std::vector<product_operator<HandlerTy>> &&terms) { 
    this->terms.reserve(terms.size());
    for (const product_operator<HandlerTy>& term : terms) {
        this->terms.push_back(std::move(term.terms[0]));
        this->coefficients.push_back(std::move(term.coefficients[0]));
    }
}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(const operator_sum<HandlerTy> &other)
    : coefficients(other.coefficients), terms(other.terms) {}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(operator_sum<HandlerTy> &&other) 
    : coefficients(std::move(other.coefficients)), terms(std::move(other.terms)) {}

// no constructor for a single product, since that one should remain a product op

template 
operator_sum<matrix_operator>::operator_sum(const product_operator<matrix_operator> &item1,
                                                const product_operator<matrix_operator> &item2);

template 
operator_sum<matrix_operator>::operator_sum(const product_operator<matrix_operator> &item1,
                                                const product_operator<matrix_operator> &item2,
                                                const product_operator<matrix_operator> &item3);

template
operator_sum<matrix_operator>::operator_sum(const std::vector<product_operator<matrix_operator>> &terms);

template
operator_sum<matrix_operator>::operator_sum(std::vector<product_operator<matrix_operator>> &&terms);

template
operator_sum<matrix_operator>::operator_sum(const operator_sum<matrix_operator> &other);

template
operator_sum<matrix_operator>::operator_sum(operator_sum<matrix_operator> &&other);

// assignments

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(const operator_sum<HandlerTy> &other) {
    if (this != &other) {
        coefficients = other.coefficients;
        terms = other.terms;
    }
    return *this;
}

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(operator_sum<HandlerTy> &&other) {
    if (this != &other) {
        coefficients = std::move(other.coefficients);
        terms = std::move(other.terms);
    }
    return *this;
}

template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator=(const operator_sum<matrix_operator>& other);

template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator=(operator_sum<matrix_operator> &&other);

// evaluations

template<typename HandlerTy>
std::string operator_sum<HandlerTy>::to_string() const {
    throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
matrix_2 operator_sum<HandlerTy>::to_matrix(const std::map<int, int> &dimensions,
                                            const std::map<std::string, std::complex<double>> &parameters) const {
  /// FIXME: Not doing any conversion to spin op yet.
  return m_evaluate(MatrixArithmetics(dimensions, parameters), dimensions,
                    parameters);
}

template
std::string operator_sum<matrix_operator>::to_string() const;

template
matrix_2 operator_sum<matrix_operator>::to_matrix(const std::map<int, int> &dimensions,
                                                      const std::map<std::string, std::complex<double>> &params) const;

// comparisons

template<typename HandlerTy>
bool operator_sum<HandlerTy>::operator==(const operator_sum<HandlerTy> &other) const {
    throw std::runtime_error("not implemented");
}

template
bool operator_sum<matrix_operator>::operator==(const operator_sum<matrix_operator> &other) const;

// unary operators

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-() const {
  std::vector<scalar_operator> coefficients;
  coefficients.reserve(this->coefficients.size());
  for (auto &coeff : this->coefficients)
    coefficients.push_back(-1. * coeff);
  operator_sum<HandlerTy> sum;
  sum.coefficients = std::move(coefficients);
  sum.terms = this->terms;
  return sum;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+() const {
  return *this;
}

template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator-() const;

template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator+() const;

// right-hand arithmetics

#define SUM_MULTIPLICATION(otherTy)                                                     \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(otherTy other) const {     \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(this->coefficients.size());                                    \
    for (auto &coeff : this->coefficients)                                              \
      coefficients.push_back(coeff * other);                                            \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = this->terms;                                                            \
    return sum;                                                                         \
  }

SUM_MULTIPLICATION(double);
SUM_MULTIPLICATION(std::complex<double>);
SUM_MULTIPLICATION(const scalar_operator &);

#define SUM_ADDITION(otherTy, op)                                                       \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(otherTy other) const {   \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(this->coefficients.size() + 1);                                \
    coefficients.push_back(op other);                                                   \
    for (auto &coeff : this->coefficients)                                              \
      coefficients.push_back(coeff);                                                    \
    std::vector<std::vector<HandlerTy>> terms;                                          \
    terms.reserve(this->terms.size() + 1);                                              \
    terms.push_back({});                                                                \
    for (auto &term : this->terms)                                                      \
      terms.push_back(term);                                                            \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = std::move(terms);                                                       \
    return sum;                                                                         \
  }

SUM_ADDITION(double, +);
SUM_ADDITION(double, -);
SUM_ADDITION(std::complex<double>, +);
SUM_ADDITION(std::complex<double>, -);
SUM_ADDITION(const scalar_operator &, +);
SUM_ADDITION(const scalar_operator &, -);

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const HandlerTy &other) const {
  std::vector<std::vector<HandlerTy>> terms;
  terms.reserve(this->terms.size());
  for (auto &term : this->terms) {
    std::vector<HandlerTy> prod;
    prod.reserve(term.size() + 1);
    for (auto &op : term)
      prod.push_back(op);
    prod.push_back(other);
    terms.push_back(std::move(prod));
  }
  operator_sum<HandlerTy> sum;
  sum.coefficients = this->coefficients;
  sum.terms = std::move(terms);
  return sum;
}

#define SUM_ADDITION_HANDLER(op)                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                                   const HandlerTy &other) const {      \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(this->coefficients.size() + 1);                                \
    coefficients.push_back(op 1.);                                                      \
    for (auto &coeff : this->coefficients)                                              \
      coefficients.push_back(coeff);                                                    \
    std::vector<std::vector<HandlerTy>> terms;                                          \
    terms.reserve(this->terms.size() + 1);                                              \
    std::vector<HandlerTy> newTerm;                                                     \
    newTerm.push_back(other);                                                           \
    terms.push_back(std::move(newTerm));                                                \
    for (auto &term : this->terms)                                                      \
      terms.push_back(term);                                                            \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = std::move(terms);                                                       \
    return sum;                                                                         \
  }

SUM_ADDITION_HANDLER(+)
SUM_ADDITION_HANDLER(-)

template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator*(double other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator+(double other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator-(double other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator*(std::complex<double> other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator+(std::complex<double> other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator-(std::complex<double> other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator*(const scalar_operator &other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator+(const scalar_operator &other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator-(const scalar_operator &other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator*(const matrix_operator &other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator+(const matrix_operator &other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator-(const matrix_operator &other) const;

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const product_operator<HandlerTy> &other) const {
  std::vector<scalar_operator> coefficients;
  coefficients.reserve(this->coefficients.size());
  for (auto &coeff : this->coefficients)
    coefficients.push_back(other.coefficients[0] * coeff);
  std::vector<std::vector<HandlerTy>> terms;
  terms.reserve(this->terms.size());
  for (auto &term : this->terms) {
    std::vector<HandlerTy> prod;
    prod.reserve(term.size() + other.terms[0].size());
    for (auto &op : term) 
      prod.push_back(op);
    for (auto &op : other.terms[0])
      prod.push_back(op);
    terms.push_back(std::move(prod));
  }
  operator_sum<HandlerTy> sum;
  sum.coefficients = std::move(coefficients);
  sum.terms = std::move(terms);
  return sum;
}

#define SUM_ADDITION_PRODUCT(op)                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                     const product_operator<HandlerTy> &other) const {  \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(this->coefficients.size() + 1);                                \
    for (auto &coeff : this->coefficients)                                              \
      coefficients.push_back(coeff);                                                    \
    coefficients.push_back(op other.coefficients[0]);                                   \
    std::vector<std::vector<HandlerTy>> terms;                                          \
    terms.reserve(this->terms.size() + 1);                                              \
    for (auto &term : this->terms)                                                      \
      terms.push_back(term);                                                            \
    terms.push_back(other.terms[0]);                                                    \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = std::move(terms);                                                       \
    return sum;                                                                         \
  }

SUM_ADDITION_PRODUCT(+)
SUM_ADDITION_PRODUCT(-)

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const operator_sum<HandlerTy> &other) const {
  std::vector<scalar_operator> coefficients;
  coefficients.reserve(this->coefficients.size() * other.coefficients.size());
  for (auto &coeff1 : this->coefficients) {
    for (auto &coeff2 : other.coefficients)
      coefficients.push_back(coeff1 * coeff2);
  }
  std::vector<std::vector<HandlerTy>> terms;
  terms.reserve(this->terms.size() * other.terms.size());
  for (auto &term1 : this->terms) {
    for (auto &term2 : other.terms) {
      std::vector<HandlerTy> prod;
      prod.reserve(term1.size() + term2.size());
      for (auto &op : term1)
        prod.push_back(op);
      for (auto &op : term2)
        prod.push_back(op);
      terms.push_back(std::move(prod));
    }
  }
  operator_sum<HandlerTy> sum;
  sum.coefficients = std::move(coefficients);
  sum.terms = std::move(terms);
  return sum;
}

#define SUM_ADDITION_SUM(op)                                                            \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                         const operator_sum<HandlerTy> &other) const {  \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(this->coefficients.size() + other.coefficients.size());        \
    for (auto &coeff : this->coefficients)                                              \
      coefficients.push_back(coeff);                                                    \
    for (auto &coeff : other.coefficients)                                              \
      coefficients.push_back(op coeff);                                                 \
    std::vector<std::vector<HandlerTy>> terms;                                          \
    terms.reserve(this->terms.size() + other.terms.size());                             \
    for (auto &term : this->terms)                                                      \
      terms.push_back(term);                                                            \
    for (auto &term : other.terms)                                                      \
      terms.push_back(term);                                                            \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = std::move(terms);                                                       \
    return sum;                                                                         \
  }

SUM_ADDITION_SUM(+);
SUM_ADDITION_SUM(-);

template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator*(const product_operator<matrix_operator> &other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator+(const product_operator<matrix_operator> &other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator-(const product_operator<matrix_operator> &other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator*(const operator_sum<matrix_operator> &other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator+(const operator_sum<matrix_operator> &other) const;
template
operator_sum<matrix_operator> operator_sum<matrix_operator>::operator-(const operator_sum<matrix_operator> &other) const;

#define SUM_MULTIPLICATION_ASSIGNMENT(otherTy)                                          \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(otherTy other) {         \
    for (auto &coeff : this->coefficients)                                              \
      coeff *= other;                                                                   \
    return *this;                                                                       \
  }

SUM_MULTIPLICATION_ASSIGNMENT(double);
SUM_MULTIPLICATION_ASSIGNMENT(std::complex<double>);
SUM_MULTIPLICATION_ASSIGNMENT(const scalar_operator &);

#define SUM_ADDITION_ASSIGNMENT(otherTy, op)                                            \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(otherTy other) {     \
    this->coefficients.push_back(op other);                                             \
    this->terms.push_back({});                                                          \
    return *this;                                                                       \
  }

SUM_ADDITION_ASSIGNMENT(double, +);
SUM_ADDITION_ASSIGNMENT(double, -);
SUM_ADDITION_ASSIGNMENT(std::complex<double>, +);
SUM_ADDITION_ASSIGNMENT(std::complex<double>, -);
SUM_ADDITION_ASSIGNMENT(const scalar_operator &, +);
SUM_ADDITION_ASSIGNMENT(const scalar_operator &, -);

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const HandlerTy &other) {
  for (auto &term : this->terms)
    term.push_back(other);
  operator_sum<HandlerTy> sum;
  return *this;
}

#define SUM_ADDITION_HANDLER_ASSIGNMENT(op)                                             \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(                     \
                                                    const HandlerTy &other) {           \
    coefficients.push_back(op 1.);                                                      \
    std::vector<HandlerTy> newTerm;                                                     \
    newTerm.push_back(other);                                                           \
    this->terms.push_back(std::move(newTerm));                                          \
    return *this;                                                                       \
  }

SUM_ADDITION_HANDLER_ASSIGNMENT(+)
SUM_ADDITION_HANDLER_ASSIGNMENT(-)

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const product_operator<HandlerTy> &other) {
  for (auto &coeff : this->coefficients)
    coeff *= other.coefficients[0];
  for (auto &term : this->terms) {
    term.reserve(term.size() + other.terms[0].size());
    for (auto &op : other.terms[0])
      term.push_back(op);
  }
  return *this;
}

#define SUM_ADDITION_PRODUCT_ASSIGNMENT(op)                                             \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(                     \
                                           const product_operator<HandlerTy> &other) {  \
    this->coefficients.push_back(op other.coefficients[0]);                             \
    this->terms.push_back(other.terms[0]);                                              \
    return *this;                                                                       \
  }

SUM_ADDITION_PRODUCT_ASSIGNMENT(+)
SUM_ADDITION_PRODUCT_ASSIGNMENT(-)

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const operator_sum<HandlerTy> &other) {
  this->coefficients.reserve(this->coefficients.size() * other.coefficients.size());
  *this = *this * other; // we need to update all coefficients and terms anyway
  return *this;
}

#define SUM_ADDITION_SUM_ASSIGNMENT(op)                                                            \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(                     \
                                               const operator_sum<HandlerTy> &other) {  \
    this->coefficients.reserve(this->coefficients.size() + other.coefficients.size());  \
    for (auto &coeff : other.coefficients)                                              \
      this->coefficients.push_back(op coeff);                                           \
    this->terms.reserve(this->terms.size() + other.terms.size());                       \
    for (auto &term : other.terms)                                                      \
      this->terms.push_back(term);                                                      \
    return *this;                                                                       \
  }

SUM_ADDITION_SUM_ASSIGNMENT(+);
SUM_ADDITION_SUM_ASSIGNMENT(-);

template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator*=(double other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator+=(double other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator-=(double other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator*=(std::complex<double> other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator+=(std::complex<double> other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator-=(std::complex<double> other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator*=(const scalar_operator &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator+=(const scalar_operator &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator-=(const scalar_operator &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator*=(const matrix_operator &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator+=(const matrix_operator &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator-=(const matrix_operator &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator*=(const product_operator<matrix_operator> &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator+=(const product_operator<matrix_operator> &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator-=(const product_operator<matrix_operator> &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator*=(const operator_sum<matrix_operator> &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator-=(const operator_sum<matrix_operator> &other);
template
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator+=(const operator_sum<matrix_operator> &other);

// left-hand arithmetics

#define SUM_MULTIPLICATION_REVERSE(otherTy)                                             \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator*(otherTy other,                                      \
                                    const operator_sum<HandlerTy> &self) {              \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(self.coefficients.size());                                     \
    for (auto &coeff : self.coefficients)                                               \
      coefficients.push_back(coeff * other);                                            \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = self.terms;                                                             \
    return sum;                                                                         \
  }

SUM_MULTIPLICATION_REVERSE(double);
SUM_MULTIPLICATION_REVERSE(std::complex<double>);
SUM_MULTIPLICATION_REVERSE(const scalar_operator &);

#define SUM_ADDITION_REVERSE(otherTy, op)                                               \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator op(otherTy other,                                    \
                                      const operator_sum<HandlerTy> &self) {            \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(self.terms.size() + 1);                                        \
    coefficients.push_back(other);                                                      \
    for (auto &coeff : self.coefficients)                                               \
      coefficients.push_back(op coeff);                                                 \
    std::vector<std::vector<HandlerTy>> terms;                                          \
    terms.reserve(self.terms.size() + 1);                                               \
    terms.push_back({});                                                                \
    for (auto &term : self.terms)                                                       \
      terms.push_back(term);                                                            \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = std::move(terms);                                                       \
    return sum;                                                                         \
  }

SUM_ADDITION_REVERSE(double, +);
SUM_ADDITION_REVERSE(double, -);
SUM_ADDITION_REVERSE(std::complex<double>, +);
SUM_ADDITION_REVERSE(std::complex<double>, -);
SUM_ADDITION_REVERSE(const scalar_operator &, +);
SUM_ADDITION_REVERSE(const scalar_operator &, -);

template <typename HandlerTy>
operator_sum<HandlerTy> operator*(const HandlerTy &other, const operator_sum<HandlerTy> &self) {
  std::vector<std::vector<HandlerTy>> terms;
  terms.reserve(self.terms.size());
  for (auto &term : self.terms) {
    std::vector<HandlerTy> prod;
    prod.reserve(term.size() + 1);
    prod.push_back(other);
    for (auto &op : term)
      prod.push_back(op);
    terms.push_back(std::move(prod));
  }
  operator_sum<HandlerTy> sum;
  sum.coefficients = self.coefficients;
  sum.terms = std::move(terms);
  return sum;
}

#define SUM_ADDITION_HANDLER_REVERSE(op)                                                \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator op(const HandlerTy &other,                           \
                                      const operator_sum<HandlerTy> &self) {            \
    std::vector<scalar_operator> coefficients;                                          \
    coefficients.reserve(self.terms.size() + 1);                                        \
    coefficients.push_back(1.);                                                         \
    for (auto &coeff : self.coefficients)                                               \
      coefficients.push_back(op coeff);                                                 \
    std::vector<std::vector<HandlerTy>> terms;                                          \
    terms.reserve(self.terms.size() + 1);                                               \
    std::vector<HandlerTy> newTerm;                                                     \
    newTerm.push_back(other);                                                           \
    terms.push_back(std::move(newTerm));                                                \
    for (auto &term : self.terms)                                                       \
      terms.push_back(term);                                                            \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients = std::move(coefficients);                                         \
    sum.terms = std::move(terms);                                                       \
    return sum;                                                                         \
  }

SUM_ADDITION_HANDLER_REVERSE(+)
SUM_ADDITION_HANDLER_REVERSE(-)

template
operator_sum<matrix_operator> operator*(const scalar_operator &other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator*(std::complex<double> other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator*(double other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator*(const matrix_operator &other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator+(const scalar_operator &other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator+(double other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator+(std::complex<double> other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator+(const matrix_operator &other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator-(const scalar_operator &other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator-(double other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator-(std::complex<double> other, const operator_sum<matrix_operator> &self);
template
operator_sum<matrix_operator> operator-(const matrix_operator &other, const operator_sum<matrix_operator> &self);

} // namespace cudaq