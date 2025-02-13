/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <algorithm>
#include <numeric>
#include <set>
#include <type_traits>

#include "cudaq/operators.h"
#include "helpers.h"
#include "manipulation.h"
#include "matrix_operators.h"
#include "spin_operators.h"
#include "boson_operators.h"

namespace cudaq {

// private methods

template<typename HandlerTy>
void operator_sum<HandlerTy>::insert(product_operator<HandlerTy> &&other) {
  // the logic below just ensures that terms are properly combined, if possible
  auto it = std::find_if(this->terms.cbegin(), this->terms.cend(), 
    [&other_ops = static_cast<const std::vector<HandlerTy>&>(other.operators)](const std::vector<HandlerTy> &self_ops) { 
      bool are_same = other_ops.size() == self_ops.size();
      for (auto i = 0; are_same && i < other_ops.size(); ++i)
        are_same = other_ops[i] == self_ops[i];
      return are_same;
  });

  if (it == this->terms.end()) {
    this->terms.push_back(std::move(other.operators));
    this->coefficients.push_back(std::move(other.coefficient));
  } else {
    this->coefficients[std::distance(this->terms.cbegin(), it)] += other.coefficient;
  }
}

template<typename HandlerTy>
void operator_sum<HandlerTy>::aggregate_all() {
  std::unordered_map<std::string, std::vector<int>> term_map;
  for (auto i = 0; i < this->terms.size(); ++i) {
    std::string key;
    for (const HandlerTy &op : this->terms[i])
      key += op.to_string(true); // includes degree(s)
    term_map[key].push_back(i);
  }

  std::vector<std::vector<HandlerTy>> terms;
  terms.reserve(this->terms.size());
  std::vector<scalar_operator> coefficients;
  coefficients.reserve(this->coefficients.size());
  for (const auto &entry : term_map) {
    auto coefficient = std::move(this->coefficients[entry.second[0]]);
    for (auto i = 1; i < entry.second.size(); ++i) {
      // FIXME: maybe double check equality here if we don't want to rely on to_string too much?
      coefficient += this->coefficients[entry.second[i]];
    }
    coefficients.push_back(std::move(coefficient));
    terms.push_back(std::move(this->terms[entry.second[0]])); // fine to move
  }
  this->terms = std::move(terms);
  this->coefficients = std::move(coefficients);
}

template<typename HandlerTy>
void operator_sum<HandlerTy>::aggregate_terms() {}

template<typename HandlerTy>
template <typename ... Args>
void operator_sum<HandlerTy>::aggregate_terms(product_operator<HandlerTy> &&head, Args&& ... args) {
  this->insert(std::forward<product_operator<HandlerTy>>(head));
  aggregate_terms(std::forward<Args>(args)...);
}

template <typename HandlerTy>
EvaluatedMatrix operator_sum<HandlerTy>::m_evaluate(
    MatrixArithmetics arithmetics, bool pad_terms) const {

  auto terms = this->get_terms();
  auto degrees = this->degrees();

  // We need to make sure all matrices are of the same size to sum them up.
  auto paddedTerm = 
    [&arithmetics, &degrees = std::as_const(degrees)](product_operator<HandlerTy> &&term) {
      std::vector<HandlerTy> prod_ops;
      prod_ops.reserve(degrees.size());
      auto term_degrees = term.degrees();
      for (auto degree : degrees) {
        auto it = std::find(term_degrees.begin(), term_degrees.end(), degree);
        if (it == term_degrees.end())
          prod_ops.push_back(HandlerTy(degree));
      }
      product_operator<HandlerTy> prod(1, std::move(prod_ops));
      prod *= term; // ensures canonical ordering
      return prod;
  };

  if (pad_terms) {
    auto padded_term = paddedTerm(std::move(terms[0]));
    EvaluatedMatrix sum = padded_term.m_evaluate(arithmetics, true);
    for (auto term_idx = 1; term_idx < terms.size(); ++term_idx) {
      padded_term = paddedTerm(std::move(terms[term_idx]));
      auto term_eval = padded_term.m_evaluate(arithmetics, true);
      sum = arithmetics.add(std::move(sum), std::move(term_eval));
    }
    return sum;
  } else {
    EvaluatedMatrix sum = terms[0].m_evaluate(arithmetics, false);
    for (auto term_idx = 1; term_idx < terms.size(); ++term_idx) {
      auto term_eval = terms[term_idx].m_evaluate(arithmetics, false);
      sum = arithmetics.add(std::move(sum), std::move(term_eval));
    }
    return sum;
  }
}

#define INSTANTIATE_SUM_PRIVATE_METHODS(HandlerTy)                                            \
                                                                                              \
  template                                                                                    \
  void operator_sum<HandlerTy>::aggregate_terms(product_operator<HandlerTy> &&item2);         \
                                                                                              \
  template                                                                                    \
  void operator_sum<HandlerTy>::aggregate_terms(product_operator<HandlerTy> &&item1,          \
                                                product_operator<HandlerTy> &&item2);         \
                                                                                              \
  template                                                                                    \
  void operator_sum<HandlerTy>::aggregate_terms(product_operator<HandlerTy> &&item1,          \
                                                product_operator<HandlerTy> &&item2,          \
                                                product_operator<HandlerTy> &&item3);         \
                                                                                              \
  template                                                                                    \
  EvaluatedMatrix operator_sum<HandlerTy>::m_evaluate(                                        \
      MatrixArithmetics arithmetics, bool pad_terms) const;

INSTANTIATE_SUM_PRIVATE_METHODS(matrix_operator);
INSTANTIATE_SUM_PRIVATE_METHODS(spin_operator);
INSTANTIATE_SUM_PRIVATE_METHODS(boson_operator);

// read-only properties

template<typename HandlerTy>
std::vector<int> operator_sum<HandlerTy>::degrees() const {
  std::set<int> unsorted_degrees;
  for (const std::vector<HandlerTy> &term : this->terms) {
    for (const HandlerTy &op : term) {
      auto op_degrees = op.degrees();
      unsorted_degrees.insert(op_degrees.cbegin(), op_degrees.cend());
    }
  }
  auto degrees = std::vector<int>(unsorted_degrees.cbegin(), unsorted_degrees.cend());
  cudaq::detail::canonicalize_degrees(degrees);
  return degrees;
}

template<typename HandlerTy>
int operator_sum<HandlerTy>::num_terms() const { 
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
#define INSTANTIATE_SUM_PROPERTIES(HandlerTy)                                               \
                                                                                            \
  template                                                                                  \
  std::vector<int> operator_sum<HandlerTy>::degrees() const;                                \
                                                                                            \
  template                                                                                  \
  int operator_sum<HandlerTy>::num_terms() const;                                             \
                                                                                            \
  template                                                                                  \
  std::vector<product_operator<HandlerTy>> operator_sum<HandlerTy>::get_terms() const;

INSTANTIATE_SUM_PROPERTIES(matrix_operator);
INSTANTIATE_SUM_PROPERTIES(spin_operator);
INSTANTIATE_SUM_PROPERTIES(boson_operator);

// constructors

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(const product_operator<HandlerTy> &prod) {
  this->coefficients.push_back(prod.coefficient);
  this->terms.push_back(prod.operators);
}

template<typename HandlerTy>
template<typename... Args, std::enable_if_t<std::conjunction<std::is_same<product_operator<HandlerTy>, Args>...>::value, bool>>
operator_sum<HandlerTy>::operator_sum(Args&&... args) {
  this->terms.reserve(sizeof...(Args));
  this->coefficients.reserve(sizeof...(Args));
  aggregate_terms(std::forward<product_operator<HandlerTy>&&>(args)...);
}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(std::vector<product_operator<HandlerTy>> &&terms) { 
  this->terms.reserve(terms.size());
  for (auto i = 0; i < terms.size(); ++i)
    this->insert(std::move(terms[i]));
}

template<typename HandlerTy>
template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool>>
operator_sum<HandlerTy>::operator_sum(const operator_sum<T> &other) {
  this->coefficients = other.coefficients;
  this->terms.reserve(other.terms.size());
  for (const auto &term : other.terms) {
    std::vector<HandlerTy> other_terms;
    other_terms.reserve(other.terms.size());
    for (const T &op : term)
      other_terms.push_back(op);
    this->terms.push_back(std::move(other_terms));
  }
}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(const operator_sum<HandlerTy> &other)
  : coefficients(other.coefficients), terms(other.terms) {}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(operator_sum<HandlerTy> &&other) 
  : coefficients(std::move(other.coefficients)), terms(std::move(other.terms)) {}

#define INSTANTIATE_SUM_CONSTRUCTORS(HandlerTy)                                                 \
                                                                                                \
  template                                                                                      \
  operator_sum<HandlerTy>::operator_sum(const product_operator<HandlerTy> &item2);              \
                                                                                                \
  template                                                                                      \
  operator_sum<HandlerTy>::operator_sum(product_operator<HandlerTy> &&item2);                   \
                                                                                                \
  template                                                                                      \
  operator_sum<HandlerTy>::operator_sum(product_operator<HandlerTy> &&item1,                    \
                                        product_operator<HandlerTy> &&item2);                   \
                                                                                                \
  template                                                                                      \
  operator_sum<HandlerTy>::operator_sum(product_operator<HandlerTy> &&item1,                    \
                                        product_operator<HandlerTy> &&item2,                    \
                                        product_operator<HandlerTy> &&item3);                   \
                                                                                                \
  template                                                                                      \
  operator_sum<HandlerTy>::operator_sum(std::vector<product_operator<HandlerTy>> &&terms);      \
                                                                                                \
  template                                                                                      \
  operator_sum<HandlerTy>::operator_sum(const operator_sum<HandlerTy> &other);                  \
                                                                                                \
  template                                                                                      \
  operator_sum<HandlerTy>::operator_sum(operator_sum<HandlerTy> &&other);

template 
operator_sum<matrix_operator>::operator_sum(const operator_sum<spin_operator> &other);
template 
operator_sum<matrix_operator>::operator_sum(const operator_sum<boson_operator> &other);

INSTANTIATE_SUM_CONSTRUCTORS(matrix_operator);
INSTANTIATE_SUM_CONSTRUCTORS(spin_operator);
INSTANTIATE_SUM_CONSTRUCTORS(boson_operator);

// assignments

template<typename HandlerTy>
  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool>>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(const product_operator<T> &other) {
  this->coefficients.clear();
  this->terms.clear();
  std::vector<HandlerTy> operators;
  operators.reserve(other.operators.size());
  for (const T &op : other.operators)
    operators.push_back(op);
  this->coefficients.push_back(other.coefficient);
  this->terms.push_back(std::move(operators));
  return *this;
}

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(product_operator<HandlerTy> &&other) {
  this->coefficients.clear();
  this->terms.clear();
  this->coefficients.push_back(std::move(other.coefficient));
  this->terms.push_back(std::move(other.operators));
  return *this;
}

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(const product_operator<HandlerTy> &other) {
  this->coefficients.clear();
  this->terms.clear();
  this->coefficients.push_back(other.coefficient);
  this->terms.push_back(other.operators);
  return *this;
}

template<typename HandlerTy>
template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool>>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(const operator_sum<T> &other) {
  *this = operator_sum<HandlerTy>(other);
  return *this;
}

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(const operator_sum<HandlerTy> &other) {
  if (this != &other) {
      this->coefficients = other.coefficients;
      this->terms = other.terms;
  }
  return *this;
}

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(operator_sum<HandlerTy> &&other) {
  if (this != &other) {
      this->coefficients = std::move(other.coefficients);
      this->terms = std::move(other.terms);
  }
  return *this;
}

#define INSTANTIATE_SUM_ASSIGNMENTS(HandlerTy)                                              \
                                                                                            \
  template                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(                              \
    product_operator<HandlerTy> &&other);                                                   \
                                                                                            \
  template                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(                              \
    const product_operator<HandlerTy> &other);                                              \
                                                                                            \
  template                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(                              \
    const operator_sum<HandlerTy> &other);                                                  \
                                                                                            \
  template                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(                              \
    operator_sum<HandlerTy> &&other);

template 
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator=(const product_operator<spin_operator> &other);
template 
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator=(const product_operator<boson_operator> &other);
template 
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator=(const operator_sum<spin_operator> &other);
template 
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator=(const operator_sum<boson_operator> &other);

INSTANTIATE_SUM_ASSIGNMENTS(matrix_operator);
INSTANTIATE_SUM_ASSIGNMENTS(spin_operator);
INSTANTIATE_SUM_ASSIGNMENTS(boson_operator);

// evaluations

template<typename HandlerTy>
std::string operator_sum<HandlerTy>::to_string() const {
    throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
matrix_2 operator_sum<HandlerTy>::to_matrix(std::unordered_map<int, int> dimensions,
                                            const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  return m_evaluate(MatrixArithmetics(dimensions, parameters)).matrix();
}

#define INSTANTIATE_SUM_EVALUATIONS(HandlerTy)                                              \
                                                                                            \
  template                                                                                  \
  std::string operator_sum<HandlerTy>::to_string() const;                                   \
                                                                                            \
  template                                                                                  \
  matrix_2 operator_sum<HandlerTy>::to_matrix(                                              \
    std::unordered_map<int, int> dimensions,                                               \
    const std::unordered_map<std::string, std::complex<double>> &params) const;

INSTANTIATE_SUM_EVALUATIONS(matrix_operator);
INSTANTIATE_SUM_EVALUATIONS(spin_operator);
INSTANTIATE_SUM_EVALUATIONS(boson_operator);

// comparisons

template<typename HandlerTy>
bool operator_sum<HandlerTy>::operator==(const operator_sum<HandlerTy> &other) const {
  bool are_same = this->terms.size() == other.terms.size();
  for (auto i = 0; are_same && i < this->terms.size(); ++i)
    are_same = product_operator<HandlerTy>(this->coefficients[i], this->terms[i]) ==
               product_operator<HandlerTy>(other.coefficients[i], other.terms[i]);
  return are_same;
}

#define INSTANTIATE_SUM_COMPARISONS(HandlerTy)                                              \
                                                                                            \
  template                                                                                  \
  bool operator_sum<HandlerTy>::operator==(const operator_sum<HandlerTy> &other) const;

INSTANTIATE_SUM_COMPARISONS(matrix_operator);
INSTANTIATE_SUM_COMPARISONS(spin_operator);
INSTANTIATE_SUM_COMPARISONS(boson_operator);

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

#define INSTANTIATE_SUM_UNARY_OPS(HandlerTy)                                            \
                                                                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-() const;                   \
                                                                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+() const;

INSTANTIATE_SUM_UNARY_OPS(matrix_operator);
INSTANTIATE_SUM_UNARY_OPS(spin_operator);
INSTANTIATE_SUM_UNARY_OPS(boson_operator);

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

/*
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
*/

#define INSTANTIATE_SUM_RHSIMPLE_OPS(HandlerTy)                                                     \
                                                                                                    \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(double other) const;                   \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(double other) const;                   \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(double other) const;                   \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(std::complex<double> other) const;     \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(std::complex<double> other) const;     \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(std::complex<double> other) const;     \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const scalar_operator &other) const;   \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const scalar_operator &other) const;   \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const scalar_operator &other) const;   \

/*
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const HandlerTy &other) const;         \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const HandlerTy &other) const;         \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const HandlerTy &other) const;
*/

INSTANTIATE_SUM_RHSIMPLE_OPS(matrix_operator);
INSTANTIATE_SUM_RHSIMPLE_OPS(spin_operator);
INSTANTIATE_SUM_RHSIMPLE_OPS(boson_operator);

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const product_operator<HandlerTy> &other) const {
  operator_sum<HandlerTy> sum;
  sum.coefficients.reserve(this->coefficients.size());
  sum.terms.reserve(this->terms.size());
  for (auto i = 0; i < this->terms.size(); ++i) {
    product_operator<HandlerTy> prod(this->coefficients[i], this->terms[i]);
    prod *= other;
    sum.coefficients.push_back(std::move(prod.coefficient));
    sum.terms.push_back(std::move(prod.operators));
  }
  return sum;
}

#define SUM_ADDITION_PRODUCT(op)                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                     const product_operator<HandlerTy> &other) const {  \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients.reserve(this->coefficients.size() + 1);                            \
    sum.terms.reserve(this->terms.size() + 1);                                          \
    for (auto &coeff : this->coefficients)                                              \
      sum.coefficients.push_back(coeff);                                                \
    for (auto &term : this->terms)                                                      \
      sum.terms.push_back(term);                                                        \
    sum.insert(op other);                                                               \
    return sum;                                                                         \
  }

SUM_ADDITION_PRODUCT(+)
SUM_ADDITION_PRODUCT(-)

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const operator_sum<HandlerTy> &other) const {
  operator_sum<HandlerTy> sum;
  sum.coefficients.reserve(this->coefficients.size() * other.coefficients.size());
  sum.terms.reserve(this->terms.size() * other.terms.size());
  for (auto i = 0; i < this->terms.size(); ++i) {
    for (auto j = 0; j < other.terms.size(); ++j) {
      product_operator<HandlerTy> prod(this->coefficients[i], this->terms[i]);
      prod *= product_operator<HandlerTy>(other.coefficients[j], other.terms[j]);
      // if we do an insert here it can get *very* expensive, hence merge terms at the end
      sum.coefficients.push_back(std::move(prod.coefficient));
      sum.terms.push_back(std::move(prod.operators));
    }
  }
  sum.aggregate_all();
  return sum;
}

#define SUM_ADDITION_SUM(op)                                                            \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                         const operator_sum<HandlerTy> &other) const {  \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients.reserve(this->coefficients.size() + other.coefficients.size());    \
    sum.terms.reserve(this->terms.size() + other.terms.size());                         \
    for (auto &coeff : this->coefficients)                                              \
      sum.coefficients.push_back(coeff);                                                \
    for (auto &coeff : other.coefficients)                                              \
      sum.coefficients.push_back(op coeff);                                             \
    for (auto &term : this->terms)                                                      \
      sum.terms.push_back(term);                                                        \
    for (auto &term : other.terms)                                                      \
      sum.terms.push_back(term);                                                        \
    sum.aggregate_all();                                                                \
    return sum;                                                                         \
  }

SUM_ADDITION_SUM(+);
SUM_ADDITION_SUM(-);

#define INSTANTIATE_SUM_RHCOMPOSITE_OPS(HandlerTy)                                      \
                                                                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(                           \
    const product_operator<HandlerTy> &other) const;                                    \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(                           \
    const product_operator<HandlerTy> &other) const;                                    \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(                           \
    const product_operator<HandlerTy> &other) const;                                    \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(                           \
    const operator_sum<HandlerTy> &other) const;                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(                           \
    const operator_sum<HandlerTy> &other) const;                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(                           \
    const operator_sum<HandlerTy> &other) const;

INSTANTIATE_SUM_RHCOMPOSITE_OPS(matrix_operator);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(spin_operator);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(boson_operator);

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

/*
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
*/

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const product_operator<HandlerTy> &other) {
  for (auto i = 0; i < this->terms.size(); ++i) {
    product_operator<HandlerTy> prod(this->coefficients[i], this->terms[i]);
    prod *= other;
    this->coefficients[i] = std::move(prod.coefficient);
    this->terms[i] = std::move(prod.operators);
  }
  return *this;
}

#define SUM_ADDITION_PRODUCT_ASSIGNMENT(op)                                             \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(                     \
                                           const product_operator<HandlerTy> &other) {  \
    this->insert(op other);                                                             \
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

#define SUM_ADDITION_SUM_ASSIGNMENT(op)                                                 \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(                     \
                                               const operator_sum<HandlerTy> &other) {  \
    this->coefficients.reserve(this->coefficients.size() + other.coefficients.size());  \
    this->terms.reserve(this->terms.size() + other.terms.size());                       \
    for (auto &coeff : other.coefficients)                                              \
      this->coefficients.push_back(op coeff);                                           \
    for (auto &term : other.terms)                                                      \
      this->terms.push_back(term);                                                      \
    this->aggregate_all();                                                              \
    return *this;                                                                       \
  }

SUM_ADDITION_SUM_ASSIGNMENT(+);
SUM_ADDITION_SUM_ASSIGNMENT(-);

#define INSTANTIATE_SUM_OPASSIGNMENTS(HandlerTy)                                                            \
                                                                                                            \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(double other);                               \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(double other);                               \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(double other);                               \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(std::complex<double> other);                 \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(std::complex<double> other);                 \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(std::complex<double> other);                 \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const scalar_operator &other);               \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(const scalar_operator &other);               \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const scalar_operator &other);               \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const product_operator<HandlerTy> &other);   \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(const product_operator<HandlerTy> &other);   \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const product_operator<HandlerTy> &other);   \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const operator_sum<HandlerTy> &other);       \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const operator_sum<HandlerTy> &other);       \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(const operator_sum<HandlerTy> &other);

/*
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const HandlerTy &other);                     \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(const HandlerTy &other);                     \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const HandlerTy &other);                     \

*/

INSTANTIATE_SUM_OPASSIGNMENTS(matrix_operator);
INSTANTIATE_SUM_OPASSIGNMENTS(spin_operator);
INSTANTIATE_SUM_OPASSIGNMENTS(boson_operator);

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

/*
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
*/

#define INSTANTIATE_SUM_LHCOMPOSITE_OPS(HandlerTy)                                                        \
                                                                                                          \
  template                                                                                                \
  operator_sum<HandlerTy> operator*(double other, const operator_sum<HandlerTy> &self);                   \
  template                                                                                                \
  operator_sum<HandlerTy> operator+(double other, const operator_sum<HandlerTy> &self);                   \
  template                                                                                                \
  operator_sum<HandlerTy> operator-(double other, const operator_sum<HandlerTy> &self);                   \
  template                                                                                                \
  operator_sum<HandlerTy> operator*(std::complex<double> other, const operator_sum<HandlerTy> &self);     \
  template                                                                                                \
  operator_sum<HandlerTy> operator+(std::complex<double> other, const operator_sum<HandlerTy> &self);     \
  template                                                                                                \
  operator_sum<HandlerTy> operator-(std::complex<double> other, const operator_sum<HandlerTy> &self);     \
  template                                                                                                \
  operator_sum<HandlerTy> operator*(const scalar_operator &other, const operator_sum<HandlerTy> &self);   \
  template                                                                                                \
  operator_sum<HandlerTy> operator+(const scalar_operator &other, const operator_sum<HandlerTy> &self);   \
  template                                                                                                \
  operator_sum<HandlerTy> operator-(const scalar_operator &other, const operator_sum<HandlerTy> &self);   \

/*
  template                                                                                                \
  operator_sum<HandlerTy> operator*(const HandlerTy &other, const operator_sum<HandlerTy> &self);         \
  template                                                                                                \
  operator_sum<HandlerTy> operator+(const HandlerTy &other, const operator_sum<HandlerTy> &self);         \
  template                                                                                                \
  operator_sum<HandlerTy> operator-(const HandlerTy &other, const operator_sum<HandlerTy> &self);
*/

INSTANTIATE_SUM_LHCOMPOSITE_OPS(matrix_operator);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(spin_operator);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(boson_operator);

// arithmetics that require conversions

#define SUM_CONVERSIONS_OPS(op)                                                               \
                                                                                              \
  template <typename LHtype, typename RHtype,                                                 \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>                                \
  operator_sum<matrix_operator> operator op(const operator_sum<LHtype> &other,                \
                                            const product_operator<RHtype> &self) {           \
    return operator_sum<matrix_operator>(other) op self;                                      \
  }                                                                                           \
                                                                                              \
  template <typename LHtype, typename RHtype,                                                 \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>                                \
  operator_sum<matrix_operator> operator op(const product_operator<LHtype> &other,            \
                                            const operator_sum<RHtype> &self) {               \
    return product_operator<matrix_operator>(other) op self;                                  \
  }                                                                                           \
                                                                                              \
  template <typename LHtype, typename RHtype,                                                 \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>                                \
  operator_sum<matrix_operator> operator op(const operator_sum<LHtype> &other,                \
                                            const operator_sum<RHtype> &self) {               \
    return operator_sum<matrix_operator>(other) op self;                                      \
  }

SUM_CONVERSIONS_OPS(*);
SUM_CONVERSIONS_OPS(+);
SUM_CONVERSIONS_OPS(-);

#define INSTANTIATE_SUM_CONVERSION_OPS(op)                                                    \
                                                                                              \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,         \
                                            const product_operator<matrix_operator> &self);   \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,        \
                                            const product_operator<matrix_operator> &self);   \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,         \
                                            const product_operator<boson_operator> &self);    \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,        \
                                            const product_operator<spin_operator> &self);     \
                                                                                              \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<spin_operator> &other,     \
                                            const operator_sum<matrix_operator> &self);       \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<boson_operator> &other,    \
                                            const operator_sum<matrix_operator> &self);       \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<spin_operator> &other,     \
                                            const operator_sum<boson_operator> &self);        \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<boson_operator> &other,    \
                                            const operator_sum<spin_operator> &self);         \
                                                                                              \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,         \
                                            const operator_sum<matrix_operator> &self);       \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,        \
                                            const operator_sum<matrix_operator> &self);       \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,         \
                                            const operator_sum<boson_operator> &self);        \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,        \
                                            const operator_sum<spin_operator> &self);         \

INSTANTIATE_SUM_CONVERSION_OPS(*);
INSTANTIATE_SUM_CONVERSION_OPS(+);
INSTANTIATE_SUM_CONVERSION_OPS(-);

// common operators

template <typename HandlerTy>
operator_sum<HandlerTy> operator_handler::empty() {
  return operator_sum<HandlerTy>();
}

template operator_sum<matrix_operator> operator_handler::empty();
template operator_sum<spin_operator> operator_handler::empty();
template operator_sum<boson_operator> operator_handler::empty();


#ifdef CUDAQ_INSTANTIATE_TEMPLATES
template class operator_sum<matrix_operator>;
template class operator_sum<spin_operator>;
template class operator_sum<boson_operator>;
#endif

} // namespace cudaq