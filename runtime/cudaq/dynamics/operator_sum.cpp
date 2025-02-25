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

#include "helpers.h"
#include "evaluation.h"
#include "cudaq/operators.h"
#include "cudaq/spin_op.h"

namespace cudaq {

// private methods

template<typename HandlerTy>
void operator_sum<HandlerTy>::insert(const product_operator<HandlerTy> &other) {
  auto term_id = other.get_term_id();
  auto it = this->term_map.find(term_id);
  if (it == this->term_map.cend()) {
    this->coefficients.push_back(other.coefficient);
    this->term_map.insert(it, std::make_pair(std::move(term_id), this->terms.size()));
    this->terms.push_back(other.operators);
  } else {
    this->coefficients[it->second] += other.coefficient;
  }
}

template<typename HandlerTy>
void operator_sum<HandlerTy>::insert(product_operator<HandlerTy> &&other) {
  auto term_id = other.get_term_id();
  auto it = this->term_map.find(term_id);
  if (it == this->term_map.cend()) {
    this->coefficients.push_back(std::move(other.coefficient));
    this->term_map.insert(it, std::make_pair(std::move(term_id), this->terms.size()));
    this->terms.push_back(std::move(other.operators));
  } else {
    this->coefficients[it->second] += other.coefficient;
  }
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
template <typename EvalTy>
EvalTy operator_sum<HandlerTy>::evaluate(operator_arithmetics<EvalTy> arithmetics) const {

  if (terms.size() == 0) return EvalTy();
  auto terms = this->get_terms();
  auto degrees = this->degrees();

  // Adding a tensor product with the identity for degrees that an operator doesn't act on.
  // Needed e.g. to make sure all matrices are of the same size before summing them up.
  auto paddedTerm = 
    [&arithmetics, &degrees = std::as_const(degrees)](product_operator<HandlerTy> &&term) {
      std::vector<HandlerTy> prod_ops;
      prod_ops.reserve(degrees.size());
      auto term_degrees = term.degrees();
      for (auto degree : degrees) {
        auto it = std::find(term_degrees.begin(), term_degrees.end(), degree);
        if (it == term_degrees.end()) {
          HandlerTy identity(degree);
          prod_ops.push_back(std::move(identity));
        }
      }
      product_operator<HandlerTy> prod(1, std::move(prod_ops));
      prod *= term; // ensures canonical ordering (if possible)
      return prod;
  };

  if (arithmetics.pad_sum_terms) {
    product_operator<HandlerTy> padded_term = paddedTerm(std::move(terms[0]));
    EvalTy sum = padded_term.template evaluate<EvalTy>(arithmetics);
    for (auto term_idx = 1; term_idx < terms.size(); ++term_idx) {
      padded_term = paddedTerm(std::move(terms[term_idx]));
      EvalTy term_eval = padded_term.template evaluate<EvalTy>(arithmetics);
      sum = arithmetics.add(std::move(sum), std::move(term_eval));
    }
    return sum;
  } else {
    EvalTy sum = terms[0].template evaluate<EvalTy>(arithmetics);
    for (auto term_idx = 1; term_idx < terms.size(); ++term_idx) {
      EvalTy term_eval = terms[term_idx].template evaluate<EvalTy>(arithmetics);
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

INSTANTIATE_SUM_PRIVATE_METHODS(matrix_operator);
INSTANTIATE_SUM_PRIVATE_METHODS(spin_operator);
INSTANTIATE_SUM_PRIVATE_METHODS(boson_operator);
INSTANTIATE_SUM_PRIVATE_METHODS(fermion_operator);

#define INSTANTIATE_SUM_EVALUATE_METHODS(HandlerTy, EvalTy)                                   \
                                                                                              \
  template                                                                                    \
  EvalTy operator_sum<HandlerTy>::evaluate(                                                   \
    operator_arithmetics<EvalTy> arithmetics) const;                                          \

INSTANTIATE_SUM_EVALUATE_METHODS(matrix_operator, operator_handler::matrix_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(spin_operator, operator_handler::canonical_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(boson_operator, operator_handler::matrix_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(fermion_operator, operator_handler::matrix_evaluation);

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
  std::sort(degrees.begin(), degrees.end(), operator_handler::canonical_order);
  return std::move(degrees);
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
  return std::move(prods); 
}

#define INSTANTIATE_SUM_PROPERTIES(HandlerTy)                                               \
                                                                                            \
  template                                                                                  \
  std::vector<int> operator_sum<HandlerTy>::degrees() const;                                \
                                                                                            \
  template                                                                                  \
  int operator_sum<HandlerTy>::num_terms() const;                                           \
                                                                                            \
  template                                                                                  \
  std::vector<product_operator<HandlerTy>> operator_sum<HandlerTy>::get_terms() const;

INSTANTIATE_SUM_PROPERTIES(matrix_operator);
INSTANTIATE_SUM_PROPERTIES(spin_operator);
INSTANTIATE_SUM_PROPERTIES(boson_operator);
INSTANTIATE_SUM_PROPERTIES(fermion_operator);

// constructors

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(const product_operator<HandlerTy> &prod) {
  this->insert(prod);
}

template<typename HandlerTy>
template<typename... Args, std::enable_if_t<std::conjunction<std::is_same<product_operator<HandlerTy>, Args>...>::value, bool>>
operator_sum<HandlerTy>::operator_sum(Args&&... args) {
  this->coefficients.reserve(sizeof...(Args));
  this->term_map.reserve(sizeof...(Args));
  this->terms.reserve(sizeof...(Args));
  aggregate_terms(std::forward<product_operator<HandlerTy>&&>(args)...);
}

template<typename HandlerTy>
template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool>>
operator_sum<HandlerTy>::operator_sum(const operator_sum<T> &other) 
: coefficients(other.coefficients) {
  this->term_map.reserve(other.terms.size());
  this->terms.reserve(other.terms.size());
  for (const auto &operators : other.terms) {
    product_operator<HandlerTy> term(product_operator<T>(1., operators)); // coefficient does not matter
    this->term_map.insert(this->term_map.cend(), std::make_pair(term.get_term_id(), this->terms.size()));
    this->terms.push_back(std::move(term.operators));
  }
}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(const operator_sum<HandlerTy> &other, int size) {
  if (size <= 0) {
    this->coefficients = other.coefficients;
    this->term_map = other.term_map;
    this->terms = other.terms;
  } else {
    this->coefficients.reserve(size);
    this->term_map.reserve(size);
    this->terms.reserve(size);
    for (const auto &coeff : other.coefficients)
      this->coefficients.push_back(coeff);
    for (const auto &entry : other.term_map)
      this->term_map.insert(this->term_map.cend(), entry);
    for (const auto &term : other.terms)
      this->terms.push_back(term);
  }
}

template<typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(operator_sum<HandlerTy> &&other, int size)
: coefficients(std::move(other.coefficients)), term_map(std::move(other.term_map)), terms(std::move(other.terms)) {
  if (size > 0) {
    this->coefficients.reserve(size);
    this->term_map.reserve(size);
    this->terms.reserve(size);
  }
}

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
  operator_sum<HandlerTy>::operator_sum(const operator_sum<HandlerTy> &other, int size);        \
                                                                                                \
  template                                                                                      \
  operator_sum<HandlerTy>::operator_sum(operator_sum<HandlerTy> &&other, int size);

template 
operator_sum<matrix_operator>::operator_sum(const operator_sum<spin_operator> &other);
template 
operator_sum<matrix_operator>::operator_sum(const operator_sum<boson_operator> &other);
template 
operator_sum<matrix_operator>::operator_sum(const operator_sum<fermion_operator> &other);

INSTANTIATE_SUM_CONSTRUCTORS(matrix_operator);
INSTANTIATE_SUM_CONSTRUCTORS(spin_operator);
INSTANTIATE_SUM_CONSTRUCTORS(boson_operator);
INSTANTIATE_SUM_CONSTRUCTORS(fermion_operator);

// assignments

template<typename HandlerTy>
  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool>>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(const product_operator<T> &other) {
  *this = product_operator<HandlerTy>(other);
  return *this;
}

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(const product_operator<HandlerTy> &other) {
  this->coefficients.clear();
  this->term_map.clear();
  this->terms.clear();
  this->coefficients.push_back(other.coefficient);
  this->term_map.insert(this->term_map.cend(), std::make_pair(other.get_term_id(), 0));
  this->terms.push_back(other.operators);
  return *this;
}

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(product_operator<HandlerTy> &&other) {
  this->coefficients.clear();
  this->term_map.clear();
  this->terms.clear();
  this->coefficients.push_back(std::move(other.coefficient));
  this->term_map.insert(this->term_map.cend(), std::make_pair(other.get_term_id(), 0));
  this->terms.push_back(std::move(other.operators));
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
    this->term_map = other.term_map;
    this->terms = other.terms;
  }
  return *this;
}

template<typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(operator_sum<HandlerTy> &&other) {
  if (this != &other) {
    this->coefficients = std::move(other.coefficients);
    this->term_map = std::move(other.term_map);
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
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator=(const product_operator<fermion_operator> &other);
template 
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator=(const operator_sum<spin_operator> &other);
template 
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator=(const operator_sum<boson_operator> &other);
template 
operator_sum<matrix_operator>& operator_sum<matrix_operator>::operator=(const operator_sum<fermion_operator> &other);

INSTANTIATE_SUM_ASSIGNMENTS(matrix_operator);
INSTANTIATE_SUM_ASSIGNMENTS(spin_operator);
INSTANTIATE_SUM_ASSIGNMENTS(boson_operator);
INSTANTIATE_SUM_ASSIGNMENTS(fermion_operator);

// evaluations

template<typename HandlerTy>
std::string operator_sum<HandlerTy>::to_string() const {
  auto prods = this->get_terms();
  auto it = prods.cbegin();
  std::string str = it->to_string();
  while (++it != prods.cend())
    str += " + " + it->to_string();
  return std::move(str);
}

template<typename HandlerTy>
matrix_2 operator_sum<HandlerTy>::to_matrix(std::unordered_map<int, int> dimensions,
                                            const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  return this->evaluate(matrix_arithmetics(dimensions, parameters)).matrix();
}

template<>
matrix_2 operator_sum<spin_operator>::to_matrix(std::unordered_map<int, int> dimensions,
                                            const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  auto evaluated = this->evaluate(canonical_arithmetics(dimensions, parameters));
  auto terms = evaluated.get_terms();
  if (terms.size() == 0) return cudaq::matrix_2(0, 0);

  auto matrix = spin_operator::to_matrix(terms[0].second, terms[0].first);
  for (auto i = 1; i < terms.size(); ++i) 
     matrix += spin_operator::to_matrix(terms[i].second, terms[i].first);
  return std::move(matrix);
}

#define INSTANTIATE_SUM_EVALUATIONS(HandlerTy)                                              \
                                                                                            \
  template                                                                                  \
  std::string operator_sum<HandlerTy>::to_string() const;                                   \
                                                                                            \
  template                                                                                  \
  matrix_2 operator_sum<HandlerTy>::to_matrix(                                              \
    std::unordered_map<int, int> dimensions,                                                \
    const std::unordered_map<std::string, std::complex<double>> &params) const;

INSTANTIATE_SUM_EVALUATIONS(matrix_operator);
INSTANTIATE_SUM_EVALUATIONS(spin_operator);
INSTANTIATE_SUM_EVALUATIONS(boson_operator);
INSTANTIATE_SUM_EVALUATIONS(fermion_operator);

// unary operators

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-() const & {
  operator_sum<HandlerTy> sum;
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map = this->term_map;
  sum.terms = this->terms;
  for (auto &coeff : this->coefficients)
    sum.coefficients.push_back(-1. * coeff);
  return std::move(sum);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-() && {
  for (auto &coeff : this->coefficients)
    coeff *= -1.;
  return std::move(*this);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+() const & {
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+() && {
  return std::move(*this);
}

#define INSTANTIATE_SUM_UNARY_OPS(HandlerTy)                                            \
                                                                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-() const &;                 \
                                                                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-() &&;                      \
                                                                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+() const &;                 \
                                                                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+() &&;                      \

INSTANTIATE_SUM_UNARY_OPS(matrix_operator);
INSTANTIATE_SUM_UNARY_OPS(spin_operator);
INSTANTIATE_SUM_UNARY_OPS(boson_operator);
INSTANTIATE_SUM_UNARY_OPS(fermion_operator);

// right-hand arithmetics

#define SUM_MULTIPLICATION(otherTy)                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(otherTy other) const & {   \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients.reserve(this->coefficients.size());                                \
    sum.term_map = this->term_map;                                                      \
    sum.terms = this->terms;                                                            \
    for (const auto &coeff : this->coefficients)                                        \
      sum.coefficients.push_back(coeff * other);                                        \
    return std::move(sum);                                                              \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(otherTy other) && {        \
    for (auto &coeff : this->coefficients)                                              \
      coeff *= other;                                                                   \
    return std::move(*this);                                                            \
  }

SUM_MULTIPLICATION(double);
SUM_MULTIPLICATION(std::complex<double>);
SUM_MULTIPLICATION(const scalar_operator &);

#define SUM_ADDITION(otherTy, op)                                                       \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(otherTy other) const & { \
    operator_sum<HandlerTy> sum(*this, this->terms.size() + 1);                         \
    sum.insert(product_operator<HandlerTy>(op other));                                  \
    return std::move(sum);                                                              \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(otherTy other) && {      \
    this->insert(product_operator<HandlerTy>(op other));                                \
    return std::move(*this);                                                            \
  }                                                                                     \

SUM_ADDITION(double, +);
SUM_ADDITION(double, -);
SUM_ADDITION(std::complex<double>, +);
SUM_ADDITION(std::complex<double>, -);
SUM_ADDITION(const scalar_operator &, +);
SUM_ADDITION(const scalar_operator &, -);

#define INSTANTIATE_SUM_RHSIMPLE_OPS(HandlerTy)                                                     \
                                                                                                    \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(double other) const &;                 \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(double other) &&;                      \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(double other) const &;                 \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(double other) &&;                      \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(double other) const &;                 \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(double other) &&;                      \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(std::complex<double> other) const &;   \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(std::complex<double> other) &&;        \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(std::complex<double> other) const &;   \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(std::complex<double> other) &&;        \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(std::complex<double> other) const &;   \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(std::complex<double> other) &&;        \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const scalar_operator &other) const &; \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const scalar_operator &other) &&;      \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const scalar_operator &other) const &; \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const scalar_operator &other) &&;      \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const scalar_operator &other) const &; \
  template                                                                                          \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const scalar_operator &other) &&; 

INSTANTIATE_SUM_RHSIMPLE_OPS(matrix_operator);
INSTANTIATE_SUM_RHSIMPLE_OPS(spin_operator);
INSTANTIATE_SUM_RHSIMPLE_OPS(boson_operator);
INSTANTIATE_SUM_RHSIMPLE_OPS(fermion_operator);

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const product_operator<HandlerTy> &other) const {
  operator_sum<HandlerTy> sum; // the entire sum needs to be rebuilt
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map.reserve(this->terms.size());
  sum.terms.reserve(this->terms.size());
  for (auto i = 0; i < this->terms.size(); ++i) {
    auto max_size = this->terms[i].size() + other.operators.size();
    product_operator<HandlerTy> prod(this->coefficients[i] * other.coefficient, this->terms[i], max_size);
    for (HandlerTy op : other.operators)
      prod.insert(std::move(op));
    sum.insert(std::move(prod));
  }
  return std::move(sum);
}

#define SUM_ADDITION_PRODUCT(op)                                                        \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                    const product_operator<HandlerTy> &other) const & { \
    operator_sum<HandlerTy> sum(*this, this->terms.size() + 1);                         \
    sum.insert(op other);                                                               \
    return std::move(sum);                                                              \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                     const product_operator<HandlerTy> &other) && {     \
    this->insert(op other);                                                             \
    return std::move(*this);                                                            \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                         product_operator<HandlerTy> &&other) const & { \
    operator_sum<HandlerTy> sum(*this, this->terms.size() + 1);                         \
    sum.insert(op std::move(other));                                                    \
    return std::move(sum);                                                              \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                          product_operator<HandlerTy> &&other) && {     \
    this->insert(op std::move(other));                                                  \
    return std::move(*this);                                                            \
  }                                                                                     \

SUM_ADDITION_PRODUCT(+)
SUM_ADDITION_PRODUCT(-)

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const operator_sum<HandlerTy> &other) const {
  operator_sum<HandlerTy> sum; // the entire sum needs to be rebuilt
  auto max_size = this->terms.size() * other.terms.size();
  sum.coefficients.reserve(max_size);
  sum.term_map.reserve(max_size);
  sum.terms.reserve(max_size);
  for (auto i = 0; i < this->terms.size(); ++i) {
    for (auto j = 0; j < other.terms.size(); ++j) {
      auto max_size = this->terms[i].size() + other.terms[j].size();
      product_operator<HandlerTy> prod(this->coefficients[i] * other.coefficients[j], this->terms[i], max_size);
      for (HandlerTy op : other.terms[j])
        prod.insert(std::move(op));
      sum.insert(std::move(prod));
    }
  }
  return std::move(sum);
}

#define SUM_ADDITION_SUM(op)                                                            \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                      const operator_sum<HandlerTy> &other) const & {   \
    operator_sum<HandlerTy> sum(*this, this->terms.size() + other.terms.size());        \
    for (auto i = 0; i < other.terms.size(); ++i) {                                     \
      product_operator<HandlerTy> prod(op other.coefficients[i], other.terms[i]);       \
      sum.insert(std::move(prod));                                                      \
    }                                                                                   \
    return std::move(sum);                                                              \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                      const operator_sum<HandlerTy> &other) && {        \
    auto max_size = this->terms.size() + other.terms.size();                            \
    this->coefficients.reserve(max_size);                                               \
    this->term_map.reserve(max_size);                                                   \
    this->terms.reserve(max_size);                                                      \
    for (auto i = 0; i < other.terms.size(); ++i)                                       \
      this->insert(product_operator<HandlerTy>(                                         \
                   op other.coefficients[i], other.terms[i]));                          \
    return std::move(*this);                                                            \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                           operator_sum<HandlerTy> &&other) const & {   \
    operator_sum<HandlerTy> sum(*this, this->terms.size() + other.terms.size());        \
    for (auto i = 0; i < other.terms.size(); ++i) {                                     \
      product_operator<HandlerTy> prod(                                                 \
        op std::move(other.coefficients[i]), std::move(other.terms[i]));                \
      sum.insert(std::move(prod));                                                      \
    }                                                                                   \
    return std::move(sum);                                                              \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                         \
                                           operator_sum<HandlerTy> &&other) && {        \
    auto max_size = this->terms.size() + other.terms.size();                            \
    this->coefficients.reserve(max_size);                                               \
    this->term_map.reserve(max_size);                                                   \
    this->terms.reserve(max_size);                                                      \
    for (auto i = 0; i < other.terms.size(); ++i)                                       \
      this->insert(product_operator<HandlerTy>(                                         \
        op std::move(other.coefficients[i]), std::move(other.terms[i])));               \
    return std::move(*this);                                                            \
  }                                                                                     \

SUM_ADDITION_SUM(+);
SUM_ADDITION_SUM(-);

#define INSTANTIATE_SUM_RHCOMPOSITE_OPS(HandlerTy)                                      \
                                                                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(                           \
    const product_operator<HandlerTy> &other) const;                                    \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(                           \
    const product_operator<HandlerTy> &other) const &;                                  \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(                           \
    const product_operator<HandlerTy> &other) &&;                                       \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(                           \
    product_operator<HandlerTy> &&other) const &;                                       \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(                           \
    product_operator<HandlerTy> &&other) &&;                                            \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(                           \
    const product_operator<HandlerTy> &other) const &;                                  \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(                           \
    const product_operator<HandlerTy> &other) &&;                                       \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(                           \
    product_operator<HandlerTy> &&other) const &;                                       \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(                           \
    product_operator<HandlerTy> &&other) &&;                                            \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(                           \
    const operator_sum<HandlerTy> &other) const;                                        \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(                           \
    const operator_sum<HandlerTy> &other) const &;                                      \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(                           \
    const operator_sum<HandlerTy> &other) &&;                                           \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(                           \
    operator_sum<HandlerTy> &&other) const &;                                           \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(                           \
    operator_sum<HandlerTy> &&other) &&;                                                \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(                           \
    const operator_sum<HandlerTy> &other) const &;                                      \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(                           \
    const operator_sum<HandlerTy> &other) &&;                                           \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(                           \
    operator_sum<HandlerTy> &&other) const &;                                           \
  template                                                                              \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(                           \
    operator_sum<HandlerTy> &&other) &&;                                                \

INSTANTIATE_SUM_RHCOMPOSITE_OPS(matrix_operator);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(spin_operator);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(boson_operator);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(fermion_operator);

#define SUM_MULTIPLICATION_ASSIGNMENT(otherTy)                                          \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(otherTy other) {         \
    for (auto &coeff : this->coefficients)                                              \
      coeff *= other;                                                                   \
    return *this;                                                                       \
  }                                                                                     \

SUM_MULTIPLICATION_ASSIGNMENT(double);
SUM_MULTIPLICATION_ASSIGNMENT(std::complex<double>);
SUM_MULTIPLICATION_ASSIGNMENT(const scalar_operator &);

#define SUM_ADDITION_ASSIGNMENT(otherTy, op)                                            \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(otherTy other) {     \
    this->insert(product_operator<HandlerTy>(op other));                                \
    return *this;                                                                       \
  }                                                                                     \

SUM_ADDITION_ASSIGNMENT(double, +);
SUM_ADDITION_ASSIGNMENT(double, -);
SUM_ADDITION_ASSIGNMENT(std::complex<double>, +);
SUM_ADDITION_ASSIGNMENT(std::complex<double>, -);
SUM_ADDITION_ASSIGNMENT(const scalar_operator &, +);
SUM_ADDITION_ASSIGNMENT(const scalar_operator &, -);

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const product_operator<HandlerTy> &other) {
  operator_sum<HandlerTy> sum;
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map.reserve(this->terms.size());
  sum.terms.reserve(this->terms.size());
  for (auto i = 0; i < this->terms.size(); ++i) {
    auto max_size = this->terms[i].size() + other.operators.size();
    product_operator<HandlerTy> prod(this->coefficients[i] * other.coefficient, this->terms[i], max_size);
    for (HandlerTy op : other.operators)
      prod.insert(std::move(op));
    sum.insert(std::move(prod));
  }
  *this = std::move(sum);
  return *this;
}

#define SUM_ADDITION_PRODUCT_ASSIGNMENT(op)                                             \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(                     \
                                           const product_operator<HandlerTy> &other) {  \
    this->insert(op other);                                                             \
    return *this;                                                                       \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(                     \
                                                product_operator<HandlerTy> &&other) {  \
    this->insert(op std::move(other));                                                  \
    return *this;                                                                       \
  }                                                                                     \

SUM_ADDITION_PRODUCT_ASSIGNMENT(+)
SUM_ADDITION_PRODUCT_ASSIGNMENT(-)

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const operator_sum<HandlerTy> &other) {
  operator_sum<HandlerTy> sum; // the entire sum needs to be rebuilt
  auto max_size = this->terms.size() * other.terms.size();
  sum.coefficients.reserve(max_size);
  sum.term_map.reserve(max_size);
  sum.terms.reserve(max_size);
  for (auto i = 0; i < this->terms.size(); ++i) {
    for (auto j = 0; j < other.terms.size(); ++j) {
      auto max_size = this->terms[i].size() + other.terms[j].size();
      product_operator<HandlerTy> prod(this->coefficients[i] * other.coefficients[j], this->terms[i], max_size);
      for (HandlerTy op : other.terms[j])
        prod.insert(std::move(op));
      sum.insert(std::move(prod));
    }
  }
  *this = std::move(sum);
  return *this;
}

#define SUM_ADDITION_SUM_ASSIGNMENT(op)                                                 \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(                     \
                                               const operator_sum<HandlerTy> &other) {  \
    auto max_size = this->terms.size() + other.terms.size();                            \
    this->coefficients.reserve(max_size);                                               \
    this->term_map.reserve(max_size);                                                   \
    this->terms.reserve(max_size);                                                      \
    for (auto i = 0; i < other.terms.size(); ++i)                                       \
      this->insert(product_operator<HandlerTy>(                                         \
                   op other.coefficients[i], other.terms[i]));                          \
    return *this;                                                                       \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator op##=(                     \
                                                    operator_sum<HandlerTy> &&other) {  \
    auto max_size = this->terms.size() + other.terms.size();                            \
    this->coefficients.reserve(max_size);                                               \
    this->term_map.reserve(max_size);                                                   \
    this->terms.reserve(max_size);                                                      \
    for (auto i = 0; i < other.terms.size(); ++i)                                       \
      this->insert(product_operator<HandlerTy>(                                         \
        op std::move(other.coefficients[i]), std::move(other.terms[i])));               \
    return *this;                                                                       \
  }                                                                                     \

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
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(product_operator<HandlerTy> &&other);        \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const product_operator<HandlerTy> &other);   \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(product_operator<HandlerTy> &&other);        \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const operator_sum<HandlerTy> &other);       \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(const operator_sum<HandlerTy> &other);       \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(operator_sum<HandlerTy> &&other);            \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const operator_sum<HandlerTy> &other);       \
  template                                                                                                  \
  operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(operator_sum<HandlerTy> &&other);            \

INSTANTIATE_SUM_OPASSIGNMENTS(matrix_operator);
INSTANTIATE_SUM_OPASSIGNMENTS(spin_operator);
INSTANTIATE_SUM_OPASSIGNMENTS(boson_operator);
INSTANTIATE_SUM_OPASSIGNMENTS(fermion_operator);

// left-hand arithmetics

#define SUM_MULTIPLICATION_REVERSE(otherTy)                                             \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator*(otherTy other,                                      \
                                    const operator_sum<HandlerTy> &self) {              \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients.reserve(self.coefficients.size());                                 \
    sum.terms = self.terms;                                                             \
    sum.term_map = self.term_map;                                                       \
    for (const auto &coeff : self.coefficients)                                         \
      sum.coefficients.push_back(coeff * other);                                        \
    return std::move(sum);                                                              \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator*(otherTy other,                                      \
                                    operator_sum<HandlerTy> &&self) {                   \
    for (auto &&coeff : self.coefficients)                                              \
      coeff *= other;                                                                   \
    return std::move(self);                                                             \
  }                                                                                     \

SUM_MULTIPLICATION_REVERSE(double);
SUM_MULTIPLICATION_REVERSE(std::complex<double>);
SUM_MULTIPLICATION_REVERSE(const scalar_operator &);

#define SUM_ADDITION_REVERSE(otherTy, op)                                               \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator op(otherTy other,                                    \
                                      const operator_sum<HandlerTy> &self) {            \
    operator_sum<HandlerTy> sum(op self);                                               \
    sum.insert(product_operator<HandlerTy>(other));                                     \
    return std::move(sum);                                                              \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator op(otherTy other,                                    \
                                      operator_sum<HandlerTy> &&self) {                 \
    for (auto &&coeff : self.coefficients)                                              \
      coeff = std::move(op coeff);                                                      \
    self.insert(product_operator<HandlerTy>(other));                                    \
    return std::move(self);                                                             \
  }                                                                                     \

SUM_ADDITION_REVERSE(double, +);
SUM_ADDITION_REVERSE(double, -);
SUM_ADDITION_REVERSE(std::complex<double>, +);
SUM_ADDITION_REVERSE(std::complex<double>, -);
SUM_ADDITION_REVERSE(const scalar_operator &, +);
SUM_ADDITION_REVERSE(const scalar_operator &, -);

#define INSTANTIATE_SUM_LHCOMPOSITE_OPS(HandlerTy)                                                        \
                                                                                                          \
  template                                                                                                \
  operator_sum<HandlerTy> operator*(double other, const operator_sum<HandlerTy> &self);                   \
  template                                                                                                \
  operator_sum<HandlerTy> operator*(double other, operator_sum<HandlerTy> &&self);                        \
  template                                                                                                \
  operator_sum<HandlerTy> operator+(double other, const operator_sum<HandlerTy> &self);                   \
  template                                                                                                \
  operator_sum<HandlerTy> operator+(double other, operator_sum<HandlerTy> &&self);                        \
  template                                                                                                \
  operator_sum<HandlerTy> operator-(double other, const operator_sum<HandlerTy> &self);                   \
  template                                                                                                \
  operator_sum<HandlerTy> operator-(double other, operator_sum<HandlerTy> &&self);                        \
  template                                                                                                \
  operator_sum<HandlerTy> operator*(std::complex<double> other, const operator_sum<HandlerTy> &self);     \
  template                                                                                                \
  operator_sum<HandlerTy> operator*(std::complex<double> other, operator_sum<HandlerTy> &&self);          \
  template                                                                                                \
  operator_sum<HandlerTy> operator+(std::complex<double> other, const operator_sum<HandlerTy> &self);     \
  template                                                                                                \
  operator_sum<HandlerTy> operator+(std::complex<double> other, operator_sum<HandlerTy> &&self);          \
  template                                                                                                \
  operator_sum<HandlerTy> operator-(std::complex<double> other, const operator_sum<HandlerTy> &self);     \
  template                                                                                                \
  operator_sum<HandlerTy> operator-(std::complex<double> other, operator_sum<HandlerTy> &&self);          \
  template                                                                                                \
  operator_sum<HandlerTy> operator*(const scalar_operator &other, const operator_sum<HandlerTy> &self);   \
  template                                                                                                \
  operator_sum<HandlerTy> operator*(const scalar_operator &other, operator_sum<HandlerTy> &&self);        \
  template                                                                                                \
  operator_sum<HandlerTy> operator+(const scalar_operator &other, const operator_sum<HandlerTy> &self);   \
  template                                                                                                \
  operator_sum<HandlerTy> operator+(const scalar_operator &other, operator_sum<HandlerTy> &&self);        \
  template                                                                                                \
  operator_sum<HandlerTy> operator-(const scalar_operator &other, const operator_sum<HandlerTy> &self);   \
  template                                                                                                \
  operator_sum<HandlerTy> operator-(const scalar_operator &other, operator_sum<HandlerTy> &&self);        \

INSTANTIATE_SUM_LHCOMPOSITE_OPS(matrix_operator);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(spin_operator);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(boson_operator);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(fermion_operator);

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
  operator_sum<matrix_operator> operator op(const operator_sum<fermion_operator> &other,      \
                                            const product_operator<matrix_operator> &self);   \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,         \
                                            const product_operator<boson_operator> &self);    \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,        \
                                            const product_operator<spin_operator> &self);     \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,         \
                                            const product_operator<fermion_operator> &self);  \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<fermion_operator> &other,      \
                                            const product_operator<spin_operator> &self);     \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,        \
                                            const product_operator<fermion_operator> &self);  \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<fermion_operator> &other,      \
                                            const product_operator<boson_operator> &self);    \
                                                                                              \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<spin_operator> &other,     \
                                            const operator_sum<matrix_operator> &self);       \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<boson_operator> &other,    \
                                            const operator_sum<matrix_operator> &self);       \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<fermion_operator> &other,  \
                                            const operator_sum<matrix_operator> &self);       \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<spin_operator> &other,     \
                                            const operator_sum<boson_operator> &self);        \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<boson_operator> &other,    \
                                            const operator_sum<spin_operator> &self);         \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<spin_operator> &other,     \
                                            const operator_sum<fermion_operator> &self);      \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<fermion_operator> &other,  \
                                            const operator_sum<spin_operator> &self);         \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<boson_operator> &other,    \
                                            const operator_sum<fermion_operator> &self);      \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const product_operator<fermion_operator> &other,  \
                                            const operator_sum<boson_operator> &self);        \
                                                                                              \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,         \
                                            const operator_sum<matrix_operator> &self);       \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,        \
                                            const operator_sum<matrix_operator> &self);       \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<fermion_operator> &other,      \
                                            const operator_sum<matrix_operator> &self);       \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,         \
                                            const operator_sum<boson_operator> &self);        \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,        \
                                            const operator_sum<spin_operator> &self);         \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,         \
                                            const operator_sum<fermion_operator> &self);      \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<fermion_operator> &other,      \
                                            const operator_sum<spin_operator> &self);         \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,        \
                                            const operator_sum<fermion_operator> &self);      \
  template                                                                                    \
  operator_sum<matrix_operator> operator op(const operator_sum<fermion_operator> &other,      \
                                            const operator_sum<boson_operator> &self);        \

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
template operator_sum<fermion_operator> operator_handler::empty();


#ifdef CUDAQ_INSTANTIATE_TEMPLATES
template class operator_sum<matrix_operator>;
template class operator_sum<spin_operator>;
template class operator_sum<boson_operator>;
template class operator_sum<fermion_operator>;
#endif

} // namespace cudaq