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
#include <utility>

#include "cudaq/operators.h"
#include "evaluation.h"
#include "helpers.h"

namespace cudaq {

// private methods

template <typename HandlerTy>
void operator_sum<HandlerTy>::insert(const product_operator<HandlerTy> &other) {
  auto term_id = other.get_term_id();
  auto it = this->term_map.find(term_id);
  if (it == this->term_map.cend()) {
    this->coefficients.push_back(other.coefficient);
    this->term_map.insert(
        it, std::make_pair(std::move(term_id), this->terms.size()));
    this->terms.push_back(other.operators);
  } else {
    this->coefficients[it->second] += other.coefficient;
  }
}

template <typename HandlerTy>
void operator_sum<HandlerTy>::insert(product_operator<HandlerTy> &&other) {
  auto term_id = other.get_term_id();
  auto it = this->term_map.find(term_id);
  if (it == this->term_map.cend()) {
    this->coefficients.push_back(std::move(other.coefficient));
    this->term_map.insert(
        it, std::make_pair(std::move(term_id), this->terms.size()));
    this->terms.push_back(std::move(other.operators));
  } else {
    this->coefficients[it->second] += other.coefficient;
  }
}

template <typename HandlerTy>
void operator_sum<HandlerTy>::aggregate_terms() {}

template <typename HandlerTy>
template <typename... Args>
void operator_sum<HandlerTy>::aggregate_terms(
    product_operator<HandlerTy> &&head, Args &&...args) {
  this->insert(std::forward<product_operator<HandlerTy>>(head));
  aggregate_terms(std::forward<Args>(args)...);
}

template <typename HandlerTy>
template <typename EvalTy>
EvalTy operator_sum<HandlerTy>::evaluate(
    operator_arithmetics<EvalTy> arithmetics) const {

  if (terms.size() == 0)
    return EvalTy();

  // NOTE: It is important that we evaluate the terms in a specific order,
  // otherwise the evaluation is not consistent with other methods.
  // The specific order does not matter, as long as all methods use the same term order.
  auto terms = this->get_terms();
  auto degrees = this->degrees(false); // keep in canonical order

  // Adding a tensor product with the identity for degrees that an operator
  // doesn't act on. Needed e.g. to make sure all matrices are of the same size
  // before summing them up.
  auto paddedTerm = [&arithmetics, &degrees = std::as_const(degrees)](
                        product_operator<HandlerTy> &&term) {
    std::vector<HandlerTy> prod_ops;
    prod_ops.reserve(degrees.size());
    auto term_degrees =
        term.degrees(false); // ordering does not really matter here
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

#define INSTANTIATE_SUM_PRIVATE_METHODS(HandlerTy)                             \
                                                                               \
  template void operator_sum<HandlerTy>::insert(                               \
      product_operator<HandlerTy> &&other);                                    \
                                                                               \
  template void operator_sum<HandlerTy>::insert(                               \
      const product_operator<HandlerTy> &other);                               \
                                                                               \
  template void operator_sum<HandlerTy>::aggregate_terms(                      \
      product_operator<HandlerTy> &&item2);                                    \
                                                                               \
  template void operator_sum<HandlerTy>::aggregate_terms(                      \
      product_operator<HandlerTy> &&item1,                                     \
      product_operator<HandlerTy> &&item2);                                    \
                                                                               \
  template void operator_sum<HandlerTy>::aggregate_terms(                      \
      product_operator<HandlerTy> &&item1,                                     \
      product_operator<HandlerTy> &&item2,                                     \
      product_operator<HandlerTy> &&item3);

#if !defined(__clang__)
INSTANTIATE_SUM_PRIVATE_METHODS(matrix_operator);
INSTANTIATE_SUM_PRIVATE_METHODS(spin_operator);
INSTANTIATE_SUM_PRIVATE_METHODS(boson_operator);
INSTANTIATE_SUM_PRIVATE_METHODS(fermion_operator);
#endif

#define INSTANTIATE_SUM_EVALUATE_METHODS(HandlerTy, EvalTy)                    \
                                                                               \
  template EvalTy operator_sum<HandlerTy>::evaluate(                           \
      operator_arithmetics<EvalTy> arithmetics) const;

#if !defined(__clang__)
INSTANTIATE_SUM_EVALUATE_METHODS(matrix_operator,
                                 operator_handler::matrix_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(spin_operator,
                                 operator_handler::canonical_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(boson_operator,
                                 operator_handler::matrix_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(fermion_operator,
                                 operator_handler::matrix_evaluation);
#endif

// read-only properties

template <typename HandlerTy>
std::vector<std::size_t>
operator_sum<HandlerTy>::degrees(bool application_order) const {
  std::set<std::size_t> unsorted_degrees;
  for (const std::vector<HandlerTy> &term : this->terms) {
    for (const HandlerTy &op : term) {
      auto op_degrees = op.degrees();
      unsorted_degrees.insert(op_degrees.cbegin(), op_degrees.cend());
    }
  }
  auto degrees =
      std::vector<std::size_t>(unsorted_degrees.cbegin(), unsorted_degrees.cend());
  if (application_order)
    std::sort(degrees.begin(), degrees.end(),
              operator_handler::user_facing_order);
  else
    std::sort(degrees.begin(), degrees.end(),
              operator_handler::canonical_order);
  return std::move(degrees);
}

template <typename HandlerTy>
std::size_t operator_sum<HandlerTy>::num_terms() const {
  return this->terms.size();
}

template <typename HandlerTy>
std::vector<product_operator<HandlerTy>>
operator_sum<HandlerTy>::get_terms() const {
  std::vector<product_operator<HandlerTy>> prods;
  prods.reserve(this->terms.size());
  for (size_t i = 0; i < this->terms.size(); ++i) {
    prods.push_back(
        product_operator<HandlerTy>(this->coefficients[i], this->terms[i]));
  }
  return std::move(prods);
}

#define INSTANTIATE_SUM_PROPERTIES(HandlerTy)                                  \
                                                                               \
  template std::vector<std::size_t> operator_sum<HandlerTy>::degrees(          \
      bool application_order) const;                                           \
                                                                               \
  template std::size_t operator_sum<HandlerTy>::num_terms() const;             \
                                                                               \
  template std::vector<product_operator<HandlerTy>>                            \
  operator_sum<HandlerTy>::get_terms() const;

#if !defined(__clang__)
INSTANTIATE_SUM_PROPERTIES(matrix_operator);
INSTANTIATE_SUM_PROPERTIES(spin_operator);
INSTANTIATE_SUM_PROPERTIES(boson_operator);
INSTANTIATE_SUM_PROPERTIES(fermion_operator);
#endif

// constructors

template <typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(const product_operator<HandlerTy> &prod) {
  this->insert(prod);
}

template <typename HandlerTy>
template <typename... Args,
          std::enable_if_t<std::conjunction<std::is_same<
                               product_operator<HandlerTy>, Args>...>::value,
                           bool>>
operator_sum<HandlerTy>::operator_sum(Args &&...args) {
  this->coefficients.reserve(sizeof...(Args));
  this->term_map.reserve(sizeof...(Args));
  this->terms.reserve(sizeof...(Args));
  aggregate_terms(std::forward<product_operator<HandlerTy> &&>(args)...);
}

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                               std::is_constructible<HandlerTy, T>::value,
                           bool>>
operator_sum<HandlerTy>::operator_sum(const operator_sum<T> &other)
    : coefficients(other.coefficients) {
  this->term_map.reserve(other.terms.size());
  this->terms.reserve(other.terms.size());
  for (const auto &operators : other.terms) {
    product_operator<HandlerTy> term(
        product_operator<T>(1., operators)); // coefficient does not matter
    this->term_map.insert(
        this->term_map.cend(),
        std::make_pair(term.get_term_id(), this->terms.size()));
    this->terms.push_back(std::move(term.operators));
  }
}

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<std::is_same<HandlerTy, matrix_operator>::value &&
                               !std::is_same<T, HandlerTy>::value &&
                               std::is_constructible<HandlerTy, T>::value,
                           bool>>
operator_sum<HandlerTy>::operator_sum(
    const operator_sum<T> &other,
    const matrix_operator::commutation_behavior &behavior)
    : coefficients(other.coefficients) {
  this->term_map.reserve(other.terms.size());
  this->terms.reserve(other.terms.size());
  for (const auto &operators : other.terms) {
    product_operator<HandlerTy> term(product_operator<T>(1., operators),
                                     behavior); // coefficient does not matter
    this->term_map.insert(
        this->term_map.cend(),
        std::make_pair(term.get_term_id(), this->terms.size()));
    this->terms.push_back(std::move(term.operators));
  }
}

template <typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(const operator_sum<HandlerTy> &other,
                                      int size) {
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

template <typename HandlerTy>
operator_sum<HandlerTy>::operator_sum(operator_sum<HandlerTy> &&other, int size)
    : coefficients(std::move(other.coefficients)),
      term_map(std::move(other.term_map)), terms(std::move(other.terms)) {
  if (size > 0) {
    this->coefficients.reserve(size);
    this->term_map.reserve(size);
    this->terms.reserve(size);
  }
}

#define INSTANTIATE_SUM_CONSTRUCTORS(HandlerTy)                                \
                                                                               \
  template operator_sum<HandlerTy>::operator_sum();                            \
                                                                               \
  template operator_sum<HandlerTy>::operator_sum(                              \
      const product_operator<HandlerTy> &item2);                               \
                                                                               \
  template operator_sum<HandlerTy>::operator_sum(                              \
      product_operator<HandlerTy> &&item2);                                    \
                                                                               \
  template operator_sum<HandlerTy>::operator_sum(                              \
      product_operator<HandlerTy> &&item1,                                     \
      product_operator<HandlerTy> &&item2);                                    \
                                                                               \
  template operator_sum<HandlerTy>::operator_sum(                              \
      product_operator<HandlerTy> &&item1,                                     \
      product_operator<HandlerTy> &&item2,                                     \
      product_operator<HandlerTy> &&item3);                                    \
                                                                               \
  template operator_sum<HandlerTy>::operator_sum(                              \
      const operator_sum<HandlerTy> &other, int size);                         \
                                                                               \
  template operator_sum<HandlerTy>::operator_sum(                              \
      operator_sum<HandlerTy> &&other, int size);

// Note:
// These are the private constructors needed by friend classes and functions
// of operator_sum. For clang, (only!) these need to be instantiated explicitly
// to be available to those.
#define INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(HandlerTy)                 \
                                                                               \
  template operator_sum<HandlerTy>::operator_sum();                            \
                                                                               \
  template operator_sum<HandlerTy>::operator_sum(                              \
      product_operator<HandlerTy> &&item1,                                     \
      product_operator<HandlerTy> &&item2);

template operator_sum<matrix_operator>::operator_sum(
    const operator_sum<spin_operator> &other);
template operator_sum<matrix_operator>::operator_sum(
    const operator_sum<boson_operator> &other);
template operator_sum<matrix_operator>::operator_sum(
    const operator_sum<fermion_operator> &other);
template operator_sum<matrix_operator>::operator_sum(
    const operator_sum<spin_operator> &other,
    const matrix_operator::commutation_behavior &behavior);
template operator_sum<matrix_operator>::operator_sum(
    const operator_sum<boson_operator> &other,
    const matrix_operator::commutation_behavior &behavior);
template operator_sum<matrix_operator>::operator_sum(
    const operator_sum<fermion_operator> &other,
    const matrix_operator::commutation_behavior &behavior);

#if !defined(__clang__)
INSTANTIATE_SUM_CONSTRUCTORS(matrix_operator);
INSTANTIATE_SUM_CONSTRUCTORS(spin_operator);
INSTANTIATE_SUM_CONSTRUCTORS(boson_operator);
INSTANTIATE_SUM_CONSTRUCTORS(fermion_operator);
#else
INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(matrix_operator);
INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(spin_operator);
INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(boson_operator);
INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(fermion_operator);
#endif

// assignments

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                               std::is_constructible<HandlerTy, T>::value,
                           bool>>
operator_sum<HandlerTy> &
operator_sum<HandlerTy>::operator=(const product_operator<T> &other) {
  *this = product_operator<HandlerTy>(other);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy> &
operator_sum<HandlerTy>::operator=(const product_operator<HandlerTy> &other) {
  this->coefficients.clear();
  this->term_map.clear();
  this->terms.clear();
  this->coefficients.push_back(other.coefficient);
  this->term_map.insert(this->term_map.cend(),
                        std::make_pair(other.get_term_id(), 0));
  this->terms.push_back(other.operators);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy> &
operator_sum<HandlerTy>::operator=(product_operator<HandlerTy> &&other) {
  this->coefficients.clear();
  this->term_map.clear();
  this->terms.clear();
  this->coefficients.push_back(std::move(other.coefficient));
  this->term_map.insert(this->term_map.cend(),
                        std::make_pair(other.get_term_id(), 0));
  this->terms.push_back(std::move(other.operators));
  return *this;
}

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                               std::is_constructible<HandlerTy, T>::value,
                           bool>>
operator_sum<HandlerTy> &
operator_sum<HandlerTy>::operator=(const operator_sum<T> &other) {
  *this = operator_sum<HandlerTy>(other);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy> &
operator_sum<HandlerTy>::operator=(const operator_sum<HandlerTy> &other) {
  if (this != &other) {
    this->coefficients = other.coefficients;
    this->term_map = other.term_map;
    this->terms = other.terms;
  }
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy> &
operator_sum<HandlerTy>::operator=(operator_sum<HandlerTy> &&other) {
  if (this != &other) {
    this->coefficients = std::move(other.coefficients);
    this->term_map = std::move(other.term_map);
    this->terms = std::move(other.terms);
  }
  return *this;
}

#define INSTANTIATE_SUM_ASSIGNMENTS(HandlerTy)                                 \
                                                                               \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator=(        \
      product_operator<HandlerTy> &&other);                                    \
                                                                               \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator=(        \
      const product_operator<HandlerTy> &other);                               \
                                                                               \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator=(        \
      const operator_sum<HandlerTy> &other);                                   \
                                                                               \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator=(        \
      operator_sum<HandlerTy> &&other);

template operator_sum<matrix_operator> &
operator_sum<matrix_operator>::operator=(
    const product_operator<spin_operator> &other);
template operator_sum<matrix_operator> &
operator_sum<matrix_operator>::operator=(
    const product_operator<boson_operator> &other);
template operator_sum<matrix_operator> &
operator_sum<matrix_operator>::operator=(
    const product_operator<fermion_operator> &other);
template operator_sum<matrix_operator> &
operator_sum<matrix_operator>::operator=(
    const operator_sum<spin_operator> &other);
template operator_sum<matrix_operator> &
operator_sum<matrix_operator>::operator=(
    const operator_sum<boson_operator> &other);
template operator_sum<matrix_operator> &
operator_sum<matrix_operator>::operator=(
    const operator_sum<fermion_operator> &other);

#if !defined(__clang__)
INSTANTIATE_SUM_ASSIGNMENTS(matrix_operator);
INSTANTIATE_SUM_ASSIGNMENTS(spin_operator);
INSTANTIATE_SUM_ASSIGNMENTS(boson_operator);
INSTANTIATE_SUM_ASSIGNMENTS(fermion_operator);
#endif

// evaluations

template <typename HandlerTy>
std::string operator_sum<HandlerTy>::to_string() const {
  auto prods = this->get_terms();
  auto it = prods.cbegin();
  std::string str = it->to_string();
  while (++it != prods.cend())
    str += " + " + it->to_string();
  return std::move(str);
}

template <typename HandlerTy>
complex_matrix operator_sum<HandlerTy>::to_matrix(
    std::unordered_map<int, int> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool application_order) const {
  auto evaluated =
      this->evaluate(operator_arithmetics<operator_handler::matrix_evaluation>(
          dimensions, parameters));
  if (!application_order || operator_handler::canonical_order(1, 0) ==
                                operator_handler::user_facing_order(1, 0))
    return std::move(evaluated.matrix);

  auto degrees = evaluated.degrees;
  std::sort(degrees.begin(), degrees.end(),
            operator_handler::user_facing_order);
  auto permutation = cudaq::detail::compute_permutation(evaluated.degrees,
                                                        degrees, dimensions);
  cudaq::detail::permute_matrix(evaluated.matrix, permutation);
  return std::move(evaluated.matrix);
}

template <>
complex_matrix operator_sum<spin_operator>::to_matrix(
    std::unordered_map<int, int> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool application_order) const {
  auto evaluated = this->evaluate(
      operator_arithmetics<operator_handler::canonical_evaluation>(dimensions,
                                                                   parameters));
  if (evaluated.terms.size() == 0)
    return cudaq::complex_matrix(0, 0);

  bool invert_order =
      application_order && operator_handler::canonical_order(1, 0) !=
                               operator_handler::user_facing_order(1, 0);
  auto matrix = spin_operator::to_matrix(
      evaluated.terms[0].second, evaluated.terms[0].first, invert_order);
  for (auto i = 1; i < terms.size(); ++i)
    matrix += spin_operator::to_matrix(evaluated.terms[i].second,
                                       evaluated.terms[i].first, invert_order);
  return std::move(matrix);
}

#define INSTANTIATE_SUM_EVALUATIONS(HandlerTy)                                 \
                                                                               \
  template std::string operator_sum<HandlerTy>::to_string() const;             \
                                                                               \
  template complex_matrix operator_sum<HandlerTy>::to_matrix(                        \
      std::unordered_map<int, int> dimensions,                                 \
      const std::unordered_map<std::string, std::complex<double>> &params,     \
      bool application_order) const;

#if !defined(__clang__)
INSTANTIATE_SUM_EVALUATIONS(matrix_operator);
INSTANTIATE_SUM_EVALUATIONS(spin_operator);
INSTANTIATE_SUM_EVALUATIONS(boson_operator);
INSTANTIATE_SUM_EVALUATIONS(fermion_operator);
#endif

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

#define INSTANTIATE_SUM_UNARY_OPS(HandlerTy)                                   \
                                                                               \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-()        \
      const &;                                                                 \
                                                                               \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-() &&;    \
                                                                               \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+()        \
      const &;                                                                 \
                                                                               \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+() &&;

#if !defined(__clang__)
INSTANTIATE_SUM_UNARY_OPS(matrix_operator);
INSTANTIATE_SUM_UNARY_OPS(spin_operator);
INSTANTIATE_SUM_UNARY_OPS(boson_operator);
INSTANTIATE_SUM_UNARY_OPS(fermion_operator);
#endif

// right-hand arithmetics

#define SUM_MULTIPLICATION_SCALAR(op)                                          \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      const scalar_operator &other) const & {                                  \
    operator_sum<HandlerTy> sum;                                               \
    sum.coefficients.reserve(this->coefficients.size());                       \
    sum.term_map = this->term_map;                                             \
    sum.terms = this->terms;                                                   \
    for (const auto &coeff : this->coefficients)                               \
      sum.coefficients.push_back(coeff op other);                              \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      const scalar_operator &other) && {                                       \
    for (auto &coeff : this->coefficients)                                     \
      coeff op## = other;                                                      \
    return std::move(*this);                                                   \
  }

SUM_MULTIPLICATION_SCALAR(*);
SUM_MULTIPLICATION_SCALAR(/);

#define SUM_ADDITION_SCALAR(op)                                                \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      const scalar_operator &other) const & {                                  \
    operator_sum<HandlerTy> sum(*this, this->terms.size() + 1);                \
    sum.insert(product_operator<HandlerTy>(op other));                         \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      scalar_operator &&other) const & {                                       \
    operator_sum<HandlerTy> sum(*this, this->terms.size() + 1);                \
    sum.insert(product_operator<HandlerTy>(op std::move(other)));              \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      const scalar_operator &other) && {                                       \
    this->insert(product_operator<HandlerTy>(op other));                       \
    return std::move(*this);                                                   \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      scalar_operator &&other) && {                                            \
    this->insert(product_operator<HandlerTy>(op std::move(other)));            \
    return std::move(*this);                                                   \
  }

SUM_ADDITION_SCALAR(+);
SUM_ADDITION_SCALAR(-);

#define INSTANTIATE_SUM_RHSIMPLE_OPS(HandlerTy)                                \
                                                                               \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(         \
      const scalar_operator &other) const &;                                   \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(         \
      const scalar_operator &other) &&;                                        \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator/(         \
      const scalar_operator &other) const &;                                   \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator/(         \
      const scalar_operator &other) &&;                                        \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      scalar_operator &&other) const &;                                        \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      scalar_operator &&other) &&;                                             \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      const scalar_operator &other) const &;                                   \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      const scalar_operator &other) &&;                                        \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      scalar_operator &&other) const &;                                        \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      scalar_operator &&other) &&;                                             \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      const scalar_operator &other) const &;                                   \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      const scalar_operator &other) &&;

#if !defined(__clang__)
INSTANTIATE_SUM_RHSIMPLE_OPS(matrix_operator);
INSTANTIATE_SUM_RHSIMPLE_OPS(spin_operator);
INSTANTIATE_SUM_RHSIMPLE_OPS(boson_operator);
INSTANTIATE_SUM_RHSIMPLE_OPS(fermion_operator);
#endif

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(
    const product_operator<HandlerTy> &other) const {
  operator_sum<HandlerTy> sum; // the entire sum needs to be rebuilt
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map.reserve(this->terms.size());
  sum.terms.reserve(this->terms.size());
  for (auto i = 0; i < this->terms.size(); ++i) {
    auto max_size = this->terms[i].size() + other.operators.size();
    product_operator<HandlerTy> prod(this->coefficients[i] * other.coefficient,
                                     this->terms[i], max_size);
    for (HandlerTy op : other.operators)
      prod.insert(std::move(op));
    sum.insert(std::move(prod));
  }
  return std::move(sum);
}

#define SUM_ADDITION_PRODUCT(op)                                               \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      const product_operator<HandlerTy> &other) const & {                      \
    operator_sum<HandlerTy> sum(*this, this->terms.size() + 1);                \
    sum.insert(op other);                                                      \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      const product_operator<HandlerTy> &other) && {                           \
    this->insert(op other);                                                    \
    return std::move(*this);                                                   \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      product_operator<HandlerTy> &&other) const & {                           \
    operator_sum<HandlerTy> sum(*this, this->terms.size() + 1);                \
    sum.insert(op std::move(other));                                           \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      product_operator<HandlerTy> &&other) && {                                \
    this->insert(op std::move(other));                                         \
    return std::move(*this);                                                   \
  }

SUM_ADDITION_PRODUCT(+)
SUM_ADDITION_PRODUCT(-)

template <typename HandlerTy>
operator_sum<HandlerTy>
operator_sum<HandlerTy>::operator*(const operator_sum<HandlerTy> &other) const {
  operator_sum<HandlerTy> sum; // the entire sum needs to be rebuilt
  auto max_size = this->terms.size() * other.terms.size();
  sum.coefficients.reserve(max_size);
  sum.term_map.reserve(max_size);
  sum.terms.reserve(max_size);
  for (auto i = 0; i < this->terms.size(); ++i) {
    for (auto j = 0; j < other.terms.size(); ++j) {
      auto max_size = this->terms[i].size() + other.terms[j].size();
      product_operator<HandlerTy> prod(this->coefficients[i] *
                                           other.coefficients[j],
                                       this->terms[i], max_size);
      for (HandlerTy op : other.terms[j])
        prod.insert(std::move(op));
      sum.insert(std::move(prod));
    }
  }
  return std::move(sum);
}

#define SUM_ADDITION_SUM(op)                                                   \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      const operator_sum<HandlerTy> &other) const & {                          \
    operator_sum<HandlerTy> sum(*this,                                         \
                                this->terms.size() + other.terms.size());      \
    for (auto i = 0; i < other.terms.size(); ++i) {                            \
      product_operator<HandlerTy> prod(op other.coefficients[i],               \
                                       other.terms[i]);                        \
      sum.insert(std::move(prod));                                             \
    }                                                                          \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      const operator_sum<HandlerTy> &other) && {                               \
    auto max_size = this->terms.size() + other.terms.size();                   \
    this->coefficients.reserve(max_size);                                      \
    this->term_map.reserve(max_size);                                          \
    this->terms.reserve(max_size);                                             \
    for (auto i = 0; i < other.terms.size(); ++i)                              \
      this->insert(product_operator<HandlerTy>(op other.coefficients[i],       \
                                               other.terms[i]));               \
    return std::move(*this);                                                   \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      operator_sum<HandlerTy> &&other) const & {                               \
    operator_sum<HandlerTy> sum(*this,                                         \
                                this->terms.size() + other.terms.size());      \
    for (auto i = 0; i < other.terms.size(); ++i) {                            \
      product_operator<HandlerTy> prod(op std::move(other.coefficients[i]),    \
                                       std::move(other.terms[i]));             \
      sum.insert(std::move(prod));                                             \
    }                                                                          \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator_sum<HandlerTy>::operator op(                \
      operator_sum<HandlerTy> &&other) && {                                    \
    auto max_size = this->terms.size() + other.terms.size();                   \
    this->coefficients.reserve(max_size);                                      \
    this->term_map.reserve(max_size);                                          \
    this->terms.reserve(max_size);                                             \
    for (auto i = 0; i < other.terms.size(); ++i)                              \
      this->insert(product_operator<HandlerTy>(                                \
          op std::move(other.coefficients[i]), std::move(other.terms[i])));    \
    return std::move(*this);                                                   \
  }

SUM_ADDITION_SUM(+);
SUM_ADDITION_SUM(-);

#define INSTANTIATE_SUM_RHCOMPOSITE_OPS(HandlerTy)                             \
                                                                               \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(         \
      const product_operator<HandlerTy> &other) const;                         \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      const product_operator<HandlerTy> &other) const &;                       \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      const product_operator<HandlerTy> &other) &&;                            \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      product_operator<HandlerTy> &&other) const &;                            \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      product_operator<HandlerTy> &&other) &&;                                 \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      const product_operator<HandlerTy> &other) const &;                       \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      const product_operator<HandlerTy> &other) &&;                            \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      product_operator<HandlerTy> &&other) const &;                            \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      product_operator<HandlerTy> &&other) &&;                                 \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(         \
      const operator_sum<HandlerTy> &other) const;                             \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      const operator_sum<HandlerTy> &other) const &;                           \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      const operator_sum<HandlerTy> &other) &&;                                \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      operator_sum<HandlerTy> &&other) const &;                                \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(         \
      operator_sum<HandlerTy> &&other) &&;                                     \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      const operator_sum<HandlerTy> &other) const &;                           \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      const operator_sum<HandlerTy> &other) &&;                                \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      operator_sum<HandlerTy> &&other) const &;                                \
  template operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(         \
      operator_sum<HandlerTy> &&other) &&;

#if !defined(__clang__)
INSTANTIATE_SUM_RHCOMPOSITE_OPS(matrix_operator);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(spin_operator);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(boson_operator);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(fermion_operator);
#endif

#define SUM_MULTIPLICATION_SCALAR_ASSIGNMENT(op)                               \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator op(               \
      const scalar_operator &other) {                                          \
    for (auto &coeff : this->coefficients)                                     \
      coeff op other;                                                          \
    return *this;                                                              \
  }

SUM_MULTIPLICATION_SCALAR_ASSIGNMENT(*=);
SUM_MULTIPLICATION_SCALAR_ASSIGNMENT(/=);

#define SUM_ADDITION_SCALAR_ASSIGNMENT(op)                                     \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator op##=(            \
      const scalar_operator &other) {                                          \
    this->insert(product_operator<HandlerTy>(op other));                       \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator op##=(            \
      scalar_operator &&other) {                                               \
    this->insert(product_operator<HandlerTy>(op std::move(other)));            \
    return *this;                                                              \
  }

SUM_ADDITION_SCALAR_ASSIGNMENT(+);
SUM_ADDITION_SCALAR_ASSIGNMENT(-);

template <typename HandlerTy>
operator_sum<HandlerTy> &
operator_sum<HandlerTy>::operator*=(const product_operator<HandlerTy> &other) {
  operator_sum<HandlerTy> sum;
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map.reserve(this->terms.size());
  sum.terms.reserve(this->terms.size());
  for (auto i = 0; i < this->terms.size(); ++i) {
    auto max_size = this->terms[i].size() + other.operators.size();
    product_operator<HandlerTy> prod(this->coefficients[i] * other.coefficient,
                                     this->terms[i], max_size);
    for (HandlerTy op : other.operators)
      prod.insert(std::move(op));
    sum.insert(std::move(prod));
  }
  *this = std::move(sum);
  return *this;
}

#define SUM_ADDITION_PRODUCT_ASSIGNMENT(op)                                    \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator op##=(            \
      const product_operator<HandlerTy> &other) {                              \
    this->insert(op other);                                                    \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator op##=(            \
      product_operator<HandlerTy> &&other) {                                   \
    this->insert(op std::move(other));                                         \
    return *this;                                                              \
  }

SUM_ADDITION_PRODUCT_ASSIGNMENT(+)
SUM_ADDITION_PRODUCT_ASSIGNMENT(-)

template <typename HandlerTy>
operator_sum<HandlerTy> &
operator_sum<HandlerTy>::operator*=(const operator_sum<HandlerTy> &other) {
  operator_sum<HandlerTy> sum; // the entire sum needs to be rebuilt
  auto max_size = this->terms.size() * other.terms.size();
  sum.coefficients.reserve(max_size);
  sum.term_map.reserve(max_size);
  sum.terms.reserve(max_size);
  for (auto i = 0; i < this->terms.size(); ++i) {
    for (auto j = 0; j < other.terms.size(); ++j) {
      auto max_size = this->terms[i].size() + other.terms[j].size();
      product_operator<HandlerTy> prod(this->coefficients[i] *
                                           other.coefficients[j],
                                       this->terms[i], max_size);
      for (HandlerTy op : other.terms[j])
        prod.insert(std::move(op));
      sum.insert(std::move(prod));
    }
  }
  *this = std::move(sum);
  return *this;
}

#define SUM_ADDITION_SUM_ASSIGNMENT(op)                                        \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator op##=(            \
      const operator_sum<HandlerTy> &other) {                                  \
    auto max_size = this->terms.size() + other.terms.size();                   \
    this->coefficients.reserve(max_size);                                      \
    this->term_map.reserve(max_size);                                          \
    this->terms.reserve(max_size);                                             \
    for (auto i = 0; i < other.terms.size(); ++i)                              \
      this->insert(product_operator<HandlerTy>(op other.coefficients[i],       \
                                               other.terms[i]));               \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator op##=(            \
      operator_sum<HandlerTy> &&other) {                                       \
    auto max_size = this->terms.size() + other.terms.size();                   \
    this->coefficients.reserve(max_size);                                      \
    this->term_map.reserve(max_size);                                          \
    this->terms.reserve(max_size);                                             \
    for (auto i = 0; i < other.terms.size(); ++i)                              \
      this->insert(product_operator<HandlerTy>(                                \
          op std::move(other.coefficients[i]), std::move(other.terms[i])));    \
    return *this;                                                              \
  }

SUM_ADDITION_SUM_ASSIGNMENT(+);
SUM_ADDITION_SUM_ASSIGNMENT(-);

#define INSTANTIATE_SUM_OPASSIGNMENTS(HandlerTy)                               \
                                                                               \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator*=(       \
      const scalar_operator &other);                                           \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator/=(       \
      const scalar_operator &other);                                           \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator+=(       \
      scalar_operator &&other);                                                \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator+=(       \
      const scalar_operator &other);                                           \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator-=(       \
      scalar_operator &&other);                                                \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator-=(       \
      const scalar_operator &other);                                           \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator*=(       \
      const product_operator<HandlerTy> &other);                               \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator+=(       \
      const product_operator<HandlerTy> &other);                               \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator+=(       \
      product_operator<HandlerTy> &&other);                                    \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator-=(       \
      const product_operator<HandlerTy> &other);                               \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator-=(       \
      product_operator<HandlerTy> &&other);                                    \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator*=(       \
      const operator_sum<HandlerTy> &other);                                   \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator+=(       \
      const operator_sum<HandlerTy> &other);                                   \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator+=(       \
      operator_sum<HandlerTy> &&other);                                        \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator-=(       \
      const operator_sum<HandlerTy> &other);                                   \
  template operator_sum<HandlerTy> &operator_sum<HandlerTy>::operator-=(       \
      operator_sum<HandlerTy> &&other);

#if !defined(__clang__)
INSTANTIATE_SUM_OPASSIGNMENTS(matrix_operator);
INSTANTIATE_SUM_OPASSIGNMENTS(spin_operator);
INSTANTIATE_SUM_OPASSIGNMENTS(boson_operator);
INSTANTIATE_SUM_OPASSIGNMENTS(fermion_operator);
#endif

// left-hand arithmetics

template <typename HandlerTy>
operator_sum<HandlerTy> operator*(const scalar_operator &other,
                                  const operator_sum<HandlerTy> &self) {
  operator_sum<HandlerTy> sum;
  sum.coefficients.reserve(self.coefficients.size());
  sum.terms = self.terms;
  sum.term_map = self.term_map;
  for (const auto &coeff : self.coefficients)
    sum.coefficients.push_back(coeff * other);
  return std::move(sum);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator*(const scalar_operator &other,
                                  operator_sum<HandlerTy> &&self) {
  for (auto &&coeff : self.coefficients)
    coeff *= other;
  return std::move(self);
}

#define SUM_ADDITION_SCALAR_REVERSE(op)                                        \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator op(const scalar_operator &other,            \
                                      const operator_sum<HandlerTy> &self) {   \
    operator_sum<HandlerTy> sum(op self);                                      \
    sum.insert(product_operator<HandlerTy>(other));                            \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator op(scalar_operator &&other,                 \
                                      const operator_sum<HandlerTy> &self) {   \
    operator_sum<HandlerTy> sum(op self);                                      \
    sum.insert(product_operator<HandlerTy>(std::move(other)));                 \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator op(const scalar_operator &other,            \
                                      operator_sum<HandlerTy> &&self) {        \
    for (auto &&coeff : self.coefficients)                                     \
      coeff = std::move(op coeff);                                             \
    self.insert(product_operator<HandlerTy>(other));                           \
    return std::move(self);                                                    \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  operator_sum<HandlerTy> operator op(scalar_operator &&other,                 \
                                      operator_sum<HandlerTy> &&self) {        \
    for (auto &&coeff : self.coefficients)                                     \
      coeff = std::move(op coeff);                                             \
    self.insert(product_operator<HandlerTy>(std::move(other)));                \
    return std::move(self);                                                    \
  }

SUM_ADDITION_SCALAR_REVERSE(+);
SUM_ADDITION_SCALAR_REVERSE(-);

#define INSTANTIATE_SUM_LHCOMPOSITE_OPS(HandlerTy)                             \
                                                                               \
  template operator_sum<HandlerTy> operator*(                                  \
      const scalar_operator &other, const operator_sum<HandlerTy> &self);      \
  template operator_sum<HandlerTy> operator*(const scalar_operator &other,     \
                                             operator_sum<HandlerTy> &&self);  \
  template operator_sum<HandlerTy> operator+(                                  \
      scalar_operator &&other, const operator_sum<HandlerTy> &self);           \
  template operator_sum<HandlerTy> operator+(scalar_operator &&other,          \
                                             operator_sum<HandlerTy> &&self);  \
  template operator_sum<HandlerTy> operator+(                                  \
      const scalar_operator &other, const operator_sum<HandlerTy> &self);      \
  template operator_sum<HandlerTy> operator+(const scalar_operator &other,     \
                                             operator_sum<HandlerTy> &&self);  \
  template operator_sum<HandlerTy> operator-(                                  \
      scalar_operator &&other, const operator_sum<HandlerTy> &self);           \
  template operator_sum<HandlerTy> operator-(scalar_operator &&other,          \
                                             operator_sum<HandlerTy> &&self);  \
  template operator_sum<HandlerTy> operator-(                                  \
      const scalar_operator &other, const operator_sum<HandlerTy> &self);      \
  template operator_sum<HandlerTy> operator-(const scalar_operator &other,     \
                                             operator_sum<HandlerTy> &&self);

INSTANTIATE_SUM_LHCOMPOSITE_OPS(matrix_operator);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(spin_operator);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(boson_operator);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(fermion_operator);

// arithmetics that require conversions

#define SUM_CONVERSIONS_OPS(op)                                                \
                                                                               \
  template <typename LHtype, typename RHtype,                                  \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype)>                        \
  operator_sum<matrix_operator> operator op(                                   \
      const operator_sum<LHtype> &other,                                       \
      const product_operator<RHtype> &self) {                                  \
    return operator_sum<matrix_operator>(other) op self;                       \
  }                                                                            \
                                                                               \
  template <typename LHtype, typename RHtype,                                  \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype)>                        \
  operator_sum<matrix_operator> operator op(                                   \
      const product_operator<LHtype> &other,                                   \
      const operator_sum<RHtype> &self) {                                      \
    return product_operator<matrix_operator>(other) op self;                   \
  }                                                                            \
                                                                               \
  template <typename LHtype, typename RHtype,                                  \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype)>                        \
  operator_sum<matrix_operator> operator op(                                   \
      const operator_sum<LHtype> &other, const operator_sum<RHtype> &self) {   \
    return operator_sum<matrix_operator>(other) op self;                       \
  }

SUM_CONVERSIONS_OPS(*);
SUM_CONVERSIONS_OPS(+);
SUM_CONVERSIONS_OPS(-);

#define INSTANTIATE_SUM_CONVERSION_OPS(op)                                     \
                                                                               \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<spin_operator> &other,                                \
      const product_operator<matrix_operator> &self);                          \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<boson_operator> &other,                               \
      const product_operator<matrix_operator> &self);                          \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<fermion_operator> &other,                             \
      const product_operator<matrix_operator> &self);                          \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<spin_operator> &other,                                \
      const product_operator<boson_operator> &self);                           \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<boson_operator> &other,                               \
      const product_operator<spin_operator> &self);                            \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<spin_operator> &other,                                \
      const product_operator<fermion_operator> &self);                         \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<fermion_operator> &other,                             \
      const product_operator<spin_operator> &self);                            \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<boson_operator> &other,                               \
      const product_operator<fermion_operator> &self);                         \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<fermion_operator> &other,                             \
      const product_operator<boson_operator> &self);                           \
                                                                               \
  template operator_sum<matrix_operator> operator op(                          \
      const product_operator<spin_operator> &other,                            \
      const operator_sum<matrix_operator> &self);                              \
  template operator_sum<matrix_operator> operator op(                          \
      const product_operator<boson_operator> &other,                           \
      const operator_sum<matrix_operator> &self);                              \
  template operator_sum<matrix_operator> operator op(                          \
      const product_operator<fermion_operator> &other,                         \
      const operator_sum<matrix_operator> &self);                              \
  template operator_sum<matrix_operator> operator op(                          \
      const product_operator<spin_operator> &other,                            \
      const operator_sum<boson_operator> &self);                               \
  template operator_sum<matrix_operator> operator op(                          \
      const product_operator<boson_operator> &other,                           \
      const operator_sum<spin_operator> &self);                                \
  template operator_sum<matrix_operator> operator op(                          \
      const product_operator<spin_operator> &other,                            \
      const operator_sum<fermion_operator> &self);                             \
  template operator_sum<matrix_operator> operator op(                          \
      const product_operator<fermion_operator> &other,                         \
      const operator_sum<spin_operator> &self);                                \
  template operator_sum<matrix_operator> operator op(                          \
      const product_operator<boson_operator> &other,                           \
      const operator_sum<fermion_operator> &self);                             \
  template operator_sum<matrix_operator> operator op(                          \
      const product_operator<fermion_operator> &other,                         \
      const operator_sum<boson_operator> &self);                               \
                                                                               \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<spin_operator> &other,                                \
      const operator_sum<matrix_operator> &self);                              \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<boson_operator> &other,                               \
      const operator_sum<matrix_operator> &self);                              \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<fermion_operator> &other,                             \
      const operator_sum<matrix_operator> &self);                              \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<spin_operator> &other,                                \
      const operator_sum<boson_operator> &self);                               \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<boson_operator> &other,                               \
      const operator_sum<spin_operator> &self);                                \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<spin_operator> &other,                                \
      const operator_sum<fermion_operator> &self);                             \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<fermion_operator> &other,                             \
      const operator_sum<spin_operator> &self);                                \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<boson_operator> &other,                               \
      const operator_sum<fermion_operator> &self);                             \
  template operator_sum<matrix_operator> operator op(                          \
      const operator_sum<fermion_operator> &other,                             \
      const operator_sum<boson_operator> &self);

INSTANTIATE_SUM_CONVERSION_OPS(*);
INSTANTIATE_SUM_CONVERSION_OPS(+);
INSTANTIATE_SUM_CONVERSION_OPS(-);

// common operators

// FIXME: remove
template <typename HandlerTy>
operator_sum<HandlerTy> operator_handler::empty() {
  return operator_sum<HandlerTy>();
}

template operator_sum<matrix_operator> operator_handler::empty();
template operator_sum<spin_operator> operator_handler::empty();
template operator_sum<boson_operator> operator_handler::empty();
template operator_sum<fermion_operator> operator_handler::empty();

// handler specific operators

#define HANDLER_SPECIFIC_TEMPLATE_DEFINITION(ConcreteTy)                                  \
  template <typename HandlerTy>                                                           \
  template <typename T, std::enable_if_t<                                                 \
                                      std::is_same<HandlerTy, ConcreteTy>::value &&       \
                                      std::is_same<HandlerTy, T>::value, bool>>

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_operator)
operator_sum<HandlerTy> operator_sum<HandlerTy>::empty() {
  return operator_sum<HandlerTy>();
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_operator)
product_operator<HandlerTy> operator_sum<HandlerTy>::i(int target) {
  return spin_operator::i(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_operator)
product_operator<HandlerTy> operator_sum<HandlerTy>::x(int target) {
  return spin_operator::x(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_operator)
product_operator<HandlerTy> operator_sum<HandlerTy>::y(int target) {
  return spin_operator::y(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_operator)
product_operator<HandlerTy> operator_sum<HandlerTy>::z(int target) {
  return spin_operator::z(target);
}

#if !defined(__clang__)
template operator_sum<spin_operator> operator_sum<spin_operator>::empty();
template product_operator<spin_operator> operator_sum<spin_operator>::i(int target);
template product_operator<spin_operator> operator_sum<spin_operator>::x(int target);
template product_operator<spin_operator> operator_sum<spin_operator>::y(int target);
template product_operator<spin_operator> operator_sum<spin_operator>::z(int target);
#endif

// general utility functions

template <typename HandlerTy>
std::vector<operator_sum<HandlerTy>> operator_sum<HandlerTy>::distribute_terms(std::size_t numChunks) const {
  // Calculate how many terms we can equally divide amongst the chunks
  auto nTermsPerChunk = num_terms() / numChunks;
  auto leftover = num_terms() % numChunks;

  // Slice the given spin_op into subsets for each chunk
  std::vector<operator_sum<HandlerTy>> chunks;
  for (auto it = this->term_map.cbegin(); it != this->term_map.cend();) { // order does not matter here
    operator_sum<HandlerTy> chunk;
    // Evenly distribute any leftovers across the early chunks
    for (auto count = nTermsPerChunk + (chunks.size() < leftover ? 1 : 0); count > 0; --count, ++it)
      chunk += product_operator<HandlerTy>(this->coefficients[it->second], this->terms[it->second]);
    chunks.push_back(chunk);
  }
  return std::move(chunks);
}

#define INSTANTIATE_SUM_UTILITY_FUNCTIONS(HandlerTy)                                      \
  template std::vector<operator_sum<HandlerTy>>                                           \
  operator_sum<HandlerTy>::distribute_terms(std::size_t numChunks) const;

#if !defined(__clang__)
INSTANTIATE_SUM_UTILITY_FUNCTIONS(matrix_operator);
INSTANTIATE_SUM_UTILITY_FUNCTIONS(spin_operator);
INSTANTIATE_SUM_UTILITY_FUNCTIONS(boson_operator);
INSTANTIATE_SUM_UTILITY_FUNCTIONS(fermion_operator);
#endif

// functions for backwards compatibility

#define SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION                                        \
  template <typename HandlerTy>                                                           \
  template <typename T, std::enable_if_t<                                                 \
                                      std::is_same<HandlerTy, spin_operator>::value &&    \
                                      std::is_same<HandlerTy, T>::value, bool>>

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
std::vector<std::vector<bool>> operator_sum<HandlerTy>::_get_binary_symplectic_form() const {
  std::unordered_map<int, int> dims;
  auto degrees = this->degrees(false); // degrees in canonical order to match the evaluation
  auto evaluated =
    this->evaluate(operator_arithmetics<operator_handler::canonical_evaluation>(
        dims, {})); // fails if we have parameters
  
  std::size_t term_size = 0;
  if (degrees.size() != 0) 
    term_size = operator_handler::canonical_order(0, 1) ? degrees.back() + 1 : degrees[0] + 1;
  std::vector<std::vector<bool>> bsf_terms;
  bsf_terms.reserve(evaluated.terms.size());

  // For compatiblity with existing code, the binary symplectic representation
  // needs to be from smallest to largest degree, and it necessarily must include 
  // all consecutive degrees starting from 0 (even if the operator doesn't act on them). 
  for (auto &term : evaluated.terms) {
    auto pauli_str = std::move(term.second);
    std::vector<bool> bsf(term_size << 1, 0);
    for (std::size_t i = 0; i < degrees.size(); ++i) {
      auto op = pauli_str[i];
      if (op == 'X')
        bsf[degrees[i]] = 1;
      else if (op == 'Z')
        bsf[degrees[i] + term_size] = 1;
      else if (op == 'Y') {
        bsf[degrees[i]] = 1;
        bsf[degrees[i] + term_size] = 1;
      }
    }
    bsf_terms.push_back(std::move(bsf));
  }

  return std::move(bsf_terms); // always little endian order by definition of the bsf
}

#if !defined(__clang__)
template std::vector<std::vector<bool>> operator_sum<spin_operator>::_get_binary_symplectic_form() const;
#endif


SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
product_operator<HandlerTy> operator_sum<HandlerTy>::from_word(const std::string &word) {
  auto prod = operator_handler::identity<HandlerTy>();
  for (std::size_t i = 0; i < word.length(); i++) {
    auto letter = word[i];
    if (letter == 'Y')
      prod *= spin_operator::y(i);
    else if (letter == 'X')
      prod *= spin_operator::x(i);
    else if (letter == 'Z')
      prod *= spin_operator::z(i);
    else if (letter == 'I')
      prod *= spin_operator::i(i);
    else
      throw std::runtime_error(
        "Invalid Pauli for spin_op::from_word, must be X, Y, Z, or I.");
  }
  return std::move(prod);
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
operator_sum<HandlerTy> operator_sum<HandlerTy>::random(std::size_t nQubits, std::size_t nTerms, unsigned int seed) {
  auto get_spin_op = [](int target, int kind) {
    if (kind == 1) return spin_operator::z(target);
    if (kind == 2) return spin_operator::x(target);
    if (kind == 3) return spin_operator::y(target);
    return spin_operator::i(target);
  };
  std::mt19937 gen(seed);
  auto sum = spin_op::empty();
  for (std::size_t i = 0; i < nTerms; i++) {
    std::vector<bool> termData(2 * nQubits);
    std::fill_n(termData.begin(), nQubits, true);
    std::shuffle(termData.begin(), termData.end(), gen);
    // duplicates are fine - they will just increase the coefficient
    auto prod = spin_operator::identity();
    for (int qubit_idx = 0; qubit_idx < nQubits; ++qubit_idx) {
      auto kind = (termData[qubit_idx << 1] << 1) | termData[(qubit_idx << 1) + 1];
      prod *= get_spin_op(qubit_idx, kind);
    }
    sum += std::move(prod);
  }
  return std::move(sum);
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
operator_sum<HandlerTy>::operator_sum(const std::vector<double> &input_vec, std::size_t nQubits) {
  auto n_terms = (int)input_vec.back();
  if (nQubits != (((input_vec.size() - 1) - 2 * n_terms) / n_terms))
    throw std::runtime_error("Invalid data representation for construction "
                              "spin_op. Number of data elements is incorrect.");

  for (std::size_t i = 0; i < input_vec.size() - 1; i += nQubits + 2) {
    auto el_real = input_vec[i + nQubits];
    auto el_imag = input_vec[i + nQubits + 1];
    auto prod = product_operator<spin_operator>(std::complex<double>{el_real, el_imag});
    for (std::size_t j = 0; j < nQubits; j++) {
      double intPart;
      if (std::modf(input_vec[j + i], &intPart) != 0.0)
        throw std::runtime_error(
            "Invalid pauli data element, must be integer value.");

      int val = (int)input_vec[j + i];
      // FIXME: align op codes with old impl
      if (val == 1) // X
        prod *= spin_operator::x(j);
      else if (val == 2) // Z
        prod *= spin_operator::z(j);
      else if (val == 3) // Y
        prod *= spin_operator::y(j);
    }
    *this += std::move(prod);
  }
}

#if !defined(__clang__)
template product_operator<spin_operator> operator_sum<spin_operator>::from_word(const std::string &word);
template operator_sum<spin_operator> operator_sum<spin_operator>::random(std::size_t nQubits, std::size_t nTerms, unsigned int seed);
template operator_sum<spin_operator>::operator_sum(const std::vector<double> &input_vec, std::size_t nQubits);
#endif

#if defined(CUDAQ_INSTANTIATE_TEMPLATES)
template class operator_sum<matrix_operator>;
template class operator_sum<spin_operator>;
template class operator_sum<boson_operator>;
template class operator_sum<fermion_operator>;
#endif

} // namespace cudaq