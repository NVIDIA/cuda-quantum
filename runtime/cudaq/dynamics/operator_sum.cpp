/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <algorithm>
#include <iostream>
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
void sum_op<HandlerTy>::insert(const product_op<HandlerTy> &other) {
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
void sum_op<HandlerTy>::insert(product_op<HandlerTy> &&other) {
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
void sum_op<HandlerTy>::aggregate_terms() {}

template <typename HandlerTy>
template <typename... Args>
void sum_op<HandlerTy>::aggregate_terms(product_op<HandlerTy> &&head,
                                        Args &&...args) {
  this->insert(std::forward<product_op<HandlerTy>>(head));
  aggregate_terms(std::forward<Args>(args)...);
}

template <typename HandlerTy>
template <typename EvalTy>
EvalTy
sum_op<HandlerTy>::evaluate(operator_arithmetics<EvalTy> arithmetics) const {

  if (terms.size() == 0)
    return EvalTy();

  // Adding a tensor product with the identity for degrees that an operator
  // doesn't act on. Needed e.g. to make sure all matrices are of the same size
  // before summing them up.
  auto degrees = this->degrees(false); // keep in canonical order
  auto paddedTerm = [&arithmetics, &degrees = std::as_const(degrees)](
                        product_op<HandlerTy> &&term) {
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
    product_op<HandlerTy> prod(1, std::move(prod_ops));
    prod *= term; // ensures canonical ordering (if possible)
    return prod;
  };

  // NOTE: It is important that we evaluate the terms in a specific order,
  // otherwise the evaluation is not consistent with other methods.
  // The specific order does not matter, as long as all methods use the same
  // term order.
  auto it = this->begin();
  auto end = this->end();
  if (arithmetics.pad_sum_terms) {
    product_op<HandlerTy> padded_term = paddedTerm(std::move(*it));
    EvalTy sum = padded_term.template evaluate<EvalTy>(arithmetics);
    while (++it != end) {
      padded_term = paddedTerm(std::move(*it));
      EvalTy term_eval = padded_term.template evaluate<EvalTy>(arithmetics);
      sum = arithmetics.add(std::move(sum), std::move(term_eval));
    }
    return sum;
  } else {
    EvalTy sum = it->template evaluate<EvalTy>(arithmetics);
    while (++it != end) {
      EvalTy term_eval = it->template evaluate<EvalTy>(arithmetics);
      sum = arithmetics.add(std::move(sum), std::move(term_eval));
    }
    return sum;
  }
}

#define INSTANTIATE_SUM_PRIVATE_METHODS(HandlerTy)                             \
                                                                               \
  template void sum_op<HandlerTy>::insert(product_op<HandlerTy> &&other);      \
                                                                               \
  template void sum_op<HandlerTy>::insert(const product_op<HandlerTy> &other); \
                                                                               \
  template void sum_op<HandlerTy>::aggregate_terms(                            \
      product_op<HandlerTy> &&item2);                                          \
                                                                               \
  template void sum_op<HandlerTy>::aggregate_terms(                            \
      product_op<HandlerTy> &&item1, product_op<HandlerTy> &&item2);           \
                                                                               \
  template void sum_op<HandlerTy>::aggregate_terms(                            \
      product_op<HandlerTy> &&item1, product_op<HandlerTy> &&item2,            \
      product_op<HandlerTy> &&item3);

#if !defined(__clang__)
INSTANTIATE_SUM_PRIVATE_METHODS(matrix_handler);
INSTANTIATE_SUM_PRIVATE_METHODS(spin_handler);
INSTANTIATE_SUM_PRIVATE_METHODS(boson_handler);
INSTANTIATE_SUM_PRIVATE_METHODS(fermion_handler);
#endif

#define INSTANTIATE_SUM_EVALUATE_METHODS(HandlerTy, EvalTy)                    \
                                                                               \
  template EvalTy sum_op<HandlerTy>::evaluate(                                 \
      operator_arithmetics<EvalTy> arithmetics) const;

#if !defined(__clang__)
INSTANTIATE_SUM_EVALUATE_METHODS(matrix_handler,
                                 operator_handler::matrix_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(spin_handler,
                                 operator_handler::canonical_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(boson_handler,
                                 operator_handler::matrix_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(fermion_handler,
                                 operator_handler::matrix_evaluation);
#endif

// read-only properties

template <typename HandlerTy>
std::vector<std::size_t>
sum_op<HandlerTy>::degrees(bool application_order) const {
  std::set<std::size_t> unsorted_degrees;
  for (const std::vector<HandlerTy> &term : this->terms) {
    for (const HandlerTy &op : term) {
      auto op_degrees = op.degrees();
      unsorted_degrees.insert(op_degrees.cbegin(), op_degrees.cend());
    }
  }
  auto degrees = std::vector<std::size_t>(unsorted_degrees.cbegin(),
                                          unsorted_degrees.cend());
  if (application_order)
    std::sort(degrees.begin(), degrees.end(),
              operator_handler::user_facing_order);
  else
    std::sort(degrees.begin(), degrees.end(),
              operator_handler::canonical_order);
  return std::move(degrees);
}

template <typename HandlerTy>
std::size_t sum_op<HandlerTy>::num_terms() const {
  return this->terms.size();
}

#define INSTANTIATE_SUM_PROPERTIES(HandlerTy)                                  \
                                                                               \
  template std::vector<std::size_t> sum_op<HandlerTy>::degrees(                \
      bool application_order) const;                                           \
                                                                               \
  template std::size_t sum_op<HandlerTy>::num_terms() const;

#if !defined(__clang__)
INSTANTIATE_SUM_PROPERTIES(matrix_handler);
INSTANTIATE_SUM_PROPERTIES(spin_handler);
INSTANTIATE_SUM_PROPERTIES(boson_handler);
INSTANTIATE_SUM_PROPERTIES(fermion_handler);
#endif

// constructors

template <typename HandlerTy>
sum_op<HandlerTy>::sum_op(const product_op<HandlerTy> &prod) {
  this->insert(prod);
}

template <typename HandlerTy>
template <
    typename... Args,
    std::enable_if_t<
        std::conjunction<std::is_same<product_op<HandlerTy>, Args>...>::value &&
            sizeof...(Args),
        bool>>
sum_op<HandlerTy>::sum_op(Args &&...args) {
  this->coefficients.reserve(sizeof...(Args));
  this->term_map.reserve(sizeof...(Args));
  this->terms.reserve(sizeof...(Args));
  aggregate_terms(std::forward<product_op<HandlerTy> &&>(args)...);
}

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                               std::is_constructible<HandlerTy, T>::value,
                           bool>>
sum_op<HandlerTy>::sum_op(const sum_op<T> &other)
    : coefficients(other.coefficients) {
  this->term_map.reserve(other.terms.size());
  this->terms.reserve(other.terms.size());
  for (const auto &operators : other.terms) {
    product_op<HandlerTy> term(
        product_op<T>(1., operators)); // coefficient does not matter
    this->term_map.insert(
        this->term_map.cend(),
        std::make_pair(term.get_term_id(), this->terms.size()));
    this->terms.push_back(std::move(term.operators));
  }
}

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<std::is_same<HandlerTy, matrix_handler>::value &&
                               !std::is_same<T, HandlerTy>::value &&
                               std::is_constructible<HandlerTy, T>::value,
                           bool>>
sum_op<HandlerTy>::sum_op(const sum_op<T> &other,
                          const matrix_handler::commutation_behavior &behavior)
    : coefficients(other.coefficients) {
  this->term_map.reserve(other.terms.size());
  this->terms.reserve(other.terms.size());
  for (const auto &operators : other.terms) {
    product_op<HandlerTy> term(product_op<T>(1., operators),
                               behavior); // coefficient does not matter
    this->term_map.insert(
        this->term_map.cend(),
        std::make_pair(term.get_term_id(), this->terms.size()));
    this->terms.push_back(std::move(term.operators));
  }
}

template <typename HandlerTy>
sum_op<HandlerTy>::sum_op(const sum_op<HandlerTy> &other, bool sized,
                          int size) {
  if (!sized) {
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
sum_op<HandlerTy>::sum_op(const sum_op<HandlerTy> &other)
    : sum_op(other, false, 0) {}

template <typename HandlerTy>
sum_op<HandlerTy>::sum_op(sum_op<HandlerTy> &&other, bool sized, int size)
    : coefficients(std::move(other.coefficients)),
      term_map(std::move(other.term_map)), terms(std::move(other.terms)) {
  if (sized) {
    this->coefficients.reserve(size);
    this->term_map.reserve(size);
    this->terms.reserve(size);
  }
}

template <typename HandlerTy>
sum_op<HandlerTy>::sum_op(sum_op<HandlerTy> &&other)
    : sum_op(std::move(other), false, 0) {}

#define INSTANTIATE_SUM_CONSTRUCTORS(HandlerTy)                                \
                                                                               \
  template sum_op<HandlerTy>::sum_op();                                        \
                                                                               \
  template sum_op<HandlerTy>::sum_op(const product_op<HandlerTy> &item2);      \
                                                                               \
  template sum_op<HandlerTy>::sum_op(product_op<HandlerTy> &&item2);           \
                                                                               \
  template sum_op<HandlerTy>::sum_op(product_op<HandlerTy> &&item1,            \
                                     product_op<HandlerTy> &&item2);           \
                                                                               \
  template sum_op<HandlerTy>::sum_op(product_op<HandlerTy> &&item1,            \
                                     product_op<HandlerTy> &&item2,            \
                                     product_op<HandlerTy> &&item3);           \
                                                                               \
  template sum_op<HandlerTy>::sum_op(const sum_op<HandlerTy> &other,           \
                                     bool sized, int size);                    \
                                                                               \
  template sum_op<HandlerTy>::sum_op(const sum_op<HandlerTy> &other);          \
                                                                               \
  template sum_op<HandlerTy>::sum_op(sum_op<HandlerTy> &&other, bool sized,    \
                                     int size);                                \
                                                                               \
  template sum_op<HandlerTy>::sum_op(sum_op<HandlerTy> &&other);

// Note:
// These are the private constructors needed by friend classes and functions
// of sum_op. For clang, (only!) these need to be instantiated explicitly
// to be available to those.
#define INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(HandlerTy)                 \
                                                                               \
  template sum_op<HandlerTy>::sum_op(product_op<HandlerTy> &&item1,            \
                                     product_op<HandlerTy> &&item2);

template sum_op<matrix_handler>::sum_op(const sum_op<spin_handler> &other);
template sum_op<matrix_handler>::sum_op(const sum_op<boson_handler> &other);
template sum_op<matrix_handler>::sum_op(const sum_op<fermion_handler> &other);
template sum_op<matrix_handler>::sum_op(
    const sum_op<spin_handler> &other,
    const matrix_handler::commutation_behavior &behavior);
template sum_op<matrix_handler>::sum_op(
    const sum_op<boson_handler> &other,
    const matrix_handler::commutation_behavior &behavior);
template sum_op<matrix_handler>::sum_op(
    const sum_op<fermion_handler> &other,
    const matrix_handler::commutation_behavior &behavior);

#if !defined(__clang__)
INSTANTIATE_SUM_CONSTRUCTORS(matrix_handler);
INSTANTIATE_SUM_CONSTRUCTORS(spin_handler);
INSTANTIATE_SUM_CONSTRUCTORS(boson_handler);
INSTANTIATE_SUM_CONSTRUCTORS(fermion_handler);
#else
INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(matrix_handler);
INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(spin_handler);
INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(boson_handler);
INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(fermion_handler);
#endif

// assignments

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                               std::is_constructible<HandlerTy, T>::value,
                           bool>>
sum_op<HandlerTy> &sum_op<HandlerTy>::operator=(const product_op<T> &other) {
  *this = product_op<HandlerTy>(other);
  return *this;
}

template <typename HandlerTy>
sum_op<HandlerTy> &
sum_op<HandlerTy>::operator=(const product_op<HandlerTy> &other) {
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
sum_op<HandlerTy> &sum_op<HandlerTy>::operator=(product_op<HandlerTy> &&other) {
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
sum_op<HandlerTy> &sum_op<HandlerTy>::operator=(const sum_op<T> &other) {
  *this = sum_op<HandlerTy>(other);
  return *this;
}

template <typename HandlerTy>
sum_op<HandlerTy> &
sum_op<HandlerTy>::operator=(const sum_op<HandlerTy> &other) {
  if (this != &other) {
    this->coefficients = other.coefficients;
    this->term_map = other.term_map;
    this->terms = other.terms;
  }
  return *this;
}

template <typename HandlerTy>
sum_op<HandlerTy> &sum_op<HandlerTy>::operator=(sum_op<HandlerTy> &&other) {
  if (this != &other) {
    this->coefficients = std::move(other.coefficients);
    this->term_map = std::move(other.term_map);
    this->terms = std::move(other.terms);
  }
  return *this;
}

#define INSTANTIATE_SUM_ASSIGNMENTS(HandlerTy)                                 \
                                                                               \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator=(                    \
      product_op<HandlerTy> &&other);                                          \
                                                                               \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator=(                    \
      const product_op<HandlerTy> &other);                                     \
                                                                               \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator=(                    \
      const sum_op<HandlerTy> &other);                                         \
                                                                               \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator=(                    \
      sum_op<HandlerTy> &&other);

template sum_op<matrix_handler> &
sum_op<matrix_handler>::operator=(const product_op<spin_handler> &other);
template sum_op<matrix_handler> &
sum_op<matrix_handler>::operator=(const product_op<boson_handler> &other);
template sum_op<matrix_handler> &
sum_op<matrix_handler>::operator=(const product_op<fermion_handler> &other);
template sum_op<matrix_handler> &
sum_op<matrix_handler>::operator=(const sum_op<spin_handler> &other);
template sum_op<matrix_handler> &
sum_op<matrix_handler>::operator=(const sum_op<boson_handler> &other);
template sum_op<matrix_handler> &
sum_op<matrix_handler>::operator=(const sum_op<fermion_handler> &other);

#if !defined(__clang__)
INSTANTIATE_SUM_ASSIGNMENTS(matrix_handler);
INSTANTIATE_SUM_ASSIGNMENTS(spin_handler);
INSTANTIATE_SUM_ASSIGNMENTS(boson_handler);
INSTANTIATE_SUM_ASSIGNMENTS(fermion_handler);
#endif

// evaluations

template <typename HandlerTy>
std::string sum_op<HandlerTy>::to_string() const {
  if (this->terms.size() == 0)
    return "";
  auto it = this->begin();
  std::string str = it->to_string();
  while (++it != this->end())
    str += " + " + it->to_string();
  return std::move(str);
}

template <typename HandlerTy>
complex_matrix sum_op<HandlerTy>::to_matrix(
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
complex_matrix sum_op<spin_handler>::to_matrix(
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
  auto matrix = spin_handler::to_matrix(evaluated.terms[0].second,
                                        evaluated.terms[0].first, invert_order);
  for (auto i = 1; i < terms.size(); ++i)
    matrix += spin_handler::to_matrix(evaluated.terms[i].second,
                                      evaluated.terms[i].first, invert_order);
  return std::move(matrix);
}

#define INSTANTIATE_SUM_EVALUATIONS(HandlerTy)                                 \
                                                                               \
  template std::string sum_op<HandlerTy>::to_string() const;                   \
                                                                               \
  template complex_matrix sum_op<HandlerTy>::to_matrix(                        \
      std::unordered_map<int, int> dimensions,                                 \
      const std::unordered_map<std::string, std::complex<double>> &params,     \
      bool application_order) const;

#if !defined(__clang__)
INSTANTIATE_SUM_EVALUATIONS(matrix_handler);
INSTANTIATE_SUM_EVALUATIONS(spin_handler);
INSTANTIATE_SUM_EVALUATIONS(boson_handler);
INSTANTIATE_SUM_EVALUATIONS(fermion_handler);
#endif

// comparisons

template <typename HandlerTy>
bool sum_op<HandlerTy>::operator==(const sum_op<HandlerTy> &other) const {
  if (this->terms.size() != other.terms.size())
    return false;
  std::vector<std::string> self_keys;
  std::vector<std::string> other_keys;
  self_keys.reserve(this->terms.size());
  other_keys.reserve(other.terms.size());
  for (const auto &entry : this->term_map)
    self_keys.push_back(entry.first);
  for (const auto &entry : other.term_map)
    other_keys.push_back(entry.first);
  std::sort(self_keys.begin(), self_keys.end());
  std::sort(other_keys.begin(), other_keys.end());
  if (self_keys != other_keys)
    return false;
  for (const auto &key : self_keys) {
    auto self_idx = this->term_map.find(key)->second;
    auto other_idx = other.term_map.find(key)->second;
    if (this->coefficients[self_idx] != other.coefficients[other_idx])
      return false;
  }
  return true;
}

#define INSTANTIATE_SUM_COMPARISONS(HandlerTy)                                 \
  template bool sum_op<HandlerTy>::operator==(const sum_op<HandlerTy> &other)  \
      const;

#if !defined(__clang__)
INSTANTIATE_SUM_COMPARISONS(matrix_handler);
INSTANTIATE_SUM_COMPARISONS(spin_handler);
INSTANTIATE_SUM_COMPARISONS(boson_handler);
INSTANTIATE_SUM_COMPARISONS(fermion_handler);
#endif

// unary operators

template <typename HandlerTy>
sum_op<HandlerTy> sum_op<HandlerTy>::operator-() const & {
  sum_op<HandlerTy> sum;
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map = this->term_map;
  sum.terms = this->terms;
  for (auto &coeff : this->coefficients)
    sum.coefficients.push_back(-1. * coeff);
  return std::move(sum);
}

template <typename HandlerTy>
sum_op<HandlerTy> sum_op<HandlerTy>::operator-() && {
  for (auto &coeff : this->coefficients)
    coeff *= -1.;
  return std::move(*this);
}

template <typename HandlerTy>
sum_op<HandlerTy> sum_op<HandlerTy>::operator+() const & {
  return *this;
}

template <typename HandlerTy>
sum_op<HandlerTy> sum_op<HandlerTy>::operator+() && {
  return std::move(*this);
}

#define INSTANTIATE_SUM_UNARY_OPS(HandlerTy)                                   \
                                                                               \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-() const &;           \
                                                                               \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-() &&;                \
                                                                               \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+() const &;           \
                                                                               \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+() &&;

#if !defined(__clang__)
INSTANTIATE_SUM_UNARY_OPS(matrix_handler);
INSTANTIATE_SUM_UNARY_OPS(spin_handler);
INSTANTIATE_SUM_UNARY_OPS(boson_handler);
INSTANTIATE_SUM_UNARY_OPS(fermion_handler);
#endif

// right-hand arithmetics

#define SUM_MULTIPLICATION_SCALAR(op)                                          \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const scalar_operator &other) const & {                                  \
    sum_op<HandlerTy> sum;                                                     \
    sum.coefficients.reserve(this->coefficients.size());                       \
    sum.term_map = this->term_map;                                             \
    sum.terms = this->terms;                                                   \
    for (const auto &coeff : this->coefficients)                               \
      sum.coefficients.push_back(coeff op other);                              \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
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
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const scalar_operator &other) const & {                                  \
    sum_op<HandlerTy> sum(*this, true, this->terms.size() + 1);                \
    sum.insert(product_op<HandlerTy>(op other));                               \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(scalar_operator &&other)    \
      const & {                                                                \
    sum_op<HandlerTy> sum(*this, true, this->terms.size() + 1);                \
    sum.insert(product_op<HandlerTy>(op std::move(other)));                    \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const scalar_operator &other) && {                                       \
    this->insert(product_op<HandlerTy>(op other));                             \
    return std::move(*this);                                                   \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      scalar_operator &&other) && {                                            \
    this->insert(product_op<HandlerTy>(op std::move(other)));                  \
    return std::move(*this);                                                   \
  }

SUM_ADDITION_SCALAR(+);
SUM_ADDITION_SCALAR(-);

#define INSTANTIATE_SUM_RHSIMPLE_OPS(HandlerTy)                                \
                                                                               \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator*(                     \
      const scalar_operator &other) const &;                                   \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator*(                     \
      const scalar_operator &other) &&;                                        \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator/(                     \
      const scalar_operator &other) const &;                                   \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator/(                     \
      const scalar_operator &other) &&;                                        \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      scalar_operator &&other) const &;                                        \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      scalar_operator &&other) &&;                                             \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      const scalar_operator &other) const &;                                   \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      const scalar_operator &other) &&;                                        \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      scalar_operator &&other) const &;                                        \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      scalar_operator &&other) &&;                                             \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      const scalar_operator &other) const &;                                   \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      const scalar_operator &other) &&;

#if !defined(__clang__)
INSTANTIATE_SUM_RHSIMPLE_OPS(matrix_handler);
INSTANTIATE_SUM_RHSIMPLE_OPS(spin_handler);
INSTANTIATE_SUM_RHSIMPLE_OPS(boson_handler);
INSTANTIATE_SUM_RHSIMPLE_OPS(fermion_handler);
#endif

template <typename HandlerTy>
sum_op<HandlerTy>
sum_op<HandlerTy>::operator*(const product_op<HandlerTy> &other) const {
  sum_op<HandlerTy> sum; // the entire sum needs to be rebuilt
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map.reserve(this->terms.size());
  sum.terms.reserve(this->terms.size());
  for (auto i = 0; i < this->terms.size(); ++i) {
    auto max_size = this->terms[i].size() + other.operators.size();
    product_op<HandlerTy> prod(this->coefficients[i] * other.coefficient,
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
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const product_op<HandlerTy> &other) const & {                            \
    sum_op<HandlerTy> sum(*this, true, this->terms.size() + 1);                \
    sum.insert(op other);                                                      \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const product_op<HandlerTy> &other) && {                                 \
    this->insert(op other);                                                    \
    return std::move(*this);                                                   \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      product_op<HandlerTy> &&other) const & {                                 \
    sum_op<HandlerTy> sum(*this, true, this->terms.size() + 1);                \
    sum.insert(op std::move(other));                                           \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      product_op<HandlerTy> &&other) && {                                      \
    this->insert(op std::move(other));                                         \
    return std::move(*this);                                                   \
  }

SUM_ADDITION_PRODUCT(+)
SUM_ADDITION_PRODUCT(-)

template <typename HandlerTy>
sum_op<HandlerTy>
sum_op<HandlerTy>::operator*(const sum_op<HandlerTy> &other) const {
  sum_op<HandlerTy> sum; // the entire sum needs to be rebuilt
  auto max_size = this->terms.size() * other.terms.size();
  sum.coefficients.reserve(max_size);
  sum.term_map.reserve(max_size);
  sum.terms.reserve(max_size);
  for (auto i = 0; i < this->terms.size(); ++i) {
    for (auto j = 0; j < other.terms.size(); ++j) {
      auto max_size = this->terms[i].size() + other.terms[j].size();
      product_op<HandlerTy> prod(this->coefficients[i] * other.coefficients[j],
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
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const sum_op<HandlerTy> &other) const & {                                \
    sum_op<HandlerTy> sum(*this, true,                                         \
                          this->terms.size() + other.terms.size());            \
    for (auto i = 0; i < other.terms.size(); ++i) {                            \
      product_op<HandlerTy> prod(op other.coefficients[i], other.terms[i]);    \
      sum.insert(std::move(prod));                                             \
    }                                                                          \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const sum_op<HandlerTy> &other) && {                                     \
    auto max_size = this->terms.size() + other.terms.size();                   \
    this->coefficients.reserve(max_size);                                      \
    this->term_map.reserve(max_size);                                          \
    this->terms.reserve(max_size);                                             \
    for (auto i = 0; i < other.terms.size(); ++i)                              \
      this->insert(                                                            \
          product_op<HandlerTy>(op other.coefficients[i], other.terms[i]));    \
    return std::move(*this);                                                   \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(sum_op<HandlerTy> &&other)  \
      const & {                                                                \
    sum_op<HandlerTy> sum(*this, true,                                         \
                          this->terms.size() + other.terms.size());            \
    for (auto i = 0; i < other.terms.size(); ++i) {                            \
      product_op<HandlerTy> prod(op std::move(other.coefficients[i]),          \
                                 std::move(other.terms[i]));                   \
      sum.insert(std::move(prod));                                             \
    }                                                                          \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      sum_op<HandlerTy> &&other) && {                                          \
    auto max_size = this->terms.size() + other.terms.size();                   \
    this->coefficients.reserve(max_size);                                      \
    this->term_map.reserve(max_size);                                          \
    this->terms.reserve(max_size);                                             \
    for (auto i = 0; i < other.terms.size(); ++i)                              \
      this->insert(product_op<HandlerTy>(op std::move(other.coefficients[i]),  \
                                         std::move(other.terms[i])));          \
    return std::move(*this);                                                   \
  }

SUM_ADDITION_SUM(+);
SUM_ADDITION_SUM(-);

#define INSTANTIATE_SUM_RHCOMPOSITE_OPS(HandlerTy)                             \
                                                                               \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator*(                     \
      const product_op<HandlerTy> &other) const;                               \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      const product_op<HandlerTy> &other) const &;                             \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      const product_op<HandlerTy> &other) &&;                                  \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      product_op<HandlerTy> &&other) const &;                                  \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      product_op<HandlerTy> &&other) &&;                                       \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      const product_op<HandlerTy> &other) const &;                             \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      const product_op<HandlerTy> &other) &&;                                  \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      product_op<HandlerTy> &&other) const &;                                  \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      product_op<HandlerTy> &&other) &&;                                       \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator*(                     \
      const sum_op<HandlerTy> &other) const;                                   \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      const sum_op<HandlerTy> &other) const &;                                 \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      const sum_op<HandlerTy> &other) &&;                                      \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      sum_op<HandlerTy> &&other) const &;                                      \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator+(                     \
      sum_op<HandlerTy> &&other) &&;                                           \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      const sum_op<HandlerTy> &other) const &;                                 \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      const sum_op<HandlerTy> &other) &&;                                      \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      sum_op<HandlerTy> &&other) const &;                                      \
  template sum_op<HandlerTy> sum_op<HandlerTy>::operator-(                     \
      sum_op<HandlerTy> &&other) &&;

#if !defined(__clang__)
INSTANTIATE_SUM_RHCOMPOSITE_OPS(matrix_handler);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(spin_handler);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(boson_handler);
INSTANTIATE_SUM_RHCOMPOSITE_OPS(fermion_handler);
#endif

#define SUM_MULTIPLICATION_SCALAR_ASSIGNMENT(op)                               \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> &sum_op<HandlerTy>::operator op(                           \
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
  sum_op<HandlerTy> &sum_op<HandlerTy>::operator op##=(                        \
      const scalar_operator &other) {                                          \
    this->insert(product_op<HandlerTy>(op other));                             \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> &sum_op<HandlerTy>::operator op##=(                        \
      scalar_operator &&other) {                                               \
    this->insert(product_op<HandlerTy>(op std::move(other)));                  \
    return *this;                                                              \
  }

SUM_ADDITION_SCALAR_ASSIGNMENT(+);
SUM_ADDITION_SCALAR_ASSIGNMENT(-);

template <typename HandlerTy>
sum_op<HandlerTy> &
sum_op<HandlerTy>::operator*=(const product_op<HandlerTy> &other) {
  sum_op<HandlerTy> sum;
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map.reserve(this->terms.size());
  sum.terms.reserve(this->terms.size());
  for (auto i = 0; i < this->terms.size(); ++i) {
    auto max_size = this->terms[i].size() + other.operators.size();
    product_op<HandlerTy> prod(this->coefficients[i] * other.coefficient,
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
  sum_op<HandlerTy> &sum_op<HandlerTy>::operator op##=(                        \
      const product_op<HandlerTy> &other) {                                    \
    this->insert(op other);                                                    \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> &sum_op<HandlerTy>::operator op##=(                        \
      product_op<HandlerTy> &&other) {                                         \
    this->insert(op std::move(other));                                         \
    return *this;                                                              \
  }

SUM_ADDITION_PRODUCT_ASSIGNMENT(+)
SUM_ADDITION_PRODUCT_ASSIGNMENT(-)

template <typename HandlerTy>
sum_op<HandlerTy> &
sum_op<HandlerTy>::operator*=(const sum_op<HandlerTy> &other) {
  sum_op<HandlerTy> sum; // the entire sum needs to be rebuilt
  auto max_size = this->terms.size() * other.terms.size();
  sum.coefficients.reserve(max_size);
  sum.term_map.reserve(max_size);
  sum.terms.reserve(max_size);
  for (auto i = 0; i < this->terms.size(); ++i) {
    for (auto j = 0; j < other.terms.size(); ++j) {
      auto max_size = this->terms[i].size() + other.terms[j].size();
      product_op<HandlerTy> prod(this->coefficients[i] * other.coefficients[j],
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
  sum_op<HandlerTy> &sum_op<HandlerTy>::operator op##=(                        \
      const sum_op<HandlerTy> &other) {                                        \
    auto max_size = this->terms.size() + other.terms.size();                   \
    this->coefficients.reserve(max_size);                                      \
    this->term_map.reserve(max_size);                                          \
    this->terms.reserve(max_size);                                             \
    for (auto i = 0; i < other.terms.size(); ++i)                              \
      this->insert(                                                            \
          product_op<HandlerTy>(op other.coefficients[i], other.terms[i]));    \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> &sum_op<HandlerTy>::operator op##=(                        \
      sum_op<HandlerTy> &&other) {                                             \
    auto max_size = this->terms.size() + other.terms.size();                   \
    this->coefficients.reserve(max_size);                                      \
    this->term_map.reserve(max_size);                                          \
    this->terms.reserve(max_size);                                             \
    for (auto i = 0; i < other.terms.size(); ++i)                              \
      this->insert(product_op<HandlerTy>(op std::move(other.coefficients[i]),  \
                                         std::move(other.terms[i])));          \
    return *this;                                                              \
  }

SUM_ADDITION_SUM_ASSIGNMENT(+);
SUM_ADDITION_SUM_ASSIGNMENT(-);

#define INSTANTIATE_SUM_OPASSIGNMENTS(HandlerTy)                               \
                                                                               \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator*=(                   \
      const scalar_operator &other);                                           \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator/=(                   \
      const scalar_operator &other);                                           \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator+=(                   \
      scalar_operator &&other);                                                \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator+=(                   \
      const scalar_operator &other);                                           \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator-=(                   \
      scalar_operator &&other);                                                \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator-=(                   \
      const scalar_operator &other);                                           \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator*=(                   \
      const product_op<HandlerTy> &other);                                     \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator+=(                   \
      const product_op<HandlerTy> &other);                                     \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator+=(                   \
      product_op<HandlerTy> &&other);                                          \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator-=(                   \
      const product_op<HandlerTy> &other);                                     \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator-=(                   \
      product_op<HandlerTy> &&other);                                          \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator*=(                   \
      const sum_op<HandlerTy> &other);                                         \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator+=(                   \
      const sum_op<HandlerTy> &other);                                         \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator+=(                   \
      sum_op<HandlerTy> &&other);                                              \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator-=(                   \
      const sum_op<HandlerTy> &other);                                         \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator-=(                   \
      sum_op<HandlerTy> &&other);

#if !defined(__clang__)
INSTANTIATE_SUM_OPASSIGNMENTS(matrix_handler);
INSTANTIATE_SUM_OPASSIGNMENTS(spin_handler);
INSTANTIATE_SUM_OPASSIGNMENTS(boson_handler);
INSTANTIATE_SUM_OPASSIGNMENTS(fermion_handler);
#endif

// left-hand arithmetics

template <typename HandlerTy>
sum_op<HandlerTy> operator*(const scalar_operator &other,
                            const sum_op<HandlerTy> &self) {
  sum_op<HandlerTy> sum;
  sum.coefficients.reserve(self.coefficients.size());
  sum.terms = self.terms;
  sum.term_map = self.term_map;
  for (const auto &coeff : self.coefficients)
    sum.coefficients.push_back(coeff * other);
  return std::move(sum);
}

template <typename HandlerTy>
sum_op<HandlerTy> operator*(const scalar_operator &other,
                            sum_op<HandlerTy> &&self) {
  for (auto &&coeff : self.coefficients)
    coeff *= other;
  return std::move(self);
}

#define SUM_ADDITION_SCALAR_REVERSE(op)                                        \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(const scalar_operator &other,                  \
                                const sum_op<HandlerTy> &self) {               \
    sum_op<HandlerTy> sum(op self);                                            \
    sum.insert(product_op<HandlerTy>(other));                                  \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(scalar_operator &&other,                       \
                                const sum_op<HandlerTy> &self) {               \
    sum_op<HandlerTy> sum(op self);                                            \
    sum.insert(product_op<HandlerTy>(std::move(other)));                       \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(const scalar_operator &other,                  \
                                sum_op<HandlerTy> &&self) {                    \
    for (auto &&coeff : self.coefficients)                                     \
      coeff = std::move(op coeff);                                             \
    self.insert(product_op<HandlerTy>(other));                                 \
    return std::move(self);                                                    \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(scalar_operator &&other,                       \
                                sum_op<HandlerTy> &&self) {                    \
    for (auto &&coeff : self.coefficients)                                     \
      coeff = std::move(op coeff);                                             \
    self.insert(product_op<HandlerTy>(std::move(other)));                      \
    return std::move(self);                                                    \
  }

SUM_ADDITION_SCALAR_REVERSE(+);
SUM_ADDITION_SCALAR_REVERSE(-);

#define INSTANTIATE_SUM_LHCOMPOSITE_OPS(HandlerTy)                             \
                                                                               \
  template sum_op<HandlerTy> operator*(const scalar_operator &other,           \
                                       const sum_op<HandlerTy> &self);         \
  template sum_op<HandlerTy> operator*(const scalar_operator &other,           \
                                       sum_op<HandlerTy> &&self);              \
  template sum_op<HandlerTy> operator+(scalar_operator &&other,                \
                                       const sum_op<HandlerTy> &self);         \
  template sum_op<HandlerTy> operator+(scalar_operator &&other,                \
                                       sum_op<HandlerTy> &&self);              \
  template sum_op<HandlerTy> operator+(const scalar_operator &other,           \
                                       const sum_op<HandlerTy> &self);         \
  template sum_op<HandlerTy> operator+(const scalar_operator &other,           \
                                       sum_op<HandlerTy> &&self);              \
  template sum_op<HandlerTy> operator-(scalar_operator &&other,                \
                                       const sum_op<HandlerTy> &self);         \
  template sum_op<HandlerTy> operator-(scalar_operator &&other,                \
                                       sum_op<HandlerTy> &&self);              \
  template sum_op<HandlerTy> operator-(const scalar_operator &other,           \
                                       const sum_op<HandlerTy> &self);         \
  template sum_op<HandlerTy> operator-(const scalar_operator &other,           \
                                       sum_op<HandlerTy> &&self);

INSTANTIATE_SUM_LHCOMPOSITE_OPS(matrix_handler);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(spin_handler);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(boson_handler);
INSTANTIATE_SUM_LHCOMPOSITE_OPS(fermion_handler);

// arithmetics that require conversions

#define SUM_CONVERSIONS_OPS(op)                                                \
                                                                               \
  template <typename LHtype, typename RHtype,                                  \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype)>                        \
  sum_op<matrix_handler> operator op(const sum_op<LHtype> &other,              \
                                     const product_op<RHtype> &self) {         \
    return sum_op<matrix_handler>(other) op self;                              \
  }                                                                            \
                                                                               \
  template <typename LHtype, typename RHtype,                                  \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype)>                        \
  sum_op<matrix_handler> operator op(const product_op<LHtype> &other,          \
                                     const sum_op<RHtype> &self) {             \
    return product_op<matrix_handler>(other) op self;                          \
  }                                                                            \
                                                                               \
  template <typename LHtype, typename RHtype,                                  \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype)>                        \
  sum_op<matrix_handler> operator op(const sum_op<LHtype> &other,              \
                                     const sum_op<RHtype> &self) {             \
    return sum_op<matrix_handler>(other) op self;                              \
  }

SUM_CONVERSIONS_OPS(*);
SUM_CONVERSIONS_OPS(+);
SUM_CONVERSIONS_OPS(-);

#define INSTANTIATE_SUM_CONVERSION_OPS(op)                                     \
                                                                               \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<spin_handler> &other,                                       \
      const product_op<matrix_handler> &self);                                 \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<boson_handler> &other,                                      \
      const product_op<matrix_handler> &self);                                 \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<fermion_handler> &other,                                    \
      const product_op<matrix_handler> &self);                                 \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<spin_handler> &other,                                       \
      const product_op<boson_handler> &self);                                  \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<boson_handler> &other,                                      \
      const product_op<spin_handler> &self);                                   \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<spin_handler> &other,                                       \
      const product_op<fermion_handler> &self);                                \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<fermion_handler> &other,                                    \
      const product_op<spin_handler> &self);                                   \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<boson_handler> &other,                                      \
      const product_op<fermion_handler> &self);                                \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<fermion_handler> &other,                                    \
      const product_op<boson_handler> &self);                                  \
                                                                               \
  template sum_op<matrix_handler> operator op(                                 \
      const product_op<spin_handler> &other,                                   \
      const sum_op<matrix_handler> &self);                                     \
  template sum_op<matrix_handler> operator op(                                 \
      const product_op<boson_handler> &other,                                  \
      const sum_op<matrix_handler> &self);                                     \
  template sum_op<matrix_handler> operator op(                                 \
      const product_op<fermion_handler> &other,                                \
      const sum_op<matrix_handler> &self);                                     \
  template sum_op<matrix_handler> operator op(                                 \
      const product_op<spin_handler> &other,                                   \
      const sum_op<boson_handler> &self);                                      \
  template sum_op<matrix_handler> operator op(                                 \
      const product_op<boson_handler> &other,                                  \
      const sum_op<spin_handler> &self);                                       \
  template sum_op<matrix_handler> operator op(                                 \
      const product_op<spin_handler> &other,                                   \
      const sum_op<fermion_handler> &self);                                    \
  template sum_op<matrix_handler> operator op(                                 \
      const product_op<fermion_handler> &other,                                \
      const sum_op<spin_handler> &self);                                       \
  template sum_op<matrix_handler> operator op(                                 \
      const product_op<boson_handler> &other,                                  \
      const sum_op<fermion_handler> &self);                                    \
  template sum_op<matrix_handler> operator op(                                 \
      const product_op<fermion_handler> &other,                                \
      const sum_op<boson_handler> &self);                                      \
                                                                               \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<spin_handler> &other, const sum_op<matrix_handler> &self);  \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<boson_handler> &other, const sum_op<matrix_handler> &self); \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<fermion_handler> &other,                                    \
      const sum_op<matrix_handler> &self);                                     \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<spin_handler> &other, const sum_op<boson_handler> &self);   \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<boson_handler> &other, const sum_op<spin_handler> &self);   \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<spin_handler> &other, const sum_op<fermion_handler> &self); \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<fermion_handler> &other, const sum_op<spin_handler> &self); \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<boson_handler> &other,                                      \
      const sum_op<fermion_handler> &self);                                    \
  template sum_op<matrix_handler> operator op(                                 \
      const sum_op<fermion_handler> &other,                                    \
      const sum_op<boson_handler> &self);

INSTANTIATE_SUM_CONVERSION_OPS(*);
INSTANTIATE_SUM_CONVERSION_OPS(+);
INSTANTIATE_SUM_CONVERSION_OPS(-);

// common operators

template <typename HandlerTy>
sum_op<HandlerTy> sum_op<HandlerTy>::empty() {
  return sum_op<HandlerTy>();
}

template <typename HandlerTy>
product_op<HandlerTy> sum_op<HandlerTy>::identity() {
  return product_op<HandlerTy>(1.0);
}

template <typename HandlerTy>
product_op<HandlerTy> sum_op<HandlerTy>::identity(int target) {
  static_assert(
      std::is_constructible_v<HandlerTy, int>,
      "operator handlers must have a constructor that take a single degree of "
      "freedom and returns the identity operator on that degree.");
  return product_op<HandlerTy>(1.0, HandlerTy(target));
}

#if !defined(__clang__)
template sum_op<matrix_handler> sum_op<matrix_handler>::empty();
template sum_op<spin_handler> sum_op<spin_handler>::empty();
template sum_op<boson_handler> sum_op<boson_handler>::empty();
template sum_op<fermion_handler> sum_op<fermion_handler>::empty();
template product_op<matrix_handler> sum_op<matrix_handler>::identity();
template product_op<spin_handler> sum_op<spin_handler>::identity();
template product_op<boson_handler> sum_op<boson_handler>::identity();
template product_op<fermion_handler> sum_op<fermion_handler>::identity();
template product_op<matrix_handler>
sum_op<matrix_handler>::identity(int target);
template product_op<spin_handler> sum_op<spin_handler>::identity(int target);
template product_op<boson_handler> sum_op<boson_handler>::identity(int target);
template product_op<fermion_handler>
sum_op<fermion_handler>::identity(int target);
#endif

// handler specific operators

#define HANDLER_SPECIFIC_TEMPLATE_DEFINITION(ConcreteTy)                       \
  template <typename HandlerTy>                                                \
  template <typename T,                                                        \
            std::enable_if_t<std::is_same<T, ConcreteTy>::value &&             \
                                 std::is_same<HandlerTy, T>::value,            \
                             bool>>

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::number(int target) {
  return matrix_handler::number(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::parity(int target) {
  return matrix_handler::parity(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::position(int target) {
  return matrix_handler::position(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::momentum(int target) {
  return matrix_handler::momentum(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::squeeze(int target) {
  return matrix_handler::squeeze(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::displace(int target) {
  return matrix_handler::displace(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
product_op<T> sum_op<HandlerTy>::i(int target) {
  return spin_handler::i(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
product_op<T> sum_op<HandlerTy>::x(int target) {
  return spin_handler::x(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
product_op<T> sum_op<HandlerTy>::y(int target) {
  return spin_handler::y(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
product_op<T> sum_op<HandlerTy>::z(int target) {
  return spin_handler::z(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
sum_op<T> sum_op<HandlerTy>::plus(int target) {
  return spin_handler::plus(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
sum_op<T> sum_op<HandlerTy>::minus(int target) {
  return spin_handler::minus(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(boson_handler)
product_op<T> sum_op<HandlerTy>::create(int target) {
  return boson_handler::create(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(boson_handler)
product_op<T> sum_op<HandlerTy>::annihilate(int target) {
  return boson_handler::annihilate(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(boson_handler)
product_op<T> sum_op<HandlerTy>::number(int target) {
  return boson_handler::number(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(boson_handler)
sum_op<T> sum_op<HandlerTy>::position(int target) {
  return boson_handler::position(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(boson_handler)
sum_op<T> sum_op<HandlerTy>::momentum(int target) {
  return boson_handler::momentum(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(fermion_handler)
product_op<T> sum_op<HandlerTy>::create(int target) {
  return fermion_handler::create(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(fermion_handler)
product_op<T> sum_op<HandlerTy>::annihilate(int target) {
  return fermion_handler::annihilate(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(fermion_handler)
product_op<T> sum_op<HandlerTy>::number(int target) {
  return fermion_handler::number(target);
}

template product_op<matrix_handler> sum_op<matrix_handler>::number(int target);
template product_op<matrix_handler> sum_op<matrix_handler>::parity(int target);
template product_op<matrix_handler>
sum_op<matrix_handler>::position(int target);
template product_op<matrix_handler>
sum_op<matrix_handler>::momentum(int target);
template product_op<matrix_handler> sum_op<matrix_handler>::squeeze(int target);
template product_op<matrix_handler>
sum_op<matrix_handler>::displace(int target);

template product_op<spin_handler> sum_op<spin_handler>::i(int target);
template product_op<spin_handler> sum_op<spin_handler>::x(int target);
template product_op<spin_handler> sum_op<spin_handler>::y(int target);
template product_op<spin_handler> sum_op<spin_handler>::z(int target);
template sum_op<spin_handler> sum_op<spin_handler>::plus(int target);
template sum_op<spin_handler> sum_op<spin_handler>::minus(int target);

template product_op<boson_handler> sum_op<boson_handler>::create(int target);
template product_op<boson_handler>
sum_op<boson_handler>::annihilate(int target);
template product_op<boson_handler> sum_op<boson_handler>::number(int target);
template sum_op<boson_handler> sum_op<boson_handler>::position(int target);
template sum_op<boson_handler> sum_op<boson_handler>::momentum(int target);

template product_op<fermion_handler>
sum_op<fermion_handler>::create(int target);
template product_op<fermion_handler>
sum_op<fermion_handler>::annihilate(int target);
template product_op<fermion_handler>
sum_op<fermion_handler>::number(int target);

// general utility functions

template <typename HandlerTy>
void sum_op<HandlerTy>::dump() const {
  auto str = to_string();
  std::cout << str;
}

template <typename HandlerTy>
std::vector<sum_op<HandlerTy>>
sum_op<HandlerTy>::distribute_terms(std::size_t numChunks) const {
  // Calculate how many terms we can equally divide amongst the chunks
  auto nTermsPerChunk = num_terms() / numChunks;
  auto leftover = num_terms() % numChunks;

  // Slice the given spin_op into subsets for each chunk
  std::vector<sum_op<HandlerTy>> chunks;
  for (auto it = this->term_map.cbegin();
       it != this->term_map.cend();) { // order does not matter here
    sum_op<HandlerTy> chunk;
    // Evenly distribute any leftovers across the early chunks
    for (auto count = nTermsPerChunk + (chunks.size() < leftover ? 1 : 0);
         count > 0; --count, ++it)
      chunk += product_op<HandlerTy>(this->coefficients[it->second],
                                     this->terms[it->second]);
    chunks.push_back(chunk);
  }
  // Not sure if we need this - we might need this when parallelizing a spin_op
  // over QPUs when the system has more processors than we have terms.
  while (chunks.size() < numChunks)
    chunks.push_back(sum_op<HandlerTy>());
  return std::move(chunks);
}

#define INSTANTIATE_SUM_UTILITY_FUNCTIONS(HandlerTy)                           \
  template std::vector<sum_op<HandlerTy>> sum_op<HandlerTy>::distribute_terms( \
      std::size_t numChunks) const;                                            \
  template void sum_op<HandlerTy>::dump() const;

#if !defined(__clang__)
INSTANTIATE_SUM_UTILITY_FUNCTIONS(matrix_handler);
INSTANTIATE_SUM_UTILITY_FUNCTIONS(spin_handler);
INSTANTIATE_SUM_UTILITY_FUNCTIONS(boson_handler);
INSTANTIATE_SUM_UTILITY_FUNCTIONS(fermion_handler);
#endif

// handler specific utility functions

#define HANDLER_SPECIFIC_TEMPLATE_DEFINITION(ConcreteTy)                       \
  template <typename HandlerTy>                                                \
  template <typename T,                                                        \
            std::enable_if_t<std::is_same<T, ConcreteTy>::value &&             \
                                 std::is_same<HandlerTy, T>::value,            \
                             bool>>

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
std::size_t sum_op<HandlerTy>::num_qubits() const {
  return this->degrees(false).size();
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
sum_op<HandlerTy>::sum_op(const std::vector<double> &input_vec) {
  auto it = input_vec.cbegin();
  auto next_int = [&it, &input_vec]() {
    if (it == input_vec.end())
      throw std::runtime_error("incorrect data format - missing entry");
    double intPart;
    if (std::modf(*it, &intPart) != 0.0)
      throw std::runtime_error(
          "Invalid pauli data element, must be integer value.");
    return (int)*it++;
  };
  auto next_double = [&it, &input_vec]() {
    if (it == input_vec.end())
      throw std::runtime_error("incorrect data format - missing entry");
    return *it++;
  };

  auto n_terms = next_int();
  for (std::size_t tidx = 0; tidx < n_terms; ++tidx) {
    auto el_real = next_double();
    auto el_imag = next_double();
    auto prod = product_op<HandlerTy>(std::complex<double>{el_real, el_imag});
    auto nr_ops = next_int();
    for (std::size_t oidx = 0; oidx < nr_ops; ++oidx) {
      auto target = next_int();
      auto val = next_int();
      if (val == 1) // Z
        prod *= sum_op<HandlerTy>::z(target);
      else if (val == 2) // X
        prod *= sum_op<HandlerTy>::x(target);
      else if (val == 3) // Y
        prod *= sum_op<HandlerTy>::y(target);
      else {
        assert(val == 0);
        prod *= sum_op<HandlerTy>::i(target);
      }
    }
    *this += std::move(prod);
  }
  if (it != input_vec.end())
    throw std::runtime_error("incorrect data format  - excess entry");
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
product_op<HandlerTy> sum_op<HandlerTy>::from_word(const std::string &word) {
  auto prod = sum_op<HandlerTy>::identity();
  for (std::size_t i = 0; i < word.length(); i++) {
    auto letter = word[i];
    if (letter == 'Y')
      prod *= sum_op<HandlerTy>::y(i);
    else if (letter == 'X')
      prod *= sum_op<HandlerTy>::x(i);
    else if (letter == 'Z')
      prod *= sum_op<HandlerTy>::z(i);
    else if (letter == 'I')
      prod *= sum_op<HandlerTy>::i(i);
    else
      throw std::runtime_error(
          "Invalid Pauli for spin_op::from_word, must be X, Y, Z, or I.");
  }
  return prod;
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
sum_op<HandlerTy> sum_op<HandlerTy>::random(std::size_t nQubits,
                                            std::size_t nTerms,
                                            unsigned int seed) {
  if (nQubits <= 30) {
    // For the given algorithm below that sets bool=true for 1/2 of the the
    // termData, the maximum number of unique terms is n choose k, where n =
    // 2*nQubits, and k=nQubits. For up to 30 qubits, we can calculate n choose
    // k without overflows (i.e. 60 choose 30 = 118264581564861424) to validate
    // that nTerms is reasonable. For anything larger, the user can't set nTerms
    // large enough to run into actual problems because they would encounter
    // memory limitations long before anything else.
    // Note: use the multiplicative formula to evaluate n-choose-k. The
    // arrangement of multiplications and divisions do not truncate any division
    // remainders.
    std::size_t maxTerms = 1;
    for (std::size_t i = 1; i <= nQubits; i++) {
      maxTerms *= 2 * nQubits + 1 - i;
      maxTerms /= i;
    }
    if (nTerms > maxTerms)
      throw std::runtime_error("Unable to produce " + std::to_string(nTerms) +
                               " unique random terms for " +
                               std::to_string(nQubits) + " qubits");
  }

  auto get_spin_op = [](int target, int kind) {
    if (kind == 1)
      return sum_op<HandlerTy>::z(target);
    if (kind == 2)
      return sum_op<HandlerTy>::x(target);
    if (kind == 3)
      return sum_op<HandlerTy>::y(target);
    return sum_op<HandlerTy>::i(target);
  };

  std::mt19937 gen(seed);
  auto sum = sum_op<HandlerTy>::empty();
  // make sure the number of terms matches the requested number...
  while (sum.terms.size() < nTerms) {
    std::vector<bool> termData(2 * nQubits);
    std::fill_n(termData.begin(), nQubits, true);
    std::shuffle(termData.begin(), termData.end(), gen);
    // ... but allow for duplicates (will be a single term with coefficient !=
    // 1)
    auto prod = sum_op<HandlerTy>::identity();
    for (int qubit_idx = 0; qubit_idx < nQubits; ++qubit_idx) {
      auto kind =
          (termData[qubit_idx << 1] << 1) | termData[(qubit_idx << 1) + 1];
      // keep identities so that we act on the requested number of qubits
      prod *= get_spin_op(qubit_idx, kind);
    }
    sum += std::move(prod);
  }
  return std::move(sum);
}

#if !defined(__clang__)
template std::size_t sum_op<spin_handler>::num_qubits() const;
template sum_op<spin_handler>::sum_op(const std::vector<double> &input_vec);
template product_op<spin_handler>
sum_op<spin_handler>::from_word(const std::string &word);
template sum_op<spin_handler> sum_op<spin_handler>::random(std::size_t nQubits,
                                                           std::size_t nTerms,
                                                           unsigned int seed);
#endif

// utility functions for backwards compatibility

#define SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION                             \
  template <typename HandlerTy>                                                \
  template <typename T,                                                        \
            std::enable_if_t<std::is_same<HandlerTy, spin_handler>::value &&   \
                                 std::is_same<HandlerTy, T>::value,            \
                             bool>>

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
sum_op<HandlerTy>::sum_op(const std::vector<double> &input_vec,
                          std::size_t nQubits) {
  auto n_terms = (int)input_vec.back();
  if (nQubits != (((input_vec.size() - 1) - 2 * n_terms) / n_terms))
    throw std::runtime_error("Invalid data representation for construction "
                             "spin_op. Number of data elements is incorrect.");

  for (std::size_t i = 0; i < input_vec.size() - 1; i += nQubits + 2) {
    auto el_real = input_vec[i + nQubits];
    auto el_imag = input_vec[i + nQubits + 1];
    auto prod = product_op<HandlerTy>(std::complex<double>{el_real, el_imag});
    for (std::size_t j = 0; j < nQubits; j++) {
      double intPart;
      if (std::modf(input_vec[j + i], &intPart) != 0.0)
        throw std::runtime_error(
            "Invalid pauli data element, must be integer value.");

      int val = (int)input_vec[j + i];
      if (val == 1) // X
        prod *= sum_op<HandlerTy>::x(j);
      else if (val == 2) // Z
        prod *= sum_op<HandlerTy>::z(j);
      else if (val == 3) // Y
        prod *= sum_op<HandlerTy>::y(j);
      else { // I
        assert(val == 0);
        prod *= sum_op<HandlerTy>::i(j);
      }
    }
    *this += std::move(prod);
  }
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
sum_op<HandlerTy>::sum_op(const std::vector<std::vector<bool>> &bsf_terms,
                          const std::vector<std::complex<double>> &coeffs) {
  if (bsf_terms.size() != coeffs.size())
    throw std::invalid_argument(
        "size of the coefficient and bsf_terms must match");
  this->coefficients.reserve(coeffs.size());
  this->terms.reserve(bsf_terms.size());

  for (const auto &term : bsf_terms) {
    auto nr_degrees = term.size() / 2;
    std::vector<HandlerTy> ops;
    ops.reserve(nr_degrees);
    for (std::size_t i = 0; i < nr_degrees; ++i) {
      if (term[i] && term[i + nr_degrees])
        ops.push_back(spin_handler(pauli::Y, i));
      else if (term[i])
        ops.push_back(spin_handler(pauli::X, i));
      else if (term[i + nr_degrees])
        ops.push_back(spin_handler(pauli::Z, i));
    }
    product_op<HandlerTy> prod(coeffs[this->terms.size()], std::move(ops));
    this->term_map.insert(
        this->term_map.cend(),
        std::make_pair(prod.get_term_id(), this->terms.size()));
    this->terms.push_back(std::move(prod.operators));
    this->coefficients.push_back(std::move(prod.coefficient));
  }
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
std::vector<double> sum_op<HandlerTy>::getDataRepresentation() const {
  // This function prints a data representing the operator sum
  std::vector<double> dataVec;
  // dataVec.reserve(n_targets * padded.terms.size() + 2 * padded.terms.size() +
  // 1);
  dataVec.push_back(this->terms.size());
  for (std::size_t i = 0; i < this->terms.size(); ++i) {
    auto coeff = this->coefficients[i].evaluate();
    dataVec.push_back(coeff.real());
    dataVec.push_back(coeff.imag());
    dataVec.push_back(this->terms[i].size());
    for (std::size_t j = 0; j < this->terms[i].size(); ++j) {
      auto op = this->terms[i][j];
      dataVec.push_back(op.degrees()[0]);
      auto pauli = op.as_pauli();
      if (pauli == pauli::Z)
        dataVec.push_back(1.);
      else if (pauli == pauli::X)
        dataVec.push_back(2.);
      else if (pauli == pauli::Y)
        dataVec.push_back(3.);
      else
        dataVec.push_back(0.);
    }
  }
  return dataVec;
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
std::pair<std::vector<std::vector<bool>>, std::vector<std::complex<double>>>
sum_op<HandlerTy>::get_raw_data() const {
  std::unordered_map<int, int> dims;
  auto degrees = this->degrees(
      false); // degrees in canonical order to match the evaluation
  auto evaluated = this->evaluate(
      operator_arithmetics<operator_handler::canonical_evaluation>(
          dims, {})); // fails if we have parameters

  std::size_t term_size = 0;
  if (degrees.size() != 0)
    term_size = operator_handler::canonical_order(0, 1) ? degrees.back() + 1
                                                        : degrees[0] + 1;

  std::vector<std::complex<double>> coeffs;
  std::vector<std::vector<bool>> bsf_terms;
  coeffs.reserve(evaluated.terms.size());
  bsf_terms.reserve(evaluated.terms.size());

  // For compatiblity with existing code, the binary symplectic representation
  // needs to be from smallest to largest degree, and it necessarily must
  // include all consecutive degrees starting from 0 (even if the operator
  // doesn't act on them).
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
    coeffs.push_back(std::move(term.first));
  }

  // always little endian order by definition of the bsf
  return std::make_pair(std::move(bsf_terms), std::move(coeffs));
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
std::string sum_op<HandlerTy>::to_string(bool printCoeffs) const {
  // This function prints a string representing the operator sum that
  // includes the full representation for any degree in [0, max_degree),
  // padding identities if necessary (opposed to pauli_word).
  std::unordered_map<int, int> dims;
  auto degrees = this->degrees(
      false); // degrees in canonical order to match the evaluation
  auto evaluated = this->evaluate(
      operator_arithmetics<operator_handler::canonical_evaluation>(dims, {}));
  auto le_order = std::less<int>();
  auto get_le_index = [&degrees, &le_order](std::size_t idx) {
    // For compatibility with existing code, the ordering for the term string
    // always needs to be from smallest to largest degree, and it necessarily
    // must include all consecutive degrees starting from 0 (even if the
    // operator doesn't act on them).
    return (operator_handler::canonical_order(1, 0) == le_order(1, 0))
               ? idx
               : degrees.size() - 1 - idx;
  };

  std::stringstream ss;
  auto first = true;
  for (auto &&term : evaluated.terms) {
    if (first)
      first = false;
    else
      ss << std::endl;
    if (printCoeffs) {
      auto coeff = term.first;
      ss << "[" << coeff.real() << (coeff.imag() < 0.0 ? "-" : "+")
         << std::fabs(coeff.imag()) << "j] ";
    }

    if (degrees.size() > 0) {
      auto max_target =
          operator_handler::canonical_order(0, 1) ? degrees.back() : degrees[0];
      std::string term_str(max_target + 1, 'I');
      for (std::size_t i = 0; i < degrees.size(); ++i)
        term_str[degrees[i]] = term.second[get_le_index(i)];
      ss << term_str;
    }
  }
  return ss.str();
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
void sum_op<HandlerTy>::for_each_term(
    std::function<void(sum_op<HandlerTy> &)> &&functor) const {
  for (auto &&prod : *this) {
    sum_op<HandlerTy> as_sum(std::move(prod));
    functor(as_sum);
  }
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
void sum_op<HandlerTy>::for_each_pauli(
    std::function<void(pauli, std::size_t)> &&functor) const {
  if (this->terms.size() == 0)
    return;
  if (this->terms.size() != 1)
    throw std::runtime_error("more than one term in for_each_pauli");
  for (const auto &op : this->terms[0])
    functor(op.as_pauli(), op.degrees()[0]);
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
bool sum_op<HandlerTy>::is_identity() const {
  for (const auto &term : this->terms) {
    for (const auto &op : term) {
      if (op != HandlerTy(op.degrees()[0]))
        return false;
    }
  }
  return true;
}

#if !defined(__clang__)
template sum_op<spin_handler>::sum_op(const std::vector<double> &input_vec,
                                      std::size_t nQubits);
template sum_op<spin_handler>::sum_op(
    const std::vector<std::vector<bool>> &bsf_terms,
    const std::vector<std::complex<double>> &coeffs);
template std::vector<double>
sum_op<spin_handler>::getDataRepresentation() const;
template std::pair<std::vector<std::vector<bool>>,
                   std::vector<std::complex<double>>>
sum_op<spin_handler>::get_raw_data() const;
template std::string sum_op<spin_handler>::to_string(bool printCoeffs) const;
template void sum_op<spin_handler>::for_each_term(
    std::function<void(sum_op<spin_handler> &)> &&functor) const;
template void sum_op<spin_handler>::for_each_pauli(
    std::function<void(pauli, std::size_t)> &&functor) const;
template bool sum_op<spin_handler>::is_identity() const;
#endif

#if defined(CUDAQ_INSTANTIATE_TEMPLATES)
template class sum_op<matrix_handler>;
template class sum_op<spin_handler>;
template class sum_op<boson_handler>;
template class sum_op<fermion_handler>;
#endif

} // namespace cudaq