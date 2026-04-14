/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenSparse.h"
#include "cudaq/operators.h"
#include "evaluation.h"
#include "helpers.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>

namespace cudaq {

#define PROPERTY_SPECIFIC_TEMPLATE_DEFINITION(HandlerTy, property)             \
  template <typename T,                                                        \
            std::enable_if_t<std::is_same<HandlerTy, T>::value && property,    \
                             std::true_type>>

#define PROPERTY_AGNOSTIC_TEMPLATE_DEFINITION(HandlerTy, property)             \
  template <typename T,                                                        \
            std::enable_if_t<std::is_same<HandlerTy, T>::value && !property,   \
                             std::false_type>>

// private methods

/// expects is_default to be false
template <typename HandlerTy>
void sum_op<HandlerTy>::insert(const product_op<HandlerTy> &other) {
  assert(!this->is_default);
  auto [it, inserted] =
      this->term_map.try_emplace(other.get_term_id(), this->terms.size());
  if (inserted) {
    this->coefficients.push_back(other.coefficient);
    this->terms.push_back(other.operators);
  } else {
    this->coefficients[it->second] += other.coefficient;
  }
}

/// expects is_default to be false
template <typename HandlerTy>
void sum_op<HandlerTy>::insert(product_op<HandlerTy> &&other) {
  assert(!this->is_default);
  auto [it, inserted] =
      this->term_map.try_emplace(other.get_term_id(), this->terms.size());
  if (inserted) {
    this->coefficients.push_back(std::move(other.coefficient));
    this->terms.push_back(std::move(other.operators));
  } else {
    this->coefficients[it->second] += other.coefficient;
  }
}

template <typename HandlerTy>
void sum_op<HandlerTy>::aggregate_terms() {}

/// expects is_default to be false
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
sum_op<HandlerTy>::transform(operator_arithmetics<EvalTy> arithmetics) const {
  if (terms.size() == 0)
    return EvalTy();

  // NOTE: It is important that we evaluate the terms in a specific order,
  // otherwise the evaluation is not consistent with other methods.
  // The specific order does not matter, as long as all methods use the same
  // term order.
  auto it = this->begin();
  auto end = this->end();
  if (arithmetics.pad_sum_terms) {
    // Canonicalizing a term adds a tensor product with the identity for degrees
    // that an operator doesn't act on. Needed e.g. to make sure all matrices
    // are of the same size before summing them up.
    std::set<std::size_t> degrees;
    for (const auto &term : this->terms)
      for (const auto &op : term) {
        auto op_degrees = op.degrees();
        degrees.insert(op_degrees.cbegin(), op_degrees.cend());
      }
    product_op<HandlerTy> padded_term = it->canonicalize(degrees);
    EvalTy sum = padded_term.template transform<EvalTy>(arithmetics);
    while (++it != end) {
      padded_term = it->canonicalize(degrees);
      EvalTy term_eval = padded_term.template transform<EvalTy>(arithmetics);
      sum = arithmetics.add(std::move(sum), std::move(term_eval));
    }
    return sum;
  } else {
    EvalTy sum = it->template transform<EvalTy>(arithmetics);
    while (++it != end) {
      EvalTy term_eval = it->template transform<EvalTy>(arithmetics);
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
  template EvalTy sum_op<HandlerTy>::transform(                                \
      operator_arithmetics<EvalTy> arithmetics) const;

#if !defined(__clang__)
INSTANTIATE_SUM_EVALUATE_METHODS(matrix_handler,
                                 operator_handler::matrix_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(spin_handler,
                                 operator_handler::canonical_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(boson_handler,
                                 operator_handler::canonical_evaluation);
INSTANTIATE_SUM_EVALUATE_METHODS(fermion_handler,
                                 operator_handler::canonical_evaluation);
#endif

// read-only properties

template <typename HandlerTy>
std::vector<std::size_t> sum_op<HandlerTy>::degrees() const {
  std::set<std::size_t> unsorted_degrees;
  for (const std::vector<HandlerTy> &term : this->terms) {
    for (const HandlerTy &op : term) {
      auto op_degrees = op.degrees();
      unsorted_degrees.insert(op_degrees.cbegin(), op_degrees.cend());
    }
  }
  auto degrees = std::vector<std::size_t>(unsorted_degrees.cbegin(),
                                          unsorted_degrees.cend());
  std::sort(degrees.begin(), degrees.end(), operator_handler::canonical_order);
  return degrees;
}

template <typename HandlerTy>
std::size_t sum_op<HandlerTy>::min_degree() const {
  auto degrees = this->degrees();
  if (degrees.size() == 0)
    throw std::runtime_error("operator is not acting on any degrees");
  return operator_handler::canonical_order(0, 1) ? degrees[0] : degrees.back();
}

template <typename HandlerTy>
std::size_t sum_op<HandlerTy>::max_degree() const {
  auto degrees = this->degrees();
  if (degrees.size() == 0)
    throw std::runtime_error("operator is not acting on any degrees");
  return operator_handler::canonical_order(0, 1) ? degrees.back() : degrees[0];
}

template <typename HandlerTy>
std::size_t sum_op<HandlerTy>::num_terms() const {
  return this->terms.size();
}

template <typename HandlerTy>
std::unordered_map<std::string, std::string>
sum_op<HandlerTy>::get_parameter_descriptions() const {
  std::unordered_map<std::string, std::string> descriptions;
  for (const auto &coeff : this->coefficients)
    for (const auto &entry : coeff.get_parameter_descriptions()) {
      // don't overwrite an existing entry with an empty description,
      // but generally just overwrite descriptions otherwise
      if (!entry.second.empty())
        descriptions.insert_or_assign(entry.first, entry.second);
      else if (descriptions.find(entry.first) == descriptions.end())
        descriptions.insert(descriptions.end(), entry);
    }
  return descriptions;
}

template <>
std::unordered_map<std::string, std::string>
sum_op<matrix_handler>::get_parameter_descriptions() const {
  std::unordered_map<std::string, std::string> descriptions;
  auto update_descriptions =
      [&descriptions](const std::pair<std::string, std::string> &entry) {
        // don't overwrite an existing entry with an empty description,
        // but generally just overwrite descriptions otherwise
        if (!entry.second.empty())
          descriptions.insert_or_assign(entry.first, entry.second);
        else if (descriptions.find(entry.first) == descriptions.end())
          descriptions.insert(descriptions.end(), entry);
      };
  for (const auto &coeff : this->coefficients)
    for (const auto &entry : coeff.get_parameter_descriptions())
      update_descriptions(entry);
  for (const auto &term : this->terms)
    for (const auto &op : term)
      for (const auto &entry : op.get_parameter_descriptions())
        update_descriptions(entry);
  return descriptions;
}

#define INSTANTIATE_SUM_PROPERTIES(HandlerTy)                                  \
                                                                               \
  template std::vector<std::size_t> sum_op<HandlerTy>::degrees() const;        \
                                                                               \
  template std::size_t sum_op<HandlerTy>::min_degree() const;                  \
                                                                               \
  template std::size_t sum_op<HandlerTy>::max_degree() const;                  \
                                                                               \
  template std::size_t sum_op<HandlerTy>::num_terms() const;                   \
                                                                               \
  template std::unordered_map<std::string, std::string>                        \
  sum_op<HandlerTy>::get_parameter_descriptions() const;

#if !defined(__clang__)
INSTANTIATE_SUM_PROPERTIES(matrix_handler);
INSTANTIATE_SUM_PROPERTIES(spin_handler);
INSTANTIATE_SUM_PROPERTIES(boson_handler);
INSTANTIATE_SUM_PROPERTIES(fermion_handler);
#endif

// constructors

template <typename HandlerTy>
sum_op<HandlerTy>::sum_op(std::size_t size) : is_default(true) {
  this->coefficients.reserve(size);
  this->term_map.reserve(size);
  this->terms.reserve(size);
}

template <typename HandlerTy>
sum_op<HandlerTy>::sum_op(const product_op<HandlerTy> &prod)
    : is_default(false) {
  this->insert(prod);
}

template <typename HandlerTy>
template <
    typename... Args,
    std::enable_if_t<
        std::conjunction<std::is_same<product_op<HandlerTy>, Args>...>::value &&
            sizeof...(Args),
        bool>>
sum_op<HandlerTy>::sum_op(Args &&...args) : is_default(false) {
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
    : is_default(other.is_default), coefficients(other.coefficients) {
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
    : is_default(other.is_default), coefficients(other.coefficients) {
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
sum_op<HandlerTy>::sum_op(const sum_op<HandlerTy> &other, bool is_default,
                          std::size_t size)
    : is_default(is_default && other.is_default) {
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
sum_op<HandlerTy>::sum_op(sum_op<HandlerTy> &&other, bool is_default,
                          std::size_t size)
    : is_default(is_default && other.is_default),
      coefficients(std::move(other.coefficients)),
      term_map(std::move(other.term_map)), terms(std::move(other.terms)) {
  if (size > 0) {
    this->coefficients.reserve(size);
    this->term_map.reserve(size);
    this->terms.reserve(size);
  }
}

#define INSTANTIATE_SUM_CONSTRUCTORS(HandlerTy)                                \
                                                                               \
  template sum_op<HandlerTy>::sum_op(bool is_default);                         \
                                                                               \
  template sum_op<HandlerTy>::sum_op(std::size_t size);                        \
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
                                     bool is_default, std::size_t size);       \
                                                                               \
  template sum_op<HandlerTy>::sum_op(sum_op<HandlerTy> &&other,                \
                                     bool is_default, std::size_t size);

// Note:
// These are the private constructors needed by friend classes and functions
// of sum_op. For clang, (only!) these need to be instantiated explicitly
// to be available to those.
#define INSTANTIATE_SUM_PRIVATE_FRIEND_CONSTRUCTORS(HandlerTy)                 \
                                                                               \
  template sum_op<HandlerTy>::sum_op(product_op<HandlerTy> &&item2);           \
                                                                               \
  template sum_op<HandlerTy>::sum_op(product_op<HandlerTy> &&item1,            \
                                     product_op<HandlerTy> &&item2);           \
                                                                               \
  template sum_op<HandlerTy>::sum_op(product_op<HandlerTy> &&item1,            \
                                     product_op<HandlerTy> &&item2,            \
                                     product_op<HandlerTy> &&item3);

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
  this->is_default = false;
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
  this->is_default = false;
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

#define INSTANTIATE_SUM_ASSIGNMENTS(HandlerTy)                                 \
                                                                               \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator=(                    \
      product_op<HandlerTy> &&other);                                          \
                                                                               \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::operator=(                    \
      const product_op<HandlerTy> &other);

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
complex_matrix sum_op<HandlerTy>::to_matrix(
    std::unordered_map<std::size_t, std::int64_t> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool invert_order) const {
  auto evaluated = this->transform(
      operator_arithmetics<operator_handler::canonical_evaluation>(dimensions,
                                                                   parameters));
  if (evaluated.terms.size() == 0)
    return cudaq::complex_matrix(0, 0);

  auto matrix = HandlerTy::to_matrix(
      evaluated.terms[0].encoding, evaluated.terms[0].relevant_dimensions,
      evaluated.terms[0].coefficient, invert_order);
  for (auto i = 1; i < terms.size(); ++i)
    matrix += HandlerTy::to_matrix(
        evaluated.terms[i].encoding, evaluated.terms[i].relevant_dimensions,
        evaluated.terms[i].coefficient, invert_order);
  return matrix;
}

template <>
complex_matrix sum_op<matrix_handler>::to_matrix(
    std::unordered_map<std::size_t, int64_t> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool invert_order) const {
  auto evaluated =
      this->transform(operator_arithmetics<operator_handler::matrix_evaluation>(
          dimensions, parameters));
  if (invert_order) {
    auto reverse_degrees = evaluated.degrees;
    std::reverse(reverse_degrees.begin(), reverse_degrees.end());
    auto permutation = cudaq::detail::compute_permutation(
        evaluated.degrees, reverse_degrees, dimensions);
    cudaq::detail::permute_matrix(evaluated.matrix, permutation);
  }
  return std::move(evaluated.matrix);
}

#define INSTANTIATE_SUM_EVALUATIONS(HandlerTy)                                 \
                                                                               \
  template complex_matrix sum_op<HandlerTy>::to_matrix(                        \
      std::unordered_map<std::size_t, std::int64_t> dimensions,                \
      const std::unordered_map<std::string, std::complex<double>> &params,     \
      bool invert_order) const;

#if !defined(__clang__)
INSTANTIATE_SUM_EVALUATIONS(matrix_handler);
INSTANTIATE_SUM_EVALUATIONS(spin_handler);
INSTANTIATE_SUM_EVALUATIONS(boson_handler);
INSTANTIATE_SUM_EVALUATIONS(fermion_handler);
#endif

// comparisons

template <typename HandlerTy>
bool sum_op<HandlerTy>::operator==(const sum_op<HandlerTy> &other) const {
  if (this->terms.size() != other.terms.size() ||
      this->is_default != other.is_default)
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
  if (this->is_default)
    throw std::runtime_error(
        "cannot apply unary operator on uninitialized sum_op");
  sum_op<HandlerTy> sum(false);
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map = this->term_map;
  sum.terms = this->terms;
  for (auto &coeff : this->coefficients)
    sum.coefficients.push_back(-1. * coeff);
  return sum;
}

template <typename HandlerTy>
sum_op<HandlerTy> sum_op<HandlerTy>::operator-() && {
  if (this->is_default)
    throw std::runtime_error(
        "cannot apply unary operator on uninitialized sum_op");
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

template <typename HandlerTy>
sum_op<HandlerTy>
sum_op<HandlerTy>::operator*(const scalar_operator &other) const & {
  if (this->is_default)
    // scalars are just a special product operator
    return product_op<HandlerTy>(other);
  sum_op<HandlerTy> sum(false);
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map = this->term_map;
  sum.terms = this->terms;
  for (const auto &coeff : this->coefficients)
    sum.coefficients.push_back(coeff * other);
  return sum;
}

template <typename HandlerTy>
sum_op<HandlerTy>
sum_op<HandlerTy>::operator*(const scalar_operator &other) && {
  if (this->is_default)
    // scalars are just a special product operator
    return product_op<HandlerTy>(other);
  for (auto &coeff : this->coefficients)
    coeff *= other;
  return std::move(*this);
}

template <typename HandlerTy>
sum_op<HandlerTy>
sum_op<HandlerTy>::operator/(const scalar_operator &other) const & {
  if (this->is_default)
    throw std::runtime_error("cannot divide uninitialized sum_op by scalar");
  sum_op<HandlerTy> sum(false);
  sum.coefficients.reserve(this->coefficients.size());
  sum.term_map = this->term_map;
  sum.terms = this->terms;
  for (const auto &coeff : this->coefficients)
    sum.coefficients.push_back(coeff / other);
  return sum;
}

template <typename HandlerTy>
sum_op<HandlerTy>
sum_op<HandlerTy>::operator/(const scalar_operator &other) && {
  if (this->is_default)
    throw std::runtime_error("cannot divide uninitialized sum_op by scalar");
  for (auto &coeff : this->coefficients)
    coeff /= other;
  return std::move(*this);
}

#define SUM_ADDITION_SCALAR(op)                                                \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const scalar_operator &other) const & {                                  \
    sum_op<HandlerTy> sum(*this, false, this->terms.size() + 1);               \
    sum.insert(product_op<HandlerTy>(op other));                               \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(scalar_operator &&other)    \
      const & {                                                                \
    sum_op<HandlerTy> sum(*this, false, this->terms.size() + 1);               \
    sum.insert(product_op<HandlerTy>(op std::move(other)));                    \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const scalar_operator &other) && {                                       \
    this->is_default = false;                                                  \
    this->insert(product_op<HandlerTy>(op other));                             \
    return std::move(*this);                                                   \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      scalar_operator &&other) && {                                            \
    this->is_default = false;                                                  \
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
  if (this->is_default)
    return other;
  sum_op<HandlerTy> sum(false); // the entire sum needs to be rebuilt
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
  return sum;
}

#define SUM_ADDITION_PRODUCT(op)                                               \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const product_op<HandlerTy> &other) const & {                            \
    sum_op<HandlerTy> sum(*this, false, this->terms.size() + 1);               \
    sum.insert(op other);                                                      \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const product_op<HandlerTy> &other) && {                                 \
    this->is_default = false;                                                  \
    this->insert(op other);                                                    \
    return std::move(*this);                                                   \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      product_op<HandlerTy> &&other) const & {                                 \
    sum_op<HandlerTy> sum(*this, false, this->terms.size() + 1);               \
    sum.insert(op std::move(other));                                           \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      product_op<HandlerTy> &&other) && {                                      \
    this->is_default = false;                                                  \
    this->insert(op std::move(other));                                         \
    return std::move(*this);                                                   \
  }

SUM_ADDITION_PRODUCT(+)
SUM_ADDITION_PRODUCT(-)

template <typename HandlerTy>
sum_op<HandlerTy>
sum_op<HandlerTy>::operator*(const sum_op<HandlerTy> &other) const {
  if (other.is_default)
    return *this;
  if (this->is_default)
    return other;

  sum_op<HandlerTy> sum(false); // the entire sum needs to be rebuilt
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
  return sum;
}

#define SUM_ADDITION_SUM(op)                                                   \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const sum_op<HandlerTy> &other) const & {                                \
    sum_op<HandlerTy> sum(*this, this->is_default &&other.is_default,          \
                          this->terms.size() + other.terms.size());            \
    for (auto i = 0; i < other.terms.size(); ++i) {                            \
      product_op<HandlerTy> prod(op other.coefficients[i], other.terms[i]);    \
      sum.insert(std::move(prod));                                             \
    }                                                                          \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      const sum_op<HandlerTy> &other) && {                                     \
    /* in case other is not default but does not have terms: */                \
    this->is_default = this->is_default && other.is_default;                   \
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
    sum_op<HandlerTy> sum(*this, this->is_default &&other.is_default,          \
                          this->terms.size() + other.terms.size());            \
    for (auto i = 0; i < other.terms.size(); ++i) {                            \
      product_op<HandlerTy> prod(op std::move(other.coefficients[i]),          \
                                 std::move(other.terms[i]));                   \
      sum.insert(std::move(prod));                                             \
    }                                                                          \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> sum_op<HandlerTy>::operator op(                            \
      sum_op<HandlerTy> &&other) && {                                          \
    /* in case other is not default but does not have terms: */                \
    this->is_default = this->is_default && other.is_default;                   \
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

template <typename HandlerTy>
sum_op<HandlerTy> &sum_op<HandlerTy>::operator*=(const scalar_operator &other) {
  if (this->is_default)
    // scalars are just a special product operator
    *this = product_op<HandlerTy>(other);
  else
    for (auto &coeff : this->coefficients)
      coeff *= other;
  return *this;
}

template <typename HandlerTy>
sum_op<HandlerTy> &sum_op<HandlerTy>::operator/=(const scalar_operator &other) {
  if (this->is_default)
    throw std::runtime_error("cannot divide uninitialized sum_op by scalar");
  for (auto &coeff : this->coefficients)
    coeff /= other;
  return *this;
}

#define SUM_ADDITION_SCALAR_ASSIGNMENT(op)                                     \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> &sum_op<HandlerTy>::operator op##=(                        \
      const scalar_operator &other) {                                          \
    this->is_default = false;                                                  \
    this->insert(product_op<HandlerTy>(op other));                             \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> &sum_op<HandlerTy>::operator op##=(                        \
      scalar_operator &&other) {                                               \
    this->is_default = false;                                                  \
    this->insert(product_op<HandlerTy>(op std::move(other)));                  \
    return *this;                                                              \
  }

SUM_ADDITION_SCALAR_ASSIGNMENT(+);
SUM_ADDITION_SCALAR_ASSIGNMENT(-);

template <typename HandlerTy>
sum_op<HandlerTy> &
sum_op<HandlerTy>::operator*=(const product_op<HandlerTy> &other) {
  if (this->is_default) {
    *this = sum_op<HandlerTy>(other);
    return *this;
  }
  sum_op<HandlerTy> sum(false);
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
    this->is_default = false;                                                  \
    this->insert(op other);                                                    \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> &sum_op<HandlerTy>::operator op##=(                        \
      product_op<HandlerTy> &&other) {                                         \
    this->is_default = false;                                                  \
    this->insert(op std::move(other));                                         \
    return *this;                                                              \
  }

SUM_ADDITION_PRODUCT_ASSIGNMENT(+)
SUM_ADDITION_PRODUCT_ASSIGNMENT(-)

template <typename HandlerTy>
sum_op<HandlerTy> &
sum_op<HandlerTy>::operator*=(const sum_op<HandlerTy> &other) {
  if (other.is_default)
    return *this;
  if (this->is_default) {
    *this = other;
    return *this;
  }

  sum_op<HandlerTy> sum(false); // the entire sum needs to be rebuilt
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
    /* in case other is not default but does not have terms: */                \
    this->is_default = this->is_default && other.is_default;                   \
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
    /* in case other is not default but does not have terms: */                \
    this->is_default = this->is_default && other.is_default;                   \
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
  if (self.is_default)
    // scalars are just a special product operator
    return product_op<HandlerTy>(other);
  sum_op<HandlerTy> sum(false);
  sum.coefficients.reserve(self.coefficients.size());
  sum.terms = self.terms;
  sum.term_map = self.term_map;
  for (const auto &coeff : self.coefficients)
    sum.coefficients.push_back(coeff * other);
  return sum;
}

template <typename HandlerTy>
sum_op<HandlerTy> operator*(const scalar_operator &other,
                            sum_op<HandlerTy> &&self) {
  if (self.is_default)
    // scalars are just a special product operator
    return product_op<HandlerTy>(other);
  for (auto &&coeff : self.coefficients)
    coeff *= other;
  return std::move(self);
}

#define SUM_ADDITION_SCALAR_REVERSE(op)                                        \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(const scalar_operator &other,                  \
                                const sum_op<HandlerTy> &self) {               \
    if (self.is_default)                                                       \
      return product_op<HandlerTy>(other);                                     \
    sum_op<HandlerTy> sum(op self);                                            \
    sum.is_default = false;                                                    \
    sum.insert(product_op<HandlerTy>(other));                                  \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(scalar_operator &&other,                       \
                                const sum_op<HandlerTy> &self) {               \
    if (self.is_default)                                                       \
      return product_op<HandlerTy>(other);                                     \
    sum_op<HandlerTy> sum(op self);                                            \
    sum.is_default = false;                                                    \
    sum.insert(product_op<HandlerTy>(std::move(other)));                       \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(const scalar_operator &other,                  \
                                sum_op<HandlerTy> &&self) {                    \
    for (auto &&coeff : self.coefficients)                                     \
      coeff = std::move(op coeff);                                             \
    self.is_default = false;                                                   \
    self.insert(product_op<HandlerTy>(other));                                 \
    return std::move(self);                                                    \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(scalar_operator &&other,                       \
                                sum_op<HandlerTy> &&self) {                    \
    for (auto &&coeff : self.coefficients)                                     \
      coeff = std::move(op coeff);                                             \
    self.is_default = false;                                                   \
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
  // The empty sum is explicitly intended to be the 0
  // element of the algebra, i.e. it is the neutral
  // element for addition, whereas multiplication with an
  // empty sum must always result in an emtpy sum.
  return sum_op<HandlerTy>(false);
}

template <typename HandlerTy>
product_op<HandlerTy> sum_op<HandlerTy>::identity() {
  return product_op<HandlerTy>(1.0);
}

template <typename HandlerTy>
product_op<HandlerTy> sum_op<HandlerTy>::identity(std::size_t target) {
  static_assert(
      std::is_constructible_v<HandlerTy, std::size_t>,
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
sum_op<matrix_handler>::identity(std::size_t target);
template product_op<spin_handler>
sum_op<spin_handler>::identity(std::size_t target);
template product_op<boson_handler>
sum_op<boson_handler>::identity(std::size_t target);
template product_op<fermion_handler>
sum_op<fermion_handler>::identity(std::size_t target);
#endif

// handler specific operators

#define HANDLER_SPECIFIC_TEMPLATE_DEFINITION(ConcreteTy)                       \
  template <typename HandlerTy>                                                \
  template <typename T,                                                        \
            std::enable_if_t<std::is_same<T, ConcreteTy>::value &&             \
                                 std::is_same<HandlerTy, T>::value,            \
                             bool>>

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::number(std::size_t target) {
  return cudaq::operators::number(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::parity(std::size_t target) {
  return cudaq::operators::parity(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::position(std::size_t target) {
  return cudaq::operators::position(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::momentum(std::size_t target) {
  return cudaq::operators::momentum(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::squeeze(std::size_t target) {
  return cudaq::operators::squeeze(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(matrix_handler)
product_op<T> sum_op<HandlerTy>::displace(std::size_t target) {
  return cudaq::operators::displace(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
product_op<T> sum_op<HandlerTy>::i(std::size_t target) {
  return cudaq::spin::i(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
product_op<T> sum_op<HandlerTy>::x(std::size_t target) {
  return cudaq::spin::x(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
product_op<T> sum_op<HandlerTy>::y(std::size_t target) {
  return cudaq::spin::y(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
product_op<T> sum_op<HandlerTy>::z(std::size_t target) {
  return cudaq::spin::z(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
sum_op<T> sum_op<HandlerTy>::plus(std::size_t target) {
  return cudaq::spin::plus(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
sum_op<T> sum_op<HandlerTy>::minus(std::size_t target) {
  return cudaq::spin::minus(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(boson_handler)
product_op<T> sum_op<HandlerTy>::create(std::size_t target) {
  return cudaq::boson::create(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(boson_handler)
product_op<T> sum_op<HandlerTy>::annihilate(std::size_t target) {
  return cudaq::boson::annihilate(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(boson_handler)
product_op<T> sum_op<HandlerTy>::number(std::size_t target) {
  return cudaq::boson::number(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(boson_handler)
sum_op<T> sum_op<HandlerTy>::position(std::size_t target) {
  return cudaq::boson::position(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(boson_handler)
sum_op<T> sum_op<HandlerTy>::momentum(std::size_t target) {
  return cudaq::boson::momentum(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(fermion_handler)
product_op<T> sum_op<HandlerTy>::create(std::size_t target) {
  return cudaq::fermion::create(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(fermion_handler)
product_op<T> sum_op<HandlerTy>::annihilate(std::size_t target) {
  return cudaq::fermion::annihilate(target);
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(fermion_handler)
product_op<T> sum_op<HandlerTy>::number(std::size_t target) {
  return cudaq::fermion::number(target);
}

template product_op<matrix_handler>
sum_op<matrix_handler>::number(std::size_t target);
template product_op<matrix_handler>
sum_op<matrix_handler>::parity(std::size_t target);
template product_op<matrix_handler>
sum_op<matrix_handler>::position(std::size_t target);
template product_op<matrix_handler>
sum_op<matrix_handler>::momentum(std::size_t target);
template product_op<matrix_handler>
sum_op<matrix_handler>::squeeze(std::size_t target);
template product_op<matrix_handler>
sum_op<matrix_handler>::displace(std::size_t target);

template product_op<spin_handler> sum_op<spin_handler>::i(std::size_t target);
template product_op<spin_handler> sum_op<spin_handler>::x(std::size_t target);
template product_op<spin_handler> sum_op<spin_handler>::y(std::size_t target);
template product_op<spin_handler> sum_op<spin_handler>::z(std::size_t target);
template sum_op<spin_handler> sum_op<spin_handler>::plus(std::size_t target);
template sum_op<spin_handler> sum_op<spin_handler>::minus(std::size_t target);

template product_op<boson_handler>
sum_op<boson_handler>::create(std::size_t target);
template product_op<boson_handler>
sum_op<boson_handler>::annihilate(std::size_t target);
template product_op<boson_handler>
sum_op<boson_handler>::number(std::size_t target);
template sum_op<boson_handler>
sum_op<boson_handler>::position(std::size_t target);
template sum_op<boson_handler>
sum_op<boson_handler>::momentum(std::size_t target);

template product_op<fermion_handler>
sum_op<fermion_handler>::create(std::size_t target);
template product_op<fermion_handler>
sum_op<fermion_handler>::annihilate(std::size_t target);
template product_op<fermion_handler>
sum_op<fermion_handler>::number(std::size_t target);

// general utility functions

template <typename HandlerTy>
std::string sum_op<HandlerTy>::to_string() const {
  if (this->terms.size() == 0)
    return "";
  auto it = this->begin();
  std::string str = it->to_string();
  while (++it != this->end())
    str += " + " + it->to_string();
  return str;
}

template <typename HandlerTy>
void sum_op<HandlerTy>::dump() const {
  auto str = to_string();
  std::cout << str << std::endl;
}

template <typename HandlerTy>
sum_op<HandlerTy> &sum_op<HandlerTy>::trim(
    double tol,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  sum_op<HandlerTy> trimmed(false);
  trimmed.term_map.reserve(this->terms.size());
  trimmed.terms.reserve(this->terms.size());
  trimmed.coefficients.reserve(this->coefficients.size());
  for (const auto &prod : *this)
    if (std::abs(prod.evaluate_coefficient(parameters)) > tol)
      trimmed.insert(std::move(prod));
  *this = trimmed;
  return *this;
}

template <typename HandlerTy>
sum_op<HandlerTy> &sum_op<HandlerTy>::canonicalize() {
  // If we make any updates, we it's best to completely rebuild the operator,
  // since this may lead to the combination of terms and therefore
  // change the structure/term_map of the operator.
  *this = canonicalize(std::move(*this));
  return *this;
}

template <typename HandlerTy>
sum_op<HandlerTy>
sum_op<HandlerTy>::canonicalize(const sum_op<HandlerTy> &orig) {
  sum_op<HandlerTy> canonicalized(false);
  for (auto &&prod : orig)
    canonicalized.insert(prod.canonicalize());
  return canonicalized;
}

template <typename HandlerTy>
sum_op<HandlerTy> &
sum_op<HandlerTy>::canonicalize(const std::set<std::size_t> &degrees) {
  // If we make any updates, we it's best to completely rebuild the operator,
  // since this may lead to the combination of terms and therefore
  // change the structure/term_map of the operator.
  *this = canonicalize(std::move(*this), degrees);
  return *this;
}

template <typename HandlerTy>
sum_op<HandlerTy>
sum_op<HandlerTy>::canonicalize(const sum_op<HandlerTy> &orig,
                                const std::set<std::size_t> &degrees) {
  std::set<std::size_t> all_degrees;
  if (degrees.size() == 0) {
    for (const auto &term : orig.terms)
      for (const auto &op : term) {
        auto op_degrees = op.degrees();
        all_degrees.insert(op_degrees.cbegin(), op_degrees.cend());
      }
  }
  sum_op<HandlerTy> canonicalized(false);
  for (auto &&prod : orig)
    canonicalized.insert(
        prod.canonicalize(degrees.size() == 0 ? all_degrees : degrees));
  return canonicalized;
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
    sum_op<HandlerTy> chunk(false);
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
    // needs to be empty (is_default = false), since it should zero out any
    // multiplication
    chunks.push_back(sum_op<HandlerTy>(false));
  return chunks;
}

#define INSTANTIATE_SUM_UTILITY_FUNCTIONS(HandlerTy)                           \
                                                                               \
  template std::string sum_op<HandlerTy>::to_string() const;                   \
                                                                               \
  template void sum_op<HandlerTy>::dump() const;                               \
                                                                               \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::canonicalize();               \
                                                                               \
  template sum_op<HandlerTy> sum_op<HandlerTy>::canonicalize(                  \
      const sum_op<HandlerTy> &orig);                                          \
                                                                               \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::canonicalize(                 \
      const std::set<std::size_t> &degrees);                                   \
                                                                               \
  template sum_op<HandlerTy> sum_op<HandlerTy>::canonicalize(                  \
      const sum_op<HandlerTy> &orig, const std::set<std::size_t> &degrees);    \
                                                                               \
  template sum_op<HandlerTy> &sum_op<HandlerTy>::trim(                         \
      double tol, const std::unordered_map<std::string, std::complex<double>>  \
                      &parameters);                                            \
                                                                               \
  template std::vector<sum_op<HandlerTy>> sum_op<HandlerTy>::distribute_terms( \
      std::size_t numChunks) const;

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
  return this->degrees().size();
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
sum_op<HandlerTy>::sum_op(const std::vector<double> &input_vec) {
  if (input_vec.size() == 0)
    throw std::runtime_error("input vector must not be empty");

  auto it = input_vec.cbegin();
  auto next_int = [&it, &input_vec]() {
    if (it == input_vec.end())
      throw std::runtime_error("incorrect data format - missing entry");
    double intPart;
    if (std::modf(*it, &intPart) != 0.0)
      throw std::runtime_error(
          "Invalid pauli data element, must be integer value.");
    return (std::size_t)*it++;
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
product_op<HandlerTy> sum_op<HandlerTy>::from_word(const std::string &arg) {
  std::string word{arg};
  word.erase(std::find(word.begin(), word.end(), '\0'), word.end());
  std::vector<HandlerTy> ops;
  ops.reserve(word.length());
  for (std::size_t i = 0; i < word.length(); i++) {
    auto letter = word[i];
    if (letter == 'Y')
      ops.push_back(HandlerTy::y(i));
    else if (letter == 'X')
      ops.push_back(HandlerTy::x(i));
    else if (letter == 'Z')
      ops.push_back(HandlerTy::z(i));
    else if (letter == 'I')
      ops.push_back(HandlerTy(i));
    else
      throw std::runtime_error(
          "Invalid Pauli for spin_op::from_word, must be X, Y, Z, or I.");
  }
  return product_op<HandlerTy>(1., std::move(ops));
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

  auto get_spin_op = [](std::size_t target, int kind) {
    if (kind == 1)
      return sum_op<HandlerTy>::z(target);
    if (kind == 2)
      return sum_op<HandlerTy>::x(target);
    if (kind == 3)
      return sum_op<HandlerTy>::y(target);
    return sum_op<HandlerTy>::i(target);
  };

  std::mt19937 gen(seed);
  auto sum = sum_op<HandlerTy>(true);
  // make sure the number of terms matches the requested number...
  while (sum.terms.size() < nTerms) {
    std::vector<bool> termData(2 * nQubits);
    std::fill_n(termData.begin(), nQubits, true);
    std::shuffle(termData.begin(), termData.end(), gen);
    // ... but allow for duplicates (will be a single term with coefficient !=
    // 1)
    auto prod = sum_op<HandlerTy>::identity();
    for (std::size_t qubit_idx = 0; qubit_idx < nQubits; ++qubit_idx) {
      auto kind =
          (termData[qubit_idx << 1] << 1) | termData[(qubit_idx << 1) + 1];
      // keep identities so that we act on the requested number of qubits
      prod *= get_spin_op(qubit_idx, kind);
    }
    sum += std::move(prod);
  }
  return sum;
}

template <typename HandlerTy>
PROPERTY_SPECIFIC_TEMPLATE_DEFINITION(HandlerTy,
                                      product_op<T>::supports_inplace_mult)
csr_spmatrix sum_op<HandlerTy>::to_sparse_matrix(
    std::unordered_map<std::size_t, std::int64_t> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool invert_order) const {
  auto evaluated = this->transform(
      operator_arithmetics<operator_handler::canonical_evaluation>(dimensions,
                                                                   parameters));

  if (evaluated.terms.size() == 0)
    return std::make_tuple<std::vector<std::complex<double>>,
                           std::vector<std::size_t>, std::vector<std::size_t>>(
        {}, {}, {});

  auto matrix = HandlerTy::to_sparse_matrix(
      evaluated.terms[0].encoding, evaluated.terms[0].relevant_dimensions,
      evaluated.terms[0].coefficient, invert_order);
  for (auto i = 1; i < terms.size(); ++i)
    matrix += HandlerTy::to_sparse_matrix(
        evaluated.terms[i].encoding, evaluated.terms[i].relevant_dimensions,
        evaluated.terms[i].coefficient, invert_order);
  return cudaq::detail::to_csr_spmatrix(
      matrix, 1ul << evaluated.terms[0].relevant_dimensions.size());
}

template <typename HandlerTy>
PROPERTY_SPECIFIC_TEMPLATE_DEFINITION(HandlerTy,
                                      product_op<T>::supports_inplace_mult)
mdiag_sparse_matrix sum_op<HandlerTy>::to_diagonal_matrix(
    std::unordered_map<std::size_t, std::int64_t> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool invert_order) const {
  auto evaluated = this->transform(
      operator_arithmetics<operator_handler::canonical_evaluation>(dimensions,
                                                                   parameters));

  if (evaluated.terms.size() == 0)
    return mdiag_sparse_matrix();

  auto dia_matrix = HandlerTy::to_diagonal_matrix(
      evaluated.terms[0].encoding, evaluated.terms[0].relevant_dimensions,
      evaluated.terms[0].coefficient, invert_order);
  for (auto i = 1; i < terms.size(); ++i)
    cudaq::detail::inplace_accumulate(
        dia_matrix,
        HandlerTy::to_diagonal_matrix(
            evaluated.terms[i].encoding, evaluated.terms[i].relevant_dimensions,
            evaluated.terms[i].coefficient, invert_order));
  return dia_matrix;
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
std::vector<double> sum_op<HandlerTy>::get_data_representation() const {
  auto nr_ops = 0;
  for (const auto &term : *this)
    nr_ops += term.operators.size();
  std::vector<double> dataVec;
  dataVec.reserve(2 * nr_ops + 3 * this->terms.size() + 1);
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

template std::size_t sum_op<spin_handler>::num_qubits() const;
template sum_op<spin_handler>::sum_op(const std::vector<double> &input_vec);
template product_op<spin_handler>
sum_op<spin_handler>::from_word(const std::string &word);
template sum_op<spin_handler> sum_op<spin_handler>::random(std::size_t nQubits,
                                                           std::size_t nTerms,
                                                           unsigned int seed);
template csr_spmatrix sum_op<spin_handler>::to_sparse_matrix(
    std::unordered_map<std::size_t, std::int64_t> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool invert_order) const;
template csr_spmatrix sum_op<fermion_handler>::to_sparse_matrix(
    std::unordered_map<std::size_t, int64_t> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool invert_order) const;
template csr_spmatrix sum_op<boson_handler>::to_sparse_matrix(
    std::unordered_map<std::size_t, int64_t> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool invert_order) const;
template mdiag_sparse_matrix sum_op<spin_handler>::to_diagonal_matrix(
    std::unordered_map<std::size_t, std::int64_t> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool invert_order) const;
template mdiag_sparse_matrix sum_op<fermion_handler>::to_diagonal_matrix(
    std::unordered_map<std::size_t, int64_t> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool invert_order) const;
template mdiag_sparse_matrix sum_op<boson_handler>::to_diagonal_matrix(
    std::unordered_map<std::size_t, int64_t> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool invert_order) const;

template std::vector<double>
sum_op<spin_handler>::get_data_representation() const;

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
  if (input_vec.size() == 0)
    throw std::runtime_error("input vector must not be empty");
  auto n_terms = (std::size_t)input_vec.back();
  if (n_terms == 0 ||
      nQubits != (((input_vec.size() - 1) - 2 * n_terms) / n_terms))
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

      int val = (std::size_t)input_vec[j + i];
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
  this->is_default = bsf_terms.size() == 0;
  this->coefficients.reserve(bsf_terms.size());
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
  // This function prints a data representing the operator sum that
  // includes the full representation for any degree in [0, max_degree),
  // padding identities if necessary.
  // NOTE: this is an imperfect representation that we will want to
  // deprecate because it does not capture targets accurately.
  auto degrees = this->degrees();
  auto le_order = std::less<std::size_t>();
  auto get_le_index = [&degrees, &le_order](std::size_t idx) {
    // For compatibility with existing code, the ordering for the term ops
    // always needs to be from smallest to largest degree.
    return (operator_handler::canonical_order(1, 0) == le_order(1, 0))
               ? idx
               : degrees.size() - 1 - idx;
  };

  // number of degrees including the ones for any injected identities
  auto n_targets = operator_handler::canonical_order(0, 1) ? degrees.back() + 1
                                                           : degrees[0] + 1;
  auto padded = *this; // copy for identity padding
  for (std::size_t j = 0; j < n_targets; ++j)
    padded *= sum_op<HandlerTy>::identity(j);

  std::vector<double> dataVec;
  dataVec.reserve(n_targets * padded.terms.size() + 2 * padded.terms.size() +
                  1);
  for (std::size_t i = 0; i < padded.terms.size(); ++i) {
    for (std::size_t j = 0; j < padded.terms[i].size(); ++j) {
      auto pauli = padded.terms[i][get_le_index(j)].as_pauli();
      if (pauli == pauli::X)
        dataVec.push_back(1.);
      else if (pauli == pauli::Z)
        dataVec.push_back(2.);
      else if (pauli == pauli::Y)
        dataVec.push_back(3.);
      else
        dataVec.push_back(0.);
    }
    auto coeff = padded.coefficients[i].evaluate();
    dataVec.push_back(coeff.real());
    dataVec.push_back(coeff.imag());
  }
  dataVec.push_back(padded.terms.size());
  return dataVec;
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
std::tuple<std::vector<double>, std::size_t>
sum_op<HandlerTy>::getDataTuple() const {
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  return std::make_tuple<std::vector<double>, std::size_t>(
      this->getDataRepresentation(), this->num_qubits());
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
std::pair<std::vector<std::vector<bool>>, std::vector<std::complex<double>>>
sum_op<HandlerTy>::get_raw_data() const {
  std::unordered_map<std::size_t, std::int64_t> dims;
  auto degrees = this->degrees();
  auto evaluated = this->transform(
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
    auto pauli_str = std::move(term.encoding);
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
    coeffs.push_back(std::move(term.coefficient));
  }

  // always little endian order by definition of the bsf
  return std::make_pair(std::move(bsf_terms), std::move(coeffs));
}

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
std::string sum_op<HandlerTy>::to_string(bool printCoeffs) const {
  // This function prints a string representing the operator sum that
  // includes the full representation for any degree in [0, max_degree),
  // padding identities if necessary (opposed to pauli_word).
  std::unordered_map<std::size_t, std::int64_t> dims;
  auto degrees = this->degrees();
  auto evaluated = this->transform(
      operator_arithmetics<operator_handler::canonical_evaluation>(dims, {}));
  auto le_order = std::less<std::size_t>();
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
      auto coeff = term.coefficient;
      ss << "[" << coeff.real() << (coeff.imag() < 0.0 ? "-" : "+")
         << std::fabs(coeff.imag()) << "j] ";
    }

    if (degrees.size() > 0) {
      auto max_target =
          operator_handler::canonical_order(0, 1) ? degrees.back() : degrees[0];
      std::string term_str(max_target + 1, 'I');
      for (std::size_t i = 0; i < degrees.size(); ++i)
        term_str[degrees[i]] = term.encoding[get_le_index(i)];
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

template sum_op<spin_handler>::sum_op(const std::vector<double> &input_vec,
                                      std::size_t nQubits);
template sum_op<spin_handler>::sum_op(
    const std::vector<std::vector<bool>> &bsf_terms,
    const std::vector<std::complex<double>> &coeffs);
template std::vector<double>
sum_op<spin_handler>::getDataRepresentation() const;
template std::tuple<std::vector<double>, std::size_t>
sum_op<spin_handler>::getDataTuple() const;
template std::pair<std::vector<std::vector<bool>>,
                   std::vector<std::complex<double>>>
sum_op<spin_handler>::get_raw_data() const;
template std::string sum_op<spin_handler>::to_string(bool printCoeffs) const;
template void sum_op<spin_handler>::for_each_term(
    std::function<void(sum_op<spin_handler> &)> &&functor) const;
template void sum_op<spin_handler>::for_each_pauli(
    std::function<void(pauli, std::size_t)> &&functor) const;
template bool sum_op<spin_handler>::is_identity() const;

#if defined(CUDAQ_INSTANTIATE_TEMPLATES)
template class sum_op<matrix_handler>;
template class sum_op<spin_handler>;
template class sum_op<boson_handler>;
template class sum_op<fermion_handler>;
#endif

} // namespace cudaq
