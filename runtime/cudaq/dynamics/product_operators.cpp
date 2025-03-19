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
#include <unordered_map>
#include <utility>

#include "cudaq/operators.h"
#include "evaluation.h"
#include "helpers.h"

namespace cudaq {

// private methods

#if !defined(NDEBUG)
// check canonicalization by default, individual handlers can set it to false to
// disable the check
bool operator_handler::can_be_canonicalized = true;

// returns true if and only if applying the operators in sequence acts only once
// on each degree of freedom and in canonical order
template <typename HandlerTy>
bool product_op<HandlerTy>::is_canonicalized() const {
  auto canon_degrees = this->degrees(false);
  std::vector<std::size_t> degrees;
  degrees.reserve(canon_degrees.size());
  for (const auto &op : this->operators) {
    for (auto d : op.degrees())
      degrees.push_back(d);
  }
  return degrees == canon_degrees;
}
#endif

template <typename HandlerTy>
std::vector<HandlerTy>::const_iterator
product_op<HandlerTy>::find_insert_at(const HandlerTy &other) {
  // This is the simplest logic for finding the insertion point that should be
  // used only when all operators are guaranteed to belong to commutation set 0
  // and have a single target. Neither is the case for matrix operators; we
  // handle this with a template specialization below. That template
  // specialization handles the most general case, but (necessarily) is also the
  // slowest. We have an additional template specialization for operators that
  // have non-trivial commutation relations across different degrees but have a
  // single target (e.g. fermions). That specialization is just to avoid the
  // overhead of having to compare multiple targets.
  assert(other.commutation_group ==
         operator_handler::default_commutation_relations);

  // The logic below ensures that terms are fully ordered in canonical order, as
  // long as the HandlerTy supports in-place multiplication.
  return std::find_if(this->operators.crbegin(), this->operators.crend(),
                      [other_degree = other.degree](const HandlerTy &self_op) {
                        return !operator_handler::canonical_order(
                            other_degree, self_op.degree);
                      })
      .base(); // base causes insert after for reverse iterator
}

template <>
std::vector<matrix_handler>::const_iterator
product_op<matrix_handler>::find_insert_at(const matrix_handler &other) {
  // This template specialization contains the most general (and least
  // performant) version of the logic to determine where to insert an operator
  // into the product. It takes commutation relations into account to try and
  // order the operators in canonical order as much as possible. For
  // multi-target operators, a canonical ordering cannot be defined/achieved.
  // IMPORTANT: It is necessary that we do try to partially canonicalize to
  // ensure that any anti- commutation relations are correct, since we achieve
  // non-trivial commutation relations by updating the coefficient upon
  // reordering terms! We should still get the correct relations even if we
  // necessarily don't have a full canonical order, since incomplete ordering
  // only occurs for multi-qubit terms. The only thing that does not work in
  // general is having an non-trivial commutation relation for multi- target
  // operators. The matrix operator class should fail to construct such an
  // operator.
  int nr_permutations = 0;
  auto rit = std::find_if(
      this->operators.crbegin(), this->operators.crend(),
      [&nr_permutations,
       &other_degrees = static_cast<const std::vector<int> &>(other.degrees()),
       &other](const matrix_handler &self_op) {
        const std::vector<int> &self_op_degrees = self_op.degrees();
        for (auto other_degree : other_degrees) {
          auto item_it =
              std::find_if(self_op_degrees.crbegin(), self_op_degrees.crend(),
                           [other_degree](int self_degree) {
                             return !operator_handler::canonical_order(
                                 other_degree, self_degree);
                           });
          if (item_it != self_op_degrees.crend()) {
            // we need to run this again to properly validate the defined
            // commutation sets - we need to know if we have an exact match of
            // the degree somewhere
            item_it =
                std::find_if(self_op_degrees.crbegin(), self_op_degrees.crend(),
                             [other_degree](int self_degree) {
                               return other_degree == self_degree;
                             });
            if (item_it != self_op_degrees.crend() &&
                other.commutation_group != self_op.commutation_group)
              // this indicates that the same degree of freedom is acted upon by
              // an operator of a different "commutation class", e.g. a fermion
              // operator has been applied and now we are trying to apply a
              // boson operator
              throw std::runtime_error(
                  "conflicting commutation relations defined for target " +
                  std::to_string(other_degree));
            return true;
          } else if (!other.commutes_across_degrees &&
                     !self_op.commutes_across_degrees &&
                     other.commutation_group !=
                         operator_handler::default_commutation_relations &&
                     self_op.commutation_group == other.commutation_group)
            nr_permutations += 1;
        }
        return false;
      });
  if (nr_permutations != 0)
    this->coefficient *=
        other.commutation_group.commutation_factor() * (double)nr_permutations;
  return rit.base(); // base causes insert after for reverse iterator
}

template <>
std::vector<fermion_handler>::const_iterator
product_op<fermion_handler>::find_insert_at(const fermion_handler &other) {
  assert(other.commutation_group ==
         operator_handler::fermion_commutation_relations);
  // This template specialization contains the same logic as the specialization
  // for matrix operators above, just written to rely on having a single target
  // qubit and a matching set id for all operators for the sake of avoiding
  // unnecessary overhead.
  bool negate_coefficient = false;
  auto rit = std::find_if(
      this->operators.crbegin(), this->operators.crend(),
      [&negate_coefficient, &other](const fermion_handler &self_op) {
        if (!operator_handler::canonical_order(other.degree, self_op.degree))
          return true;
        if (!other.commutes_across_degrees && !self_op.commutes_across_degrees)
          negate_coefficient = !negate_coefficient;
        return false;
      });
  if (negate_coefficient)
    this->coefficient *= -1.;
  return rit.base(); // base causes insert after for reverse iterator
}

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<std::is_same<HandlerTy, T>::value &&
                               !product_op<T>::supports_inplace_mult,
                           std::false_type>>
void product_op<HandlerTy>::insert(T &&other) {
  auto pos = this->find_insert_at(other);
  this->operators.insert(pos, other);
}

template <typename HandlerTy>
template <typename T, std::enable_if_t<std::is_same<HandlerTy, T>::value &&
                                           product_op<T>::supports_inplace_mult,
                                       std::true_type>>
void product_op<HandlerTy>::insert(T &&other) {
  auto pos = this->find_insert_at(other);
  if (pos != this->operators.begin() && (pos - 1)->degree == other.degree) {
    auto it = this->operators.erase(
        pos - 1,
        pos - 1); // erase: constant time conversion to non-const iterator
    it->inplace_mult(other);
  } else
    this->operators.insert(pos, std::move(other));
}

template <>
template <typename T, std::enable_if_t<std::is_same<spin_handler, T>::value &&
                                           product_op<T>::supports_inplace_mult,
                                       std::true_type>>
void product_op<spin_handler>::insert(T &&other) {
  auto pos = this->find_insert_at(other);
  if (pos != this->operators.begin() && (pos - 1)->degree == other.degree) {
    auto it = this->operators.erase(
        pos - 1,
        pos - 1); // erase: constant time conversion to non-const iterator
    this->coefficient *= it->inplace_mult(other);
  } else
    this->operators.insert(pos, std::move(other));
}

template <typename HandlerTy>
void product_op<HandlerTy>::aggregate_terms() {}

template <typename HandlerTy>
template <typename... Args>
void product_op<HandlerTy>::aggregate_terms(HandlerTy &&head, Args &&...args) {
  this->insert(std::forward<HandlerTy>(head));
  aggregate_terms(std::forward<Args>(args)...);
}

template <typename HandlerTy>
template <typename EvalTy>
EvalTy product_op<HandlerTy>::evaluate(
    operator_arithmetics<EvalTy> arithmetics) const {

  assert(!HandlerTy::can_be_canonicalized || this->is_canonicalized());
  auto degrees = this->degrees(false); // keep in canonical order

  auto padded_op = [&arithmetics,
                    &degrees = std::as_const(degrees)](const HandlerTy &op) {
    std::vector<EvalTy> evaluated;
    auto op_degrees = op.degrees();
    bool op_evaluated = false;
    for (const auto &degree : degrees) {
      if (std::find(op_degrees.cbegin(), op_degrees.cend(), degree) ==
          op_degrees.cend())
        evaluated.push_back(arithmetics.evaluate(HandlerTy(degree)));
      // if op has more than one degree of freedom, then evaluating it here
      // would potentially lead to a matrix reordering upon application of each
      // subsequent id
      else if (op_degrees.size() == 1 && !op_evaluated) {
        evaluated.push_back(arithmetics.evaluate(op));
        op_evaluated = true;
      }
    }

    if (evaluated.size() == 0)
      return arithmetics.evaluate(op);

    // Creating the tensor product with op being last is most efficient if op
    // acts on more than one degree of freedom - this ensure that only a single
    // reordering happens at at the end.
    EvalTy product = std::move(evaluated[0]);
    for (auto i = 1; i < evaluated.size(); ++i)
      product = arithmetics.tensor(std::move(product), std::move(evaluated[i]));
    if (op_evaluated)
      return std::move(product);
    else
      return arithmetics.tensor(std::move(product), arithmetics.evaluate(op));
  };

  if (arithmetics.pad_product_terms) {
    if (degrees.size() == 0)
      return arithmetics.evaluate(this->coefficient);
    EvalTy prod = padded_op(this->operators[0]);
    for (auto op_idx = 1; op_idx < this->operators.size(); ++op_idx) {
      auto op_degrees = this->operators[op_idx].degrees();
      if (op_degrees.size() != 1 ||
          this->operators[op_idx] != HandlerTy(op_degrees[0]))
        prod = arithmetics.mul(std::move(prod),
                               padded_op(this->operators[op_idx]));
    }
    return arithmetics.mul(this->coefficient, std::move(prod));
  } else {
    EvalTy prod = arithmetics.evaluate(this->coefficient);
    for (auto op_idx = 0; op_idx < this->operators.size(); ++op_idx) {
      EvalTy eval = arithmetics.evaluate(this->operators[op_idx]);
      prod = arithmetics.tensor(std::move(prod), std::move(eval));
    }
    return std::move(prod);
  }
}

// Note:
// These are the private methods needed by friend classes and functions
// of product_op. For clang, (only!) these need to be instantiated
// explicitly to be available to those.
#define INSTANTIATE_PRODUCT_PRIVATE_METHODS(HandlerTy)                         \
                                                                               \
  template typename std::vector<HandlerTy>::const_iterator                     \
  product_op<HandlerTy>::find_insert_at(const HandlerTy &other);               \
                                                                               \
  template void product_op<HandlerTy>::insert(HandlerTy &&other);              \
                                                                               \
  template void product_op<HandlerTy>::aggregate_terms(HandlerTy &&item1,      \
                                                       HandlerTy &&item2);     \
                                                                               \
  template void product_op<HandlerTy>::aggregate_terms(                        \
      HandlerTy &&item1, HandlerTy &&item2, HandlerTy &&item3);

#define INSTANTIATE_PRODUCT_PRIVATE_FRIEND_METHODS(HandlerTy)                  \
                                                                               \
  template void product_op<HandlerTy>::insert(HandlerTy &&other);

#if !defined(__clang__)
INSTANTIATE_PRODUCT_PRIVATE_METHODS(matrix_handler);
INSTANTIATE_PRODUCT_PRIVATE_METHODS(spin_handler);
INSTANTIATE_PRODUCT_PRIVATE_METHODS(boson_handler);
INSTANTIATE_PRODUCT_PRIVATE_METHODS(fermion_handler);
#else
INSTANTIATE_PRODUCT_PRIVATE_FRIEND_METHODS(matrix_handler);
INSTANTIATE_PRODUCT_PRIVATE_FRIEND_METHODS(spin_handler);
INSTANTIATE_PRODUCT_PRIVATE_FRIEND_METHODS(boson_handler);
INSTANTIATE_PRODUCT_PRIVATE_FRIEND_METHODS(fermion_handler);
#endif

#define INSTANTIATE_PRODUCT_EVALUATE_METHODS(HandlerTy, EvalTy)                \
                                                                               \
  template EvalTy product_op<HandlerTy>::evaluate(                             \
      operator_arithmetics<EvalTy> arithmetics) const;

INSTANTIATE_PRODUCT_EVALUATE_METHODS(matrix_handler,
                                     operator_handler::matrix_evaluation);
INSTANTIATE_PRODUCT_EVALUATE_METHODS(spin_handler,
                                     operator_handler::canonical_evaluation);
INSTANTIATE_PRODUCT_EVALUATE_METHODS(boson_handler,
                                     operator_handler::matrix_evaluation);
INSTANTIATE_PRODUCT_EVALUATE_METHODS(fermion_handler,
                                     operator_handler::matrix_evaluation);

// read-only properties

template <typename HandlerTy>
std::vector<std::size_t>
product_op<HandlerTy>::degrees(bool application_order) const {
  std::set<std::size_t> unsorted_degrees;
  for (const HandlerTy &term : this->operators) {
    auto term_degrees = term.degrees();
    unsorted_degrees.insert(term_degrees.cbegin(), term_degrees.cend());
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
std::size_t product_op<HandlerTy>::num_ops() const {
  return this->operators.size();
}

template <typename HandlerTy>
std::string product_op<HandlerTy>::get_term_id() const {
  std::string term_id;
  for (const auto &op : this->operators)
    term_id += op.unique_id();
  return std::move(term_id);
}

template <typename HandlerTy>
scalar_operator product_op<HandlerTy>::get_coefficient() const {
  return this->coefficient;
}

#define INSTANTIATE_PRODUCT_PROPERTIES(HandlerTy)                              \
                                                                               \
  template std::vector<std::size_t> product_op<HandlerTy>::degrees(            \
      bool application_order) const;                                           \
                                                                               \
  template std::size_t product_op<HandlerTy>::num_ops() const;                 \
                                                                               \
  template std::string product_op<HandlerTy>::get_term_id() const;             \
                                                                               \
  template scalar_operator product_op<HandlerTy>::get_coefficient() const;

#if !defined(__clang__)
INSTANTIATE_PRODUCT_PROPERTIES(matrix_handler);
INSTANTIATE_PRODUCT_PROPERTIES(spin_handler);
INSTANTIATE_PRODUCT_PROPERTIES(boson_handler);
INSTANTIATE_PRODUCT_PROPERTIES(fermion_handler);
#endif

// constructors

template <typename HandlerTy>
product_op<HandlerTy>::product_op(double coefficient)
    : coefficient(coefficient) {}

template <typename HandlerTy>
product_op<HandlerTy>::product_op(std::complex<double> coefficient)
    : coefficient(coefficient) {}

template <typename HandlerTy>
product_op<HandlerTy>::product_op(HandlerTy &&atomic) : coefficient(1.) {
  this->operators.push_back(std::move(atomic));
  assert(!HandlerTy::can_be_canonicalized ||
         this->is_canonicalized()); // relevant for custom matrix operators
                                    // acting on multiple degrees of freedom
}

template <typename HandlerTy>
template <typename... Args,
          std::enable_if_t<
              std::conjunction<std::is_same<HandlerTy, Args>...>::value, bool>>
product_op<HandlerTy>::product_op(scalar_operator coefficient, Args &&...args)
    : coefficient(std::move(coefficient)) {
  this->operators.reserve(sizeof...(Args));
  aggregate_terms(std::forward<HandlerTy &&>(args)...);
  assert(!HandlerTy::can_be_canonicalized || this->is_canonicalized());
}

// assumes canonical ordering (if possible)
template <typename HandlerTy>
product_op<HandlerTy>::product_op(
    scalar_operator coefficient, const std::vector<HandlerTy> &atomic_operators,
    int size)
    : coefficient(std::move(coefficient)) {
  if (size <= 0)
    this->operators = atomic_operators;
  else {
    this->operators.reserve(size);
    for (const auto &op : atomic_operators)
      this->operators.push_back(op);
  }
  assert(!HandlerTy::can_be_canonicalized || this->is_canonicalized());
}

// assumes canonical ordering (if possible)
template <typename HandlerTy>
product_op<HandlerTy>::product_op(scalar_operator coefficient,
                                  std::vector<HandlerTy> &&atomic_operators,
                                  int size)
    : coefficient(std::move(coefficient)),
      operators(std::move(atomic_operators)) {
  if (size > 0)
    this->operators.reserve(size);
  assert(!HandlerTy::can_be_canonicalized || this->is_canonicalized());
}

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                               std::is_constructible<HandlerTy, T>::value,
                           bool>>
product_op<HandlerTy>::product_op(const product_op<T> &other)
    : coefficient(other.coefficient) {
  this->operators.reserve(other.operators.size());
  for (const T &other_op : other.operators) {
    HandlerTy op(other_op);
    this->operators.push_back(op);
  }
}

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<std::is_same<HandlerTy, matrix_handler>::value &&
                               !std::is_same<T, HandlerTy>::value &&
                               std::is_constructible<HandlerTy, T>::value,
                           bool>>
product_op<HandlerTy>::product_op(
    const product_op<T> &other,
    const matrix_handler::commutation_behavior &behavior)
    : coefficient(other.coefficient) {
  this->operators.reserve(other.operators.size());
  for (const T &other_op : other.operators) {
    HandlerTy op(other_op, behavior);
    this->operators.push_back(op);
  }
}

template <typename HandlerTy>
product_op<HandlerTy>::product_op(const product_op<HandlerTy> &other, int size)
    : coefficient(other.coefficient) {
  if (size <= 0)
    this->operators = other.operators;
  else {
    this->operators.reserve(size);
    for (const auto &op : other.operators)
      this->operators.push_back(op);
  }
}

template <typename HandlerTy>
product_op<HandlerTy>::product_op(product_op<HandlerTy> &&other, int size)
    : coefficient(std::move(other.coefficient)),
      operators(std::move(other.operators)) {
  if (size > 0)
    this->operators.reserve(size);
}

#define INSTANTIATE_PRODUCT_CONSTRUCTORS(HandlerTy)                            \
                                                                               \
  template product_op<HandlerTy>::product_op(double coefficient);              \
                                                                               \
  template product_op<HandlerTy>::product_op(                                  \
      std::complex<double> coefficient);                                       \
                                                                               \
  template product_op<HandlerTy>::product_op(scalar_operator coefficient);     \
                                                                               \
  template product_op<HandlerTy>::product_op(HandlerTy &&atomic);              \
                                                                               \
  template product_op<HandlerTy>::product_op(scalar_operator coefficient,      \
                                             HandlerTy &&atomic1);             \
                                                                               \
  template product_op<HandlerTy>::product_op(                                  \
      scalar_operator coefficient, HandlerTy &&atomic1, HandlerTy &&atomic2);  \
                                                                               \
  template product_op<HandlerTy>::product_op(                                  \
      scalar_operator coefficient, HandlerTy &&atomic1, HandlerTy &&atomic2,   \
      HandlerTy &&atomic3);                                                    \
                                                                               \
  template product_op<HandlerTy>::product_op(                                  \
      scalar_operator coefficient,                                             \
      const std::vector<HandlerTy> &atomic_operators, int size);               \
                                                                               \
  template product_op<HandlerTy>::product_op(                                  \
      scalar_operator coefficient, std::vector<HandlerTy> &&atomic_operators,  \
      int size);                                                               \
                                                                               \
  template product_op<HandlerTy>::product_op(                                  \
      const product_op<HandlerTy> &other, int size);                           \
                                                                               \
  template product_op<HandlerTy>::product_op(product_op<HandlerTy> &&other,    \
                                             int size);

// Note:
// These are the private constructors needed by friend classes and functions
// of product_op. For clang, (only!) these need to be instantiated
// explicitly to be available to those.
#define INSTANTIATE_PRODUCT_PRIVATE_FRIEND_CONSTRUCTORS(HandlerTy)             \
                                                                               \
  template product_op<HandlerTy>::product_op(scalar_operator coefficient);     \
                                                                               \
  template product_op<HandlerTy>::product_op(scalar_operator coefficient,      \
                                             HandlerTy &&atomic1);

template product_op<matrix_handler>::product_op(
    const product_op<spin_handler> &other);
template product_op<matrix_handler>::product_op(
    const product_op<boson_handler> &other);
template product_op<matrix_handler>::product_op(
    const product_op<fermion_handler> &other);
template product_op<matrix_handler>::product_op(
    const product_op<spin_handler> &other,
    const matrix_handler::commutation_behavior &behavior);
template product_op<matrix_handler>::product_op(
    const product_op<boson_handler> &other,
    const matrix_handler::commutation_behavior &behavior);
template product_op<matrix_handler>::product_op(
    const product_op<fermion_handler> &other,
    const matrix_handler::commutation_behavior &behavior);

#if !defined(__clang__)
INSTANTIATE_PRODUCT_CONSTRUCTORS(matrix_handler);
INSTANTIATE_PRODUCT_CONSTRUCTORS(spin_handler);
INSTANTIATE_PRODUCT_CONSTRUCTORS(boson_handler);
INSTANTIATE_PRODUCT_CONSTRUCTORS(fermion_handler);
#else
INSTANTIATE_PRODUCT_PRIVATE_FRIEND_CONSTRUCTORS(matrix_handler);
INSTANTIATE_PRODUCT_PRIVATE_FRIEND_CONSTRUCTORS(spin_handler);
INSTANTIATE_PRODUCT_PRIVATE_FRIEND_CONSTRUCTORS(boson_handler);
INSTANTIATE_PRODUCT_PRIVATE_FRIEND_CONSTRUCTORS(fermion_handler);
#endif

// assignments

template <typename HandlerTy>
template <typename T,
          std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                               std::is_constructible<HandlerTy, T>::value,
                           bool>>
product_op<HandlerTy> &
product_op<HandlerTy>::operator=(const product_op<T> &other) {
  *this = product_op<HandlerTy>(other);
  return *this;
}

template <typename HandlerTy>
product_op<HandlerTy> &
product_op<HandlerTy>::operator=(const product_op<HandlerTy> &other) {
  if (this != &other) {
    this->coefficient = other.coefficient;
    this->operators = other.operators;
  }
  return *this;
}

template <typename HandlerTy>
product_op<HandlerTy> &
product_op<HandlerTy>::operator=(product_op<HandlerTy> &&other) {
  if (this != &other) {
    this->coefficient = std::move(other.coefficient);
    this->operators = std::move(other.operators);
  }
  return *this;
}

#define INSTANTIATE_PRODUCT_ASSIGNMENTS(HandlerTy)                             \
                                                                               \
  template product_op<HandlerTy> &product_op<HandlerTy>::operator=(            \
      const product_op<HandlerTy> &other);                                     \
                                                                               \
  template product_op<HandlerTy> &product_op<HandlerTy>::operator=(            \
      product_op<HandlerTy> &&other);

template product_op<matrix_handler> &
product_op<matrix_handler>::operator=(const product_op<spin_handler> &other);
template product_op<matrix_handler> &
product_op<matrix_handler>::operator=(const product_op<boson_handler> &other);
template product_op<matrix_handler> &
product_op<matrix_handler>::operator=(const product_op<fermion_handler> &other);

#if !defined(__clang__)
INSTANTIATE_PRODUCT_ASSIGNMENTS(matrix_handler);
INSTANTIATE_PRODUCT_ASSIGNMENTS(spin_handler);
INSTANTIATE_PRODUCT_ASSIGNMENTS(boson_handler);
INSTANTIATE_PRODUCT_ASSIGNMENTS(fermion_handler);
#endif

// evaluations

template <typename HandlerTy>
std::string product_op<HandlerTy>::to_string() const {
  std::stringstream str;
  str << this->coefficient.to_string();
  if (this->operators.size() > 0)
    str << " * ";
  for (const auto &op : this->operators)
    str << op.to_string(true);
  return str.str();
}

template <typename HandlerTy>
complex_matrix product_op<HandlerTy>::to_matrix(
    std::unordered_map<int, int> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool application_order) const {
  auto evaluated =
      this->evaluate(operator_arithmetics<operator_handler::matrix_evaluation>(
          dimensions, parameters));
  auto matrix = std::move(evaluated.matrix);
  if (!application_order || operator_handler::canonical_order(1, 0) ==
                                operator_handler::user_facing_order(1, 0))
    return std::move(matrix);

  auto degrees = evaluated.degrees;
  std::sort(degrees.begin(), degrees.end(),
            operator_handler::user_facing_order);
  auto permutation = cudaq::detail::compute_permutation(evaluated.degrees,
                                                        degrees, dimensions);
  cudaq::detail::permute_matrix(matrix, permutation);
  return std::move(matrix);
}

template <>
complex_matrix product_op<spin_handler>::to_matrix(
    std::unordered_map<int, int> dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool application_order) const {
  auto terms = std::move(
      this->evaluate(
              operator_arithmetics<operator_handler::canonical_evaluation>(
                  dimensions, parameters))
          .terms);
  assert(terms.size() == 1);
  bool invert_order =
      application_order && operator_handler::canonical_order(1, 0) !=
                               operator_handler::user_facing_order(1, 0);
  auto matrix =
      spin_handler::to_matrix(terms[0].second, terms[0].first, invert_order);
  return std::move(matrix);
}

#define INSTANTIATE_PRODUCT_EVALUATIONS(HandlerTy)                             \
                                                                               \
  template std::string product_op<HandlerTy>::to_string() const;               \
                                                                               \
  template complex_matrix product_op<HandlerTy>::to_matrix(                    \
      std::unordered_map<int, int> dimensions,                                 \
      const std::unordered_map<std::string, std::complex<double>> &parameters, \
      bool application_order) const;

#if !defined(__clang__)
INSTANTIATE_PRODUCT_EVALUATIONS(matrix_handler);
INSTANTIATE_PRODUCT_EVALUATIONS(spin_handler);
INSTANTIATE_PRODUCT_EVALUATIONS(boson_handler);
INSTANTIATE_PRODUCT_EVALUATIONS(fermion_handler);
#endif

// comparisons

template <typename HandlerTy>
bool product_op<HandlerTy>::operator==(
    const product_op<HandlerTy> &other) const {
  return this->coefficient == other.coefficient &&
         this->get_term_id() == other.get_term_id();
}

#define INSTANTIATE_PRODUCT_COMPARISONS(HandlerTy)                             \
                                                                               \
  template bool product_op<HandlerTy>::operator==(                             \
      const product_op<HandlerTy> &other) const;

#if !defined(__clang__)
INSTANTIATE_PRODUCT_COMPARISONS(matrix_handler);
INSTANTIATE_PRODUCT_COMPARISONS(spin_handler);
INSTANTIATE_PRODUCT_COMPARISONS(boson_handler);
INSTANTIATE_PRODUCT_COMPARISONS(fermion_handler);
#endif

// unary operators

template <typename HandlerTy>
product_op<HandlerTy> product_op<HandlerTy>::operator-() const & {
  return product_op<HandlerTy>(-1. * this->coefficient, this->operators);
}

template <typename HandlerTy>
product_op<HandlerTy> product_op<HandlerTy>::operator-() && {
  this->coefficient *= -1.;
  return std::move(*this);
}

template <typename HandlerTy>
product_op<HandlerTy> product_op<HandlerTy>::operator+() const & {
  return *this;
}

template <typename HandlerTy>
product_op<HandlerTy> product_op<HandlerTy>::operator+() && {
  return std::move(*this);
}

#define INSTANTIATE_PRODUCT_UNARY_OPS(HandlerTy)                               \
  template product_op<HandlerTy> product_op<HandlerTy>::operator-() const &;   \
  template product_op<HandlerTy> product_op<HandlerTy>::operator-() &&;        \
  template product_op<HandlerTy> product_op<HandlerTy>::operator+() const &;   \
  template product_op<HandlerTy> product_op<HandlerTy>::operator+() &&;

#if !defined(__clang__)
INSTANTIATE_PRODUCT_UNARY_OPS(matrix_handler);
INSTANTIATE_PRODUCT_UNARY_OPS(spin_handler);
INSTANTIATE_PRODUCT_UNARY_OPS(boson_handler);
INSTANTIATE_PRODUCT_UNARY_OPS(fermion_handler);
#endif

// right-hand arithmetics

#define PRODUCT_MULTIPLICATION_SCALAR(op)                                      \
                                                                               \
  template <typename HandlerTy>                                                \
  product_op<HandlerTy> product_op<HandlerTy>::operator op(                    \
      const scalar_operator &other) const & {                                  \
    return product_op<HandlerTy>(this->coefficient op other, this->operators); \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  product_op<HandlerTy> product_op<HandlerTy>::operator op(                    \
      scalar_operator &&other) const & {                                       \
    return product_op<HandlerTy>(this->coefficient op std::move(other),        \
                                 this->operators);                             \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  product_op<HandlerTy> product_op<HandlerTy>::operator op(                    \
      const scalar_operator &other) && {                                       \
    this->coefficient op## = other;                                            \
    return std::move(*this);                                                   \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  product_op<HandlerTy> product_op<HandlerTy>::operator op(                    \
      scalar_operator &&other) && {                                            \
    this->coefficient op## = std::move(other);                                 \
    return std::move(*this);                                                   \
  }

PRODUCT_MULTIPLICATION_SCALAR(*);
PRODUCT_MULTIPLICATION_SCALAR(/);

#define PRODUCT_ADDITION_SCALAR(op)                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      const scalar_operator &other) const & {                                  \
    return sum_op<HandlerTy>(product_op<HandlerTy>(op other),                  \
                             product_op<HandlerTy>(*this));                    \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      scalar_operator &&other) const & {                                       \
    return sum_op<HandlerTy>(product_op<HandlerTy>(op std::move(other)),       \
                             product_op<HandlerTy>(*this));                    \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      const scalar_operator &other) && {                                       \
    return sum_op<HandlerTy>(product_op<HandlerTy>(op other),                  \
                             std::move(*this));                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      scalar_operator &&other) && {                                            \
    return sum_op<HandlerTy>(product_op<HandlerTy>(op std::move(other)),       \
                             std::move(*this));                                \
  }

PRODUCT_ADDITION_SCALAR(+);
PRODUCT_ADDITION_SCALAR(-);

#define INSTANTIATE_PRODUCT_RHSIMPLE_OPS(HandlerTy)                            \
                                                                               \
  template product_op<HandlerTy> product_op<HandlerTy>::operator*(             \
      scalar_operator &&other) const &;                                        \
  template product_op<HandlerTy> product_op<HandlerTy>::operator*(             \
      scalar_operator &&other) &&;                                             \
  template product_op<HandlerTy> product_op<HandlerTy>::operator*(             \
      const scalar_operator &other) const &;                                   \
  template product_op<HandlerTy> product_op<HandlerTy>::operator*(             \
      const scalar_operator &other) &&;                                        \
  template product_op<HandlerTy> product_op<HandlerTy>::operator/(             \
      scalar_operator &&other) const &;                                        \
  template product_op<HandlerTy> product_op<HandlerTy>::operator/(             \
      scalar_operator &&other) &&;                                             \
  template product_op<HandlerTy> product_op<HandlerTy>::operator/(             \
      const scalar_operator &other) const &;                                   \
  template product_op<HandlerTy> product_op<HandlerTy>::operator/(             \
      const scalar_operator &other) &&;                                        \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      scalar_operator &&other) const &;                                        \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      scalar_operator &&other) &&;                                             \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      const scalar_operator &other) const &;                                   \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      const scalar_operator &other) &&;                                        \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      scalar_operator &&other) const &;                                        \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      scalar_operator &&other) &&;                                             \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      const scalar_operator &other) const &;                                   \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      const scalar_operator &other) &&;

#if !defined(__clang__)
INSTANTIATE_PRODUCT_RHSIMPLE_OPS(matrix_handler);
INSTANTIATE_PRODUCT_RHSIMPLE_OPS(spin_handler);
INSTANTIATE_PRODUCT_RHSIMPLE_OPS(boson_handler);
INSTANTIATE_PRODUCT_RHSIMPLE_OPS(fermion_handler);
#endif

template <typename HandlerTy>
product_op<HandlerTy>
product_op<HandlerTy>::operator*(const product_op<HandlerTy> &other) const & {
  product_op<HandlerTy> prod(this->coefficient * other.coefficient,
                             this->operators,
                             this->operators.size() + other.operators.size());
  for (HandlerTy op : other.operators)
    prod.insert(std::move(op));
  return std::move(prod);
}

template <typename HandlerTy>
product_op<HandlerTy>
product_op<HandlerTy>::operator*(const product_op<HandlerTy> &other) && {
  this->coefficient *= other.coefficient;
  this->operators.reserve(this->operators.size() + other.operators.size());
  for (HandlerTy op : other.operators)
    this->insert(std::move(op));
  return std::move(*this);
}

template <typename HandlerTy>
product_op<HandlerTy>
product_op<HandlerTy>::operator*(product_op<HandlerTy> &&other) const & {
  product_op<HandlerTy> prod(this->coefficient * std::move(other.coefficient),
                             this->operators,
                             this->operators.size() + other.operators.size());
  for (auto &&op : other.operators)
    prod.insert(std::move(op));
  return std::move(prod);
}

template <typename HandlerTy>
product_op<HandlerTy>
product_op<HandlerTy>::operator*(product_op<HandlerTy> &&other) && {
  this->coefficient *= std::move(other.coefficient);
  this->operators.reserve(this->operators.size() + other.operators.size());
  for (auto &&op : other.operators)
    this->insert(std::move(op));
  return std::move(*this);
}

#define PRODUCT_ADDITION_PRODUCT(op)                                           \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      const product_op<HandlerTy> &other) const & {                            \
    return sum_op<HandlerTy>(product_op<HandlerTy>(*this), op other);          \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      const product_op<HandlerTy> &other) && {                                 \
    return sum_op<HandlerTy>(std::move(*this), op other);                      \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      product_op<HandlerTy> &&other) const & {                                 \
    return sum_op<HandlerTy>(product_op<HandlerTy>(*this),                     \
                             op std::move(other));                             \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      product_op<HandlerTy> &&other) && {                                      \
    return sum_op<HandlerTy>(std::move(*this), op std::move(other));           \
  }

PRODUCT_ADDITION_PRODUCT(+)
PRODUCT_ADDITION_PRODUCT(-)

template <typename HandlerTy>
sum_op<HandlerTy>
product_op<HandlerTy>::operator*(const sum_op<HandlerTy> &other) const {
  sum_op<HandlerTy>
      sum; // everything needs to be updated, so creating a new sum makes sense
  sum.coefficients.reserve(other.coefficients.size());
  sum.term_map.reserve(other.terms.size());
  sum.terms.reserve(other.terms.size());
  for (auto i = 0; i < other.terms.size(); ++i) {
    auto prod =
        *this * product_op<HandlerTy>(other.coefficients[i], other.terms[i]);
    sum.insert(std::move(prod));
  }
  return std::move(sum);
}

#define PRODUCT_ADDITION_SUM(op)                                               \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      const sum_op<HandlerTy> &other) const & {                                \
    sum_op<HandlerTy> sum;                                                     \
    sum.coefficients.reserve(other.coefficients.size() + 1);                   \
    sum.term_map = other.term_map;                                             \
    sum.terms = other.terms;                                                   \
    for (auto &coeff : other.coefficients)                                     \
      sum.coefficients.push_back(op coeff);                                    \
    sum.insert(*this);                                                         \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      const sum_op<HandlerTy> &other) && {                                     \
    sum_op<HandlerTy> sum;                                                     \
    sum.coefficients.reserve(other.coefficients.size() + 1);                   \
    sum.term_map = other.term_map;                                             \
    sum.terms = other.terms;                                                   \
    for (auto &coeff : other.coefficients)                                     \
      sum.coefficients.push_back(op coeff);                                    \
    sum.insert(std::move(*this));                                              \
    return std::move(sum);                                                     \
  }                                                                            \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      sum_op<HandlerTy> &&other) const & {                                     \
    sum_op<HandlerTy> sum(op std::move(other));                                \
    sum.insert(*this);                                                         \
    return std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> product_op<HandlerTy>::operator op(                        \
      sum_op<HandlerTy> &&other) && {                                          \
    sum_op<HandlerTy> sum(op std::move(other));                                \
    sum.insert(std::move(*this));                                              \
    return std::move(sum);                                                     \
  }

PRODUCT_ADDITION_SUM(+)
PRODUCT_ADDITION_SUM(-)

#define INSTANTIATE_PRODUCT_RHCOMPOSITE_OPS(HandlerTy)                         \
                                                                               \
  template product_op<HandlerTy> product_op<HandlerTy>::operator*(             \
      const product_op<HandlerTy> &other) const &;                             \
  template product_op<HandlerTy> product_op<HandlerTy>::operator*(             \
      const product_op<HandlerTy> &other) &&;                                  \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      const product_op<HandlerTy> &other) const &;                             \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      const product_op<HandlerTy> &other) &&;                                  \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      product_op<HandlerTy> &&other) const &;                                  \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      product_op<HandlerTy> &&other) &&;                                       \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      const product_op<HandlerTy> &other) const &;                             \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      const product_op<HandlerTy> &other) &&;                                  \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      product_op<HandlerTy> &&other) const &;                                  \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      product_op<HandlerTy> &&other) &&;                                       \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator*(                 \
      const sum_op<HandlerTy> &other) const;                                   \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      const sum_op<HandlerTy> &other) const &;                                 \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      const sum_op<HandlerTy> &other) &&;                                      \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      sum_op<HandlerTy> &&other) const &;                                      \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator+(                 \
      sum_op<HandlerTy> &&other) &&;                                           \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      const sum_op<HandlerTy> &other) const &;                                 \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      const sum_op<HandlerTy> &other) &&;                                      \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      sum_op<HandlerTy> &&other) const &;                                      \
  template sum_op<HandlerTy> product_op<HandlerTy>::operator-(                 \
      sum_op<HandlerTy> &&other) &&;

#if !defined(__clang__)
INSTANTIATE_PRODUCT_RHCOMPOSITE_OPS(matrix_handler);
INSTANTIATE_PRODUCT_RHCOMPOSITE_OPS(spin_handler);
INSTANTIATE_PRODUCT_RHCOMPOSITE_OPS(boson_handler);
INSTANTIATE_PRODUCT_RHCOMPOSITE_OPS(fermion_handler);
#endif

#define PRODUCT_MULTIPLICATION_SCALAR_ASSIGNMENT(op)                           \
  template <typename HandlerTy>                                                \
  product_op<HandlerTy> &product_op<HandlerTy>::operator op(                   \
      const scalar_operator &other) {                                          \
    this->coefficient op other;                                                \
    return *this;                                                              \
  }

PRODUCT_MULTIPLICATION_SCALAR_ASSIGNMENT(*=);
PRODUCT_MULTIPLICATION_SCALAR_ASSIGNMENT(/=);

template <typename HandlerTy>
product_op<HandlerTy> &
product_op<HandlerTy>::operator*=(const product_op<HandlerTy> &other) {
  this->coefficient *= other.coefficient;
  this->operators.reserve(this->operators.size() + other.operators.size());
  for (HandlerTy op : other.operators)
    this->insert(std::move(op));
  return *this;
}

template <typename HandlerTy>
product_op<HandlerTy> &
product_op<HandlerTy>::operator*=(product_op<HandlerTy> &&other) {
  this->coefficient *= std::move(other.coefficient);
  this->operators.reserve(this->operators.size() + other.operators.size());
  for (auto &&op : other.operators)
    this->insert(std::move(op));
  return *this;
}

#define INSTANTIATE_PRODUCT_OPASSIGNMENTS(HandlerTy)                           \
                                                                               \
  template product_op<HandlerTy> &product_op<HandlerTy>::operator*=(           \
      const scalar_operator &other);                                           \
  template product_op<HandlerTy> &product_op<HandlerTy>::operator/=(           \
      const scalar_operator &other);                                           \
  template product_op<HandlerTy> &product_op<HandlerTy>::operator*=(           \
      const product_op<HandlerTy> &other);                                     \
  template product_op<HandlerTy> &product_op<HandlerTy>::operator*=(           \
      product_op<HandlerTy> &&other);

#if !defined(__clang__)
INSTANTIATE_PRODUCT_OPASSIGNMENTS(matrix_handler);
INSTANTIATE_PRODUCT_OPASSIGNMENTS(spin_handler);
INSTANTIATE_PRODUCT_OPASSIGNMENTS(boson_handler);
INSTANTIATE_PRODUCT_OPASSIGNMENTS(fermion_handler);
#endif

// left-hand arithmetics

template <typename HandlerTy>
product_op<HandlerTy> operator*(const scalar_operator &other,
                                const product_op<HandlerTy> &self) {
  return product_op<HandlerTy>(other * self.coefficient, self.operators);
}

template <typename HandlerTy>
product_op<HandlerTy> operator*(scalar_operator &&other,
                                const product_op<HandlerTy> &self) {
  other *= self.coefficient;
  return product_op<HandlerTy>(std::move(other), self.operators);
}

template <typename HandlerTy>
product_op<HandlerTy> operator*(const scalar_operator &other,
                                product_op<HandlerTy> &&self) {
  self.coefficient *= other;
  return std::move(self);
}

template <typename HandlerTy>
product_op<HandlerTy> operator*(scalar_operator &&other,
                                product_op<HandlerTy> &&self) {
  self.coefficient *= std::move(other);
  return std::move(self);
}

#define PRODUCT_ADDITION_SCALAR_REVERSE(op)                                    \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(const scalar_operator &other,                  \
                                const product_op<HandlerTy> &self) {           \
    return sum_op<HandlerTy>(product_op<HandlerTy>(other), op self);           \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(scalar_operator &&other,                       \
                                const product_op<HandlerTy> &self) {           \
    return sum_op<HandlerTy>(product_op<HandlerTy>(std::move(other)),          \
                             op self);                                         \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(const scalar_operator &other,                  \
                                product_op<HandlerTy> &&self) {                \
    return sum_op<HandlerTy>(product_op<HandlerTy>(other),                     \
                             op std::move(self));                              \
  }                                                                            \
                                                                               \
  template <typename HandlerTy>                                                \
  sum_op<HandlerTy> operator op(scalar_operator &&other,                       \
                                product_op<HandlerTy> &&self) {                \
    return sum_op<HandlerTy>(product_op<HandlerTy>(std::move(other)),          \
                             op std::move(self));                              \
  }

PRODUCT_ADDITION_SCALAR_REVERSE(+);
PRODUCT_ADDITION_SCALAR_REVERSE(-);

#define INSTANTIATE_PRODUCT_LHCOMPOSITE_OPS(HandlerTy)                         \
                                                                               \
  template product_op<HandlerTy> operator*(scalar_operator &&other,            \
                                           const product_op<HandlerTy> &self); \
  template product_op<HandlerTy> operator*(scalar_operator &&other,            \
                                           product_op<HandlerTy> &&self);      \
  template product_op<HandlerTy> operator*(const scalar_operator &other,       \
                                           const product_op<HandlerTy> &self); \
  template product_op<HandlerTy> operator*(const scalar_operator &other,       \
                                           product_op<HandlerTy> &&self);      \
  template sum_op<HandlerTy> operator+(scalar_operator &&other,                \
                                       const product_op<HandlerTy> &self);     \
  template sum_op<HandlerTy> operator+(scalar_operator &&other,                \
                                       product_op<HandlerTy> &&self);          \
  template sum_op<HandlerTy> operator+(const scalar_operator &other,           \
                                       const product_op<HandlerTy> &self);     \
  template sum_op<HandlerTy> operator+(const scalar_operator &other,           \
                                       product_op<HandlerTy> &&self);          \
  template sum_op<HandlerTy> operator-(scalar_operator &&other,                \
                                       const product_op<HandlerTy> &self);     \
  template sum_op<HandlerTy> operator-(scalar_operator &&other,                \
                                       product_op<HandlerTy> &&self);          \
  template sum_op<HandlerTy> operator-(const scalar_operator &other,           \
                                       const product_op<HandlerTy> &self);     \
  template sum_op<HandlerTy> operator-(const scalar_operator &other,           \
                                       product_op<HandlerTy> &&self);

INSTANTIATE_PRODUCT_LHCOMPOSITE_OPS(matrix_handler);
INSTANTIATE_PRODUCT_LHCOMPOSITE_OPS(spin_handler);
INSTANTIATE_PRODUCT_LHCOMPOSITE_OPS(boson_handler);
INSTANTIATE_PRODUCT_LHCOMPOSITE_OPS(fermion_handler);

// arithmetics that require conversions

#define PRODUCT_CONVERSIONS_OPS(op, returnTy)                                  \
  template <typename LHtype, typename RHtype,                                  \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype)>                        \
  returnTy<matrix_handler> operator op(const product_op<LHtype> &other,        \
                                       const product_op<RHtype> &self) {       \
    return product_op<matrix_handler>(other) op self;                          \
  }

PRODUCT_CONVERSIONS_OPS(*, product_op);
PRODUCT_CONVERSIONS_OPS(+, sum_op);
PRODUCT_CONVERSIONS_OPS(-, sum_op);

#define INSTANTIATE_PRODUCT_CONVERSION_OPS(op, returnTy)                       \
                                                                               \
  template returnTy<matrix_handler> operator op(                               \
      const product_op<spin_handler> &other,                                   \
      const product_op<matrix_handler> &self);                                 \
  template returnTy<matrix_handler> operator op(                               \
      const product_op<boson_handler> &other,                                  \
      const product_op<matrix_handler> &self);                                 \
  template returnTy<matrix_handler> operator op(                               \
      const product_op<fermion_handler> &other,                                \
      const product_op<matrix_handler> &self);                                 \
  template returnTy<matrix_handler> operator op(                               \
      const product_op<spin_handler> &other,                                   \
      const product_op<boson_handler> &self);                                  \
  template returnTy<matrix_handler> operator op(                               \
      const product_op<boson_handler> &other,                                  \
      const product_op<spin_handler> &self);                                   \
  template returnTy<matrix_handler> operator op(                               \
      const product_op<spin_handler> &other,                                   \
      const product_op<fermion_handler> &self);                                \
  template returnTy<matrix_handler> operator op(                               \
      const product_op<fermion_handler> &other,                                \
      const product_op<spin_handler> &self);                                   \
  template returnTy<matrix_handler> operator op(                               \
      const product_op<boson_handler> &other,                                  \
      const product_op<fermion_handler> &self);                                \
  template returnTy<matrix_handler> operator op(                               \
      const product_op<fermion_handler> &other,                                \
      const product_op<boson_handler> &self);

INSTANTIATE_PRODUCT_CONVERSION_OPS(*, product_op);
INSTANTIATE_PRODUCT_CONVERSION_OPS(+, sum_op);
INSTANTIATE_PRODUCT_CONVERSION_OPS(-, sum_op);

// general utility functions

template <typename HandlerTy>
bool product_op<HandlerTy>::is_identity() const {
  for (const auto &op : *this) {
    if (op != HandlerTy(op.degree))
      return false;
  }
  return true;
}

template <>
bool product_op<matrix_handler>::is_identity() const {
  for (const auto &op : *this) {
    auto degrees = op.degrees();
    if (degrees.size() != 1 || op != matrix_handler(degrees[0]))
      return false;
  }
  return true;
}

template <typename HandlerTy>
void product_op<HandlerTy>::dump() const {
  auto str = to_string();
  std::cout << str;
}

#if !defined(__clang__)
template bool product_op<matrix_handler>::is_identity() const;
template bool product_op<spin_handler>::is_identity() const;
template bool product_op<boson_handler>::is_identity() const;
template bool product_op<fermion_handler>::is_identity() const;
template void product_op<matrix_handler>::dump() const;
template void product_op<spin_handler>::dump() const;
template void product_op<boson_handler>::dump() const;
template void product_op<fermion_handler>::dump() const;
#endif

// handler specific utility functions

#define HANDLER_SPECIFIC_TEMPLATE_DEFINITION(ConcreteTy)                       \
  template <typename HandlerTy>                                                \
  template <typename T,                                                        \
            std::enable_if_t<std::is_same<T, ConcreteTy>::value &&             \
                                 std::is_same<HandlerTy, T>::value,            \
                             bool>>

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
std::size_t product_op<HandlerTy>::num_qubits() const {
  return this->degrees(false).size();
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
std::string product_op<HandlerTy>::get_pauli_word() const {
  // No padding here (only covers the operators we have),
  // and does not include the coefficient
  std::unordered_map<int, int> dims;
  auto terms = std::move(
      this->evaluate(
              operator_arithmetics<operator_handler::canonical_evaluation>(dims,
                                                                           {}))
          .terms);
  assert(terms.size() == 1);
  auto str = std::move(terms[0].second);
  if (operator_handler::canonical_order(1, 0) !=
      operator_handler::user_facing_order(1, 0))
    std::reverse(str.begin(), str.end());
  return str;
}

HANDLER_SPECIFIC_TEMPLATE_DEFINITION(spin_handler)
std::vector<bool> product_op<HandlerTy>::get_binary_symplectic_form() const {
  if (this->operators.size() == 0)
    return {};

  std::unordered_map<int, int> dims;
  auto degrees = this->degrees(
      false); // degrees in canonical order to match the evaluation
  auto evaluated = this->evaluate(
      operator_arithmetics<operator_handler::canonical_evaluation>(dims, {}));

  std::size_t max_degree =
      operator_handler::canonical_order(0, 1) ? degrees.back() : degrees[0];
  std::size_t term_size = max_degree + 1;
  assert(evaluated.terms.size() == 1);
  auto term = std::move(evaluated.terms[0]);

  // For compatiblity with existing code, the binary symplectic representation
  // needs to be from smallest to largest degree, and it necessarily must
  // include all consecutive degrees starting from 0 (even if the operator
  // doesn't act on them).
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
  return bsf; // always little endian order by definition of the bsf
}

#if !defined(__clang__)
template std::size_t product_op<spin_handler>::num_qubits() const;
template std::string product_op<spin_handler>::get_pauli_word() const;
template std::vector<bool>
product_op<spin_handler>::get_binary_symplectic_form() const;
#endif

// utility functions for backwards compatibility

#define SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION                             \
  template <typename HandlerTy>                                                \
  template <typename T,                                                        \
            std::enable_if_t<std::is_same<HandlerTy, spin_handler>::value &&   \
                                 std::is_same<HandlerTy, T>::value,            \
                             bool>>

SPIN_OPS_BACKWARD_COMPATIBILITY_DEFINITION
std::string product_op<HandlerTy>::to_string(bool printCoeffs) const {
  return sum_op(*this).to_string(printCoeffs);
}

#if !defined(__clang__)
template std::string
product_op<spin_handler>::to_string(bool printCoeffs) const;
#endif

#if defined(CUDAQ_INSTANTIATE_TEMPLATES)
template class product_op<matrix_handler>;
template class product_op<spin_handler>;
template class product_op<boson_handler>;
template class product_op<fermion_handler>;
#endif

} // namespace cudaq