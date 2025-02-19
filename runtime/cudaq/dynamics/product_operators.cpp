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
#include <unordered_map>
#include <type_traits>

#include "cudaq/operators.h"
#include "helpers.h"
#include "manipulation.h"
#include "matrix_operators.h"
#include "spin_operators.h"
#include "boson_operators.h"

namespace cudaq {

// private methods

#if !defined(NDEBUG)
// check canonicalization by default, individual handlers can set it to false to disable the check
bool operator_handler::can_be_canonicalized = true;

// returns true if and only if applying the operators in sequence acts only once on each degree of freedom and in canonical order
template <typename HandlerTy>
bool product_operator<HandlerTy>::is_canonicalized() const {
  auto canon_degrees = this->degrees();
  std::vector<int> degrees;
  degrees.reserve(canon_degrees.size());
  for (const auto &op : this->operators) {
    for (auto d : op.degrees())
      degrees.push_back(d);
  }
  return degrees == canon_degrees;
}
#endif

template<typename HandlerTy>
std::vector<HandlerTy>::const_iterator product_operator<HandlerTy>::find_insert_at(const HandlerTy &other) const {
  // the logic below just ensures that terms are fully or partially ordered in canonical order -
  // a best effort is made to order terms, but a full canonical ordering is not possible for certain handlers
  return std::find_if(this->operators.crbegin(), this->operators.crend(), 
              [other_target = other.target] (const HandlerTy& self_op) { 
    return other_target <= self_op.target; // FIXME: relies on canonical order
  }).base(); // base causes insert after for reverse iterator
}

template<>
std::vector<matrix_operator>::const_iterator product_operator<matrix_operator>::find_insert_at(const matrix_operator &other) const {
  // the logic below just ensures that terms are fully or partially ordered in canonical order -
  // a best effort is made to order terms, but a full canonical ordering is not possible for certain handlers
  return std::find_if(this->operators.crbegin(), this->operators.crend(), 
              [&other_degrees = static_cast<const std::vector<int>&>(other.degrees())] 
              (const matrix_operator& self_op) { 
    const std::vector<int> &self_op_degrees = self_op.degrees();
    for (auto other_degree : other_degrees) {
      auto item_it = std::find_if(self_op_degrees.crbegin(), self_op_degrees.crend(), 
        [other_degree](int self_degree) { return other_degree <= self_degree; }); // FIXME: depends on canonical order (otherwise matrix computation is rather inefficient)
      if (item_it != self_op_degrees.crend()) return true;
    }
    return false;
  }).base(); // base causes insert after for reverse iterator
}

template<typename HandlerTy>
template<typename T, std::enable_if_t<std::is_same<HandlerTy, T>::value && 
                                      !product_operator<T>::supports_inplace_mult, std::false_type>>
void product_operator<HandlerTy>::insert(T &&other) {  
  auto pos = this->find_insert_at(other);
  this->operators.insert(pos, other);
}

template<typename HandlerTy>
template<typename T, std::enable_if_t<std::is_same<HandlerTy, T>::value && 
                                      product_operator<T>::supports_inplace_mult, std::true_type>>
void product_operator<HandlerTy>::insert(T &&other) {
  auto pos = this->find_insert_at(other);
  if (pos != this->operators.begin() && (pos - 1)->target == other.target) {
    auto it = this->operators.erase(pos - 1, pos - 1); // erase: constant time conversion to non-const iterator
    it->inplace_mult(other);
  }
  else this->operators.insert(pos, std::move(other));
}

template<>
template<typename T, std::enable_if_t<std::is_same<spin_operator, T>::value && 
                                      product_operator<T>::supports_inplace_mult, std::true_type>>
void product_operator<spin_operator>::insert(T &&other) {
  auto pos = this->find_insert_at(other);
  if (pos != this->operators.begin() && (pos - 1)->target == other.target) {
    auto it = this->operators.erase(pos - 1, pos - 1); // erase: constant time conversion to non-const iterator
    this->coefficient *= it->inplace_mult(other);
  }
  else this->operators.insert(pos, std::move(other));
}

template<typename HandlerTy>
std::string product_operator<HandlerTy>::get_term_id() const {
  std::string term_id;
  for (const auto &op : this->operators)
    term_id += op.unique_id();
  return std::move(term_id);
}

template<typename HandlerTy>
void product_operator<HandlerTy>::aggregate_terms() {}

template<typename HandlerTy>
template<typename... Args>
void product_operator<HandlerTy>::aggregate_terms(HandlerTy &&head, Args&& ... args) {
  this->insert(std::forward<HandlerTy>(head));
  aggregate_terms(std::forward<Args>(args)...);
}

// FIXME: EVALUATE IS NOT SUPPOSED TO RETURN A MATRIX - 
// IT SUPPOSED TO TAKE A TRANSFORMATION (ANY OPERATOR ARITHMETICS) AND APPLY IT
template <typename HandlerTy>
EvaluatedMatrix product_operator<HandlerTy>::m_evaluate(
    MatrixArithmetics arithmetics, bool pad_terms) const {
  auto degrees = this->degrees();
  cudaq::matrix_2 result;

  auto padded_op = [&arithmetics, &degrees = std::as_const(degrees)](const HandlerTy &op) {
      std::vector<EvaluatedMatrix> padded;
      auto op_degrees = op.degrees();
      for (const auto &degree : degrees) {
        if (std::find(op_degrees.cbegin(), op_degrees.cend(), degree) == op_degrees.cend()) {
          auto identity = HandlerTy(degree);
          padded.push_back(EvaluatedMatrix(identity.degrees(), identity.to_matrix(arithmetics.m_dimensions)));
        }
      }
      /// Creating the tensor product with op being last is most efficient.
      if (padded.size() == 0)
        return EvaluatedMatrix(op_degrees, op.to_matrix(arithmetics.m_dimensions, arithmetics.m_parameters));
      EvaluatedMatrix ids(std::move(padded[0]));
      for (auto i = 1; i < padded.size(); ++i)
        ids = arithmetics.tensor(std::move(ids), std::move(padded[i]));
      return arithmetics.tensor(std::move(ids), EvaluatedMatrix(op_degrees, op.to_matrix(arithmetics.m_dimensions, arithmetics.m_parameters)));
  };

  auto coefficient = this->coefficient.evaluate(arithmetics.m_parameters);
  if (this->operators.size() > 0) {
    if (pad_terms) {
      EvaluatedMatrix prod = padded_op(this->operators[0]);
      for (auto op_idx = 1; op_idx < this->operators.size(); ++op_idx) {
        auto op_degrees = this->operators[op_idx].degrees();
        if (op_degrees.size() != 1 || this->operators[op_idx] != HandlerTy(op_degrees[0]))
          prod = arithmetics.mul(std::move(prod), padded_op(this->operators[op_idx]));
      }
      return EvaluatedMatrix(std::move(prod.degrees()), coefficient * prod.matrix());
    } else {
      EvaluatedMatrix prod(this->operators[0].degrees(), this->operators[0].to_matrix(arithmetics.m_dimensions, arithmetics.m_parameters));
      for (auto op_idx = 1; op_idx < this->operators.size(); ++op_idx) {
        auto mat = this->operators[op_idx].to_matrix(arithmetics.m_dimensions, arithmetics.m_parameters);
        prod = arithmetics.mul(std::move(prod), EvaluatedMatrix(this->operators[op_idx].degrees(), mat));
      }
      return EvaluatedMatrix(std::move(prod.degrees()), coefficient * prod.matrix());
    }
  } else {
    assert(degrees.size() == 0); // degrees are stored with each term
    return EvaluatedMatrix({}, coefficient * cudaq::matrix_2::identity(1));
  }
}

#define INSTANTIATE_PRODUCT_PRIVATE_METHODS(HandlerTy)                                        \
                                                                                              \
  template                                                                                    \
  void product_operator<HandlerTy>::aggregate_terms(HandlerTy &&item1,                        \
                                                    HandlerTy &&item2);                       \
                                                                                              \
  template                                                                                    \
  void product_operator<HandlerTy>::aggregate_terms(HandlerTy &&item1,                        \
                                                    HandlerTy &&item2,                        \
                                                    HandlerTy &&item3);                       \
                                                                                              \
  template                                                                                    \
  EvaluatedMatrix product_operator<HandlerTy>::m_evaluate(                                    \
      MatrixArithmetics arithmetics, bool pad_terms) const;

INSTANTIATE_PRODUCT_PRIVATE_METHODS(matrix_operator);
INSTANTIATE_PRODUCT_PRIVATE_METHODS(spin_operator);
INSTANTIATE_PRODUCT_PRIVATE_METHODS(boson_operator);

// read-only properties

template<typename HandlerTy>
std::vector<int> product_operator<HandlerTy>::degrees() const {
  std::set<int> unsorted_degrees;
  for (const HandlerTy &term : this->operators) {
    auto term_degrees = term.degrees();
    unsorted_degrees.insert(term_degrees.cbegin(), term_degrees.cend());
  }
  auto degrees = std::vector<int>(unsorted_degrees.cbegin(), unsorted_degrees.cend());
  cudaq::detail::canonicalize_degrees(degrees);
  return std::move(degrees);
}

template<typename HandlerTy>
int product_operator<HandlerTy>::num_terms() const { 
  return this->operators.size(); 
}

template<typename HandlerTy>
const std::vector<HandlerTy>& product_operator<HandlerTy>::get_terms() const { 
  return this->operators; 
}

template<typename HandlerTy>
scalar_operator product_operator<HandlerTy>::get_coefficient() const { 
  return this->coefficient; 
}

#define INSTANTIATE_PRODUCT_PROPERTIES(HandlerTy)                                            \
                                                                                             \
  template                                                                                   \
  std::vector<int> product_operator<HandlerTy>::degrees() const;                             \
                                                                                             \
  template                                                                                   \
  int product_operator<HandlerTy>::num_terms() const;                                          \
                                                                                             \
  template                                                                                   \
  const std::vector<HandlerTy>& product_operator<HandlerTy>::get_terms() const;              \
                                                                                             \
  template                                                                                   \
  scalar_operator product_operator<HandlerTy>::get_coefficient() const;

INSTANTIATE_PRODUCT_PROPERTIES(matrix_operator);
INSTANTIATE_PRODUCT_PROPERTIES(spin_operator);
INSTANTIATE_PRODUCT_PROPERTIES(boson_operator);

// constructors

template<typename HandlerTy>
product_operator<HandlerTy>::product_operator(double coefficient)
  : coefficient(coefficient) {}

template<typename HandlerTy>
product_operator<HandlerTy>::product_operator(HandlerTy &&atomic)
  : coefficient(1.) {
  this->operators.push_back(std::move(atomic));
  assert (!HandlerTy::can_be_canonicalized || this->is_canonicalized()); // relevant for custom matrix operators acting on multiple degrees of freedom
}

template<typename HandlerTy>
template<typename... Args, std::enable_if_t<std::conjunction<std::is_same<HandlerTy, Args>...>::value, bool>>
product_operator<HandlerTy>::product_operator(scalar_operator coefficient, Args&&... args)
  : coefficient(std::move(coefficient)) {
  this->operators.reserve(sizeof...(Args));
  aggregate_terms(std::forward<HandlerTy &&>(args)...);
  assert (!HandlerTy::can_be_canonicalized || this->is_canonicalized());
}

// assumes canonical ordering (if possible)
template<typename HandlerTy>
product_operator<HandlerTy>::product_operator(scalar_operator coefficient, const std::vector<HandlerTy> &atomic_operators, int size) 
  : coefficient(std::move(coefficient)) { 
  if (size <= 0) this->operators = atomic_operators;
  else {
    this->operators.reserve(size);
    for (const auto &op : atomic_operators)
      this->operators.push_back(op);
  }
  assert (!HandlerTy::can_be_canonicalized || this->is_canonicalized());
}

// assumes canonical ordering (if possible)
template<typename HandlerTy>
product_operator<HandlerTy>::product_operator(scalar_operator coefficient, std::vector<HandlerTy> &&atomic_operators, int size)
  : coefficient(std::move(coefficient)), operators(std::move(atomic_operators)) {
  if (size > 0) this->operators.reserve(size);
  assert (!HandlerTy::can_be_canonicalized || this->is_canonicalized());
}

template<typename HandlerTy>
template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool>>
product_operator<HandlerTy>::product_operator(const product_operator<T> &other) 
  : coefficient(other.coefficient) {
  this->operators.reserve(other.operators.size());
  for (const T &other_op : other.operators) {
    HandlerTy op(other_op);
    this->operators.push_back(op);
  }
}

template<typename HandlerTy>
product_operator<HandlerTy>::product_operator(const product_operator<HandlerTy> &other, int size) 
  : coefficient(other.coefficient) {
  if (size <= 0) this->operators = other.operators;
  else {
    this->operators.reserve(size);
    for (const auto &op : other.operators)
      this->operators.push_back(op);
  }
}

template<typename HandlerTy>
product_operator<HandlerTy>::product_operator(product_operator<HandlerTy> &&other, int size) 
  : coefficient(std::move(other.coefficient)), operators(std::move(other.operators)) {
  if (size > 0) this->operators.reserve(size);
}

#define INSTANTIATE_PRODUCT_CONSTRUCTORS(HandlerTy)                                          \
                                                                                             \
  template                                                                                   \
  product_operator<HandlerTy>::product_operator(double coefficient);                         \
                                                                                             \
  template                                                                                   \
  product_operator<HandlerTy>::product_operator(scalar_operator coefficient);                \
                                                                                             \
  template                                                                                   \
  product_operator<HandlerTy>::product_operator(HandlerTy &&atomic);                         \
                                                                                             \
  template                                                                                   \
  product_operator<HandlerTy>::product_operator(scalar_operator coefficient,                 \
                                                HandlerTy &&atomic1);                        \
                                                                                             \
  template                                                                                   \
  product_operator<HandlerTy>::product_operator(scalar_operator coefficient,                 \
                                                HandlerTy &&atomic1,                         \
                                                HandlerTy &&atomic2);                        \
                                                                                             \
  template                                                                                   \
  product_operator<HandlerTy>::product_operator(scalar_operator coefficient,                 \
                                                HandlerTy &&atomic1,                         \
                                                HandlerTy &&atomic2,                         \
                                                HandlerTy &&atomic3);                        \
                                                                                             \
  template                                                                                   \
  product_operator<HandlerTy>::product_operator(scalar_operator coefficient,                 \
    const std::vector<HandlerTy> &atomic_operators, int size);                               \
                                                                                             \
  template                                                                                   \
  product_operator<HandlerTy>::product_operator(scalar_operator coefficient,                 \
    std::vector<HandlerTy> &&atomic_operators, int size);                                    \
                                                                                             \
  template                                                                                   \
  product_operator<HandlerTy>::product_operator(                                             \
    const product_operator<HandlerTy> &other, int size);                                     \
                                                                                             \
  template                                                                                   \
  product_operator<HandlerTy>::product_operator(                                             \
    product_operator<HandlerTy> &&other, int size);

template 
product_operator<matrix_operator>::product_operator(const product_operator<spin_operator> &other);
template 
product_operator<matrix_operator>::product_operator(const product_operator<boson_operator> &other);

INSTANTIATE_PRODUCT_CONSTRUCTORS(matrix_operator);
INSTANTIATE_PRODUCT_CONSTRUCTORS(spin_operator);
INSTANTIATE_PRODUCT_CONSTRUCTORS(boson_operator);

// assignments

template<typename HandlerTy>
template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool>>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator=(const product_operator<T> &other) {
  *this = product_operator<HandlerTy>(other);
  return *this;
}

template<typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator=(const product_operator<HandlerTy> &other) {
  if (this != &other) {
    this->coefficient = other.coefficient;
    this->operators = other.operators;
  }
  return *this;
}

template<typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator=(product_operator<HandlerTy> &&other) {
  if (this != &other) {
    this->coefficient = std::move(other.coefficient);
    this->operators = std::move(other.operators);
  }
  return *this;
}

#define INSTANTIATE_PRODUCT_ASSIGNMENTS(HandlerTy)                                          \
                                                                                            \
  template                                                                                  \
  product_operator<HandlerTy>& product_operator<HandlerTy>::operator=(                      \
    const product_operator<HandlerTy> &other);                                              \
                                                                                            \
  template                                                                                  \
  product_operator<HandlerTy>& product_operator<HandlerTy>::operator=(                      \
    product_operator<HandlerTy> &&other);

template 
product_operator<matrix_operator>& product_operator<matrix_operator>::operator=(const product_operator<spin_operator> &other);
template 
product_operator<matrix_operator>& product_operator<matrix_operator>::operator=(const product_operator<boson_operator> &other);

INSTANTIATE_PRODUCT_ASSIGNMENTS(matrix_operator);
INSTANTIATE_PRODUCT_ASSIGNMENTS(spin_operator);
INSTANTIATE_PRODUCT_ASSIGNMENTS(boson_operator);

// evaluations

template<typename HandlerTy>
std::string product_operator<HandlerTy>::to_string() const {
  auto str = "(" + this->coefficient.to_string() + ") * ";
  for (const auto &op : this->operators)
    str += op.to_string(true);
  return std::move(str);
}

template<typename HandlerTy>
matrix_2 product_operator<HandlerTy>::to_matrix(std::unordered_map<int, int> dimensions,
                                                const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  return std::move(this->m_evaluate(MatrixArithmetics(dimensions, parameters)).matrix());
}

#define INSTANTIATE_PRODUCT_EVALUATIONS(HandlerTy)                                          \
                                                                                            \
  template                                                                                  \
  std::string product_operator<HandlerTy>::to_string() const;                               \
                                                                                            \
  template                                                                                  \
  matrix_2 product_operator<HandlerTy>::to_matrix(                                          \
    std::unordered_map<int, int> dimensions,                                               \
    const std::unordered_map<std::string, std::complex<double>> &parameters) const;

INSTANTIATE_PRODUCT_EVALUATIONS(matrix_operator);
INSTANTIATE_PRODUCT_EVALUATIONS(spin_operator);
INSTANTIATE_PRODUCT_EVALUATIONS(boson_operator);

// comparisons

template<typename HandlerTy>
bool product_operator<HandlerTy>::operator==(const product_operator<HandlerTy> &other) const {
  return this->coefficient == other.coefficient && this->get_term_id() == other.get_term_id();
}

#define INSTANTIATE_PRODUCT_COMPARISONS(HandlerTy)                                          \
                                                                                            \
  template                                                                                  \
  bool product_operator<HandlerTy>::operator==(                                             \
    const product_operator<HandlerTy> &other) const;

INSTANTIATE_PRODUCT_COMPARISONS(matrix_operator);
INSTANTIATE_PRODUCT_COMPARISONS(spin_operator);
INSTANTIATE_PRODUCT_COMPARISONS(boson_operator);

// unary operators

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator-() const & {
  return product_operator<HandlerTy>(-1. * this->coefficient, this->operators);
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator-() && {
  this->coefficient *= -1.;
  return std::move(*this);
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator+() const & {
  return *this;
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator+() && {
  return std::move(*this);
}

#define INSTANTIATE_PRODUCT_UNARY_OPS(HandlerTy)                                            \
  template                                                                                  \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator-() const &;             \
  template                                                                                  \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator-() &&;                  \
  template                                                                                  \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator+() const &;             \
  template                                                                                  \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator+() &&;

INSTANTIATE_PRODUCT_UNARY_OPS(matrix_operator);
INSTANTIATE_PRODUCT_UNARY_OPS(spin_operator);
INSTANTIATE_PRODUCT_UNARY_OPS(boson_operator);

// right-hand arithmetics

#define PRODUCT_MULTIPLICATION(otherTy)                                                 \
                                                                                        \
  template <typename HandlerTy>                                                         \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator*(                   \
                                                           otherTy other) const & {     \
    return product_operator<HandlerTy>(other * this->coefficient,                       \
                                       this->operators);                                \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator*(                   \
                                                           otherTy other) && {          \
    this->coefficient *= other;                                                         \
    return std::move(*this);                                                            \
  }                                                                                     \

PRODUCT_MULTIPLICATION(double);
PRODUCT_MULTIPLICATION(std::complex<double>);
PRODUCT_MULTIPLICATION(const scalar_operator &);

#define PRODUCT_ADDITION(otherTy, op)                                                   \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator op(                     \
                                                       otherTy other) const & {         \
    return operator_sum<HandlerTy>(product_operator<HandlerTy>(op other),               \
                                   product_operator<HandlerTy>(*this));                 \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator op(                     \
                                                       otherTy other) && {              \
    return operator_sum<HandlerTy>(product_operator<HandlerTy>(op other),               \
                                   std::move(*this));                                   \
  }                                                                                     \

PRODUCT_ADDITION(double, +);
PRODUCT_ADDITION(double, -);
PRODUCT_ADDITION(std::complex<double>, +);
PRODUCT_ADDITION(std::complex<double>, -);
PRODUCT_ADDITION(const scalar_operator &, +);
PRODUCT_ADDITION(const scalar_operator &, -);

#define INSTANTIATE_PRODUCT_RHSIMPLE_OPS(HandlerTy)                                                         \
                                                                                                            \
  template                                                                                                  \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator*(double other) const &;                 \
  template                                                                                                  \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator*(double other) &&;                      \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(double other) const &;                     \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(double other) &&;                          \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(double other) const &;                     \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(double other) &&;                          \
  template                                                                                                  \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator*(std::complex<double> other) const &;   \
  template                                                                                                  \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator*(std::complex<double> other) &&;        \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(std::complex<double> other) const &;       \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(std::complex<double> other) &&;            \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(std::complex<double> other) const &;       \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(std::complex<double> other) &&;            \
  template                                                                                                  \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator*(const scalar_operator &other) const &; \
  template                                                                                                  \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator*(const scalar_operator &other) &&;      \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(const scalar_operator &other) const &;     \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(const scalar_operator &other) &&;          \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(const scalar_operator &other) const &;     \
  template                                                                                                  \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(const scalar_operator &other) &&;          \

INSTANTIATE_PRODUCT_RHSIMPLE_OPS(matrix_operator);
INSTANTIATE_PRODUCT_RHSIMPLE_OPS(spin_operator);
INSTANTIATE_PRODUCT_RHSIMPLE_OPS(boson_operator);

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator*(const product_operator<HandlerTy> &other) const & {
  product_operator<HandlerTy> prod(this->coefficient * other.coefficient, this->operators,
                                   this->operators.size() + other.operators.size());
  for (HandlerTy op : other.operators)
    prod.insert(std::move(op));
  return std::move(prod);
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator*(const product_operator<HandlerTy> &other) && {
  this->coefficient *= other.coefficient;
  this->operators.reserve(this->operators.size() + other.operators.size());
  for (HandlerTy op : other.operators)
    this->insert(std::move(op));
  return std::move(*this);
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator*(product_operator<HandlerTy> &&other) const & {
  product_operator<HandlerTy> prod(this->coefficient * std::move(other.coefficient), 
    this->operators, this->operators.size() + other.operators.size());
  for (auto &&op : other.operators)
    prod.insert(std::move(op));
  return std::move(prod);
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator*(product_operator<HandlerTy> &&other) && {
  this->coefficient *= std::move(other.coefficient);
  this->operators.reserve(this->operators.size() + other.operators.size());
  for (auto &&op : other.operators)
    this->insert(std::move(op));
  return std::move(*this);
}

#define PRODUCT_ADDITION_PRODUCT(op)                                                    \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator op(                     \
                                   const product_operator<HandlerTy> &other) const & {  \
    return operator_sum<HandlerTy>(product_operator<HandlerTy>(*this), op other);       \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator op(                     \
                                     const product_operator<HandlerTy> &other) && {     \
    return operator_sum<HandlerTy>(std::move(*this), op other);                         \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator op(                     \
                                        product_operator<HandlerTy> &&other) const & {  \
    return operator_sum<HandlerTy>(product_operator<HandlerTy>(*this),                  \
                                   op std::move(other));                                \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator op(                     \
                                          product_operator<HandlerTy> &&other) && {     \
    return operator_sum<HandlerTy>(std::move(*this), op std::move(other));              \
  }                                                                                     \

PRODUCT_ADDITION_PRODUCT(+)
PRODUCT_ADDITION_PRODUCT(-)

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator*(const operator_sum<HandlerTy> &other) const {
  operator_sum<HandlerTy> sum; // everything needs to be updated, so creating a new sum makes sense
  sum.coefficients.reserve(other.coefficients.size());
  sum.term_map.reserve(other.terms.size());
  sum.terms.reserve(other.terms.size());
  for (auto i = 0; i < other.terms.size(); ++i) {
    auto prod = *this * product_operator<HandlerTy>(other.coefficients[i], other.terms[i]);
    sum.insert(std::move(prod));
  }
  return std::move(sum);
}

#define PRODUCT_ADDITION_SUM(op)                                                        \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator op(                     \
                                     const operator_sum<HandlerTy> &other) const & {    \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients.reserve(other.coefficients.size() + 1);                            \
    sum.term_map = other.term_map;                                                      \
    sum.terms = other.terms;                                                            \
    for (auto &coeff : other.coefficients)                                              \
      sum.coefficients.push_back(op coeff);                                             \
    sum.insert(*this);                                                                  \
    return std::move(sum);                                                              \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator op(                     \
                                     const operator_sum<HandlerTy> &other) && {         \
    operator_sum<HandlerTy> sum;                                                        \
    sum.coefficients.reserve(other.coefficients.size() + 1);                            \
    sum.term_map = other.term_map;                                                      \
    sum.terms = other.terms;                                                            \
    for (auto &coeff : other.coefficients)                                              \
      sum.coefficients.push_back(op coeff);                                             \
    sum.insert(std::move(*this));                                                       \
    return std::move(sum);                                                              \
  }                                                                                     \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator op(                     \
                                          operator_sum<HandlerTy> &&other) const & {    \
    operator_sum<HandlerTy> sum(op std::move(other));                                   \
    sum.insert(*this);                                                                  \
    return std::move(sum);                                                              \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator op(                     \
                                          operator_sum<HandlerTy> &&other) && {         \
    operator_sum<HandlerTy> sum(op std::move(other));                                   \
    sum.insert(std::move(*this));                                                       \
    return std::move(sum);                                                              \
  }                                                                                     \

PRODUCT_ADDITION_SUM(+)
PRODUCT_ADDITION_SUM(-)

#define INSTANTIATE_PRODUCT_RHCOMPOSITE_OPS(HandlerTy)                                  \
                                                                                        \
  template                                                                              \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator*(                   \
    const product_operator<HandlerTy> &other) const &;                                  \
  template                                                                              \
  product_operator<HandlerTy> product_operator<HandlerTy>::operator*(                   \
    const product_operator<HandlerTy> &other) &&;                                       \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(                       \
    const product_operator<HandlerTy> &other) const &;                                  \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(                       \
    const product_operator<HandlerTy> &other) &&;                                       \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(                       \
    product_operator<HandlerTy> &&other) const &;                                       \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(                       \
    product_operator<HandlerTy> &&other) &&;                                            \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(                       \
    const product_operator<HandlerTy> &other) const &;                                  \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(                       \
    const product_operator<HandlerTy> &other) &&;                                       \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(                       \
    product_operator<HandlerTy> &&other) const &;                                       \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(                       \
    product_operator<HandlerTy> &&other) &&;                                            \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator*(                       \
    const operator_sum<HandlerTy> &other) const;                                        \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(                       \
    const operator_sum<HandlerTy> &other) const &;                                      \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(                       \
    const operator_sum<HandlerTy> &other) &&;                                           \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(                       \
    operator_sum<HandlerTy> &&other) const &;                                           \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(                       \
    operator_sum<HandlerTy> &&other) &&;                                                \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(                       \
    const operator_sum<HandlerTy> &other) const &;                                      \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(                       \
    const operator_sum<HandlerTy> &other) &&;                                           \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(                       \
    operator_sum<HandlerTy> &&other) const &;                                           \
  template                                                                              \
  operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(                       \
    operator_sum<HandlerTy> &&other) &&;                                                \

INSTANTIATE_PRODUCT_RHCOMPOSITE_OPS(matrix_operator);
INSTANTIATE_PRODUCT_RHCOMPOSITE_OPS(spin_operator);
INSTANTIATE_PRODUCT_RHCOMPOSITE_OPS(boson_operator);

#define PRODUCT_MULTIPLICATION_ASSIGNMENT(otherTy)                                      \
  template <typename HandlerTy>                                                         \
  product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(otherTy other) { \
    this->coefficient *= other;                                                         \
    return *this;                                                                       \
  }

PRODUCT_MULTIPLICATION_ASSIGNMENT(double);
PRODUCT_MULTIPLICATION_ASSIGNMENT(std::complex<double>);
PRODUCT_MULTIPLICATION_ASSIGNMENT(const scalar_operator &);

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(const product_operator<HandlerTy> &other) {
  this->coefficient *= other.coefficient;
  this->operators.reserve(this->operators.size() + other.operators.size());
  for (HandlerTy op : other.operators)
    this->insert(std::move(op));
  return *this;
}

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(product_operator<HandlerTy> &&other) {
  this->coefficient *= std::move(other.coefficient);
  this->operators.reserve(this->operators.size() + other.operators.size());
  for (auto &&op : other.operators)
    this->insert(std::move(op));
  return *this;
}

#define INSTANTIATE_PRODUCT_OPASSIGNMENTS(HandlerTy)                                                              \
                                                                                                                  \
  template                                                                                                        \
  product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(double other);                             \
  template                                                                                                        \
  product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(std::complex<double> other);               \
  template                                                                                                        \
  product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(const scalar_operator &other);             \
  template                                                                                                        \
  product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(const product_operator<HandlerTy> &other); \
  template                                                                                                        \
  product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(product_operator<HandlerTy> &&other);      \

INSTANTIATE_PRODUCT_OPASSIGNMENTS(matrix_operator);
INSTANTIATE_PRODUCT_OPASSIGNMENTS(spin_operator);
INSTANTIATE_PRODUCT_OPASSIGNMENTS(boson_operator);

// left-hand arithmetics

#define PRODUCT_MULTIPLICATION_REVERSE(otherTy)                                         \
                                                                                        \
  template <typename HandlerTy>                                                         \
  product_operator<HandlerTy> operator*(otherTy other,                                  \
                                        const product_operator<HandlerTy> &self) {      \
    return product_operator<HandlerTy>(other * self.coefficient,                        \
                                       self.operators);                                 \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  product_operator<HandlerTy> operator*(otherTy other,                                  \
                                        product_operator<HandlerTy> &&self) {           \
    self.coefficient *= other;                                                          \
    return std::move(self);                                                             \
  }                                                                                     \

PRODUCT_MULTIPLICATION_REVERSE(double);
PRODUCT_MULTIPLICATION_REVERSE(std::complex<double>);
PRODUCT_MULTIPLICATION_REVERSE(const scalar_operator &);

#define PRODUCT_ADDITION_REVERSE(otherTy, op)                                           \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator op(otherTy other,                                    \
                                      const product_operator<HandlerTy> &self) {        \
    return operator_sum<HandlerTy>(product_operator<HandlerTy>(other), op self);        \
  }                                                                                     \
                                                                                        \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator op(otherTy other,                                    \
                                      product_operator<HandlerTy> &&self) {             \
    return operator_sum<HandlerTy>(product_operator<HandlerTy>(other),                  \
                                   op std::move(self));                                 \
  }                                                                                     \

PRODUCT_ADDITION_REVERSE(double, +);
PRODUCT_ADDITION_REVERSE(double, -);
PRODUCT_ADDITION_REVERSE(std::complex<double>, +);
PRODUCT_ADDITION_REVERSE(std::complex<double>, -);
PRODUCT_ADDITION_REVERSE(const scalar_operator &, +);
PRODUCT_ADDITION_REVERSE(const scalar_operator &, -);

#define INSTANTIATE_PRODUCT_LHCOMPOSITE_OPS(HandlerTy)                                                              \
                                                                                                                    \
  template                                                                                                          \
  product_operator<HandlerTy> operator*(double other, const product_operator<HandlerTy> &self);                     \
  template                                                                                                          \
  product_operator<HandlerTy> operator*(double other, product_operator<HandlerTy> &&self);                          \
  template                                                                                                          \
  operator_sum<HandlerTy> operator+(double other, const product_operator<HandlerTy> &self);                         \
  template                                                                                                          \
  operator_sum<HandlerTy> operator+(double other, product_operator<HandlerTy> &&self);                              \
  template                                                                                                          \
  operator_sum<HandlerTy> operator-(double other, const product_operator<HandlerTy> &self);                         \
  template                                                                                                          \
  operator_sum<HandlerTy> operator-(double other, product_operator<HandlerTy> &&self);                              \
  template                                                                                                          \
  product_operator<HandlerTy> operator*(std::complex<double> other, const product_operator<HandlerTy> &self);       \
  template                                                                                                          \
  product_operator<HandlerTy> operator*(std::complex<double> other, product_operator<HandlerTy> &&self);            \
  template                                                                                                          \
  operator_sum<HandlerTy> operator+(std::complex<double> other, const product_operator<HandlerTy> &self);           \
  template                                                                                                          \
  operator_sum<HandlerTy> operator+(std::complex<double> other, product_operator<HandlerTy> &&self);                \
  template                                                                                                          \
  operator_sum<HandlerTy> operator-(std::complex<double> other, const product_operator<HandlerTy> &self);           \
  template                                                                                                          \
  operator_sum<HandlerTy> operator-(std::complex<double> other, product_operator<HandlerTy> &&self);                \
  template                                                                                                          \
  product_operator<HandlerTy> operator*(const scalar_operator &other, const product_operator<HandlerTy> &self);     \
  template                                                                                                          \
  product_operator<HandlerTy> operator*(const scalar_operator &other, product_operator<HandlerTy> &&self);          \
  template                                                                                                          \
  operator_sum<HandlerTy> operator+(const scalar_operator &other, const product_operator<HandlerTy> &self);         \
  template                                                                                                          \
  operator_sum<HandlerTy> operator+(const scalar_operator &other, product_operator<HandlerTy> &&self);              \
  template                                                                                                          \
  operator_sum<HandlerTy> operator-(const scalar_operator &other, const product_operator<HandlerTy> &self);         \
  template                                                                                                          \
  operator_sum<HandlerTy> operator-(const scalar_operator &other, product_operator<HandlerTy> &&self);              \

INSTANTIATE_PRODUCT_LHCOMPOSITE_OPS(matrix_operator);
INSTANTIATE_PRODUCT_LHCOMPOSITE_OPS(spin_operator);
INSTANTIATE_PRODUCT_LHCOMPOSITE_OPS(boson_operator);

// arithmetics that require conversions

#define PRODUCT_CONVERSIONS_OPS(op, returnTy)                                                 \
  template <typename LHtype, typename RHtype,                                                 \
            TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>                                \
  returnTy<matrix_operator> operator op(const product_operator<LHtype> &other,                \
                                        const product_operator<RHtype> &self) {               \
    return product_operator<matrix_operator>(other) op self;                                  \
  }

PRODUCT_CONVERSIONS_OPS(*, product_operator);
PRODUCT_CONVERSIONS_OPS(+, operator_sum);
PRODUCT_CONVERSIONS_OPS(-, operator_sum);

#define INSTANTIATE_PRODUCT_CONVERSION_OPS(op, returnTy)                                      \
                                                                                              \
  template                                                                                    \
  returnTy<matrix_operator> operator op(const product_operator<spin_operator> &other,         \
                                        const product_operator<matrix_operator> &self);       \
  template                                                                                    \
  returnTy<matrix_operator> operator op(const product_operator<boson_operator> &other,        \
                                        const product_operator<matrix_operator> &self);       \
  template                                                                                    \
  returnTy<matrix_operator> operator op(const product_operator<spin_operator> &other,         \
                                        const product_operator<boson_operator> &self);        \
  template                                                                                    \
  returnTy<matrix_operator> operator op(const product_operator<boson_operator> &other,        \
                                        const product_operator<spin_operator> &self);

INSTANTIATE_PRODUCT_CONVERSION_OPS(*, product_operator);
INSTANTIATE_PRODUCT_CONVERSION_OPS(+, operator_sum);
INSTANTIATE_PRODUCT_CONVERSION_OPS(-, operator_sum);

// common operators

template<typename HandlerTy, typename... Args, std::enable_if_t<std::conjunction<std::is_same<int, Args>...>::value, bool> = true>
product_operator<HandlerTy> operator_handler::identity(Args... targets) {
  static_assert (std::is_constructible_v<HandlerTy, int>, "operator handlers must have a constructor that take a single degree of freedom and returns the identity operator on that degree.");
  return product_operator<HandlerTy>(1.0, HandlerTy(targets)...);
}

template product_operator<matrix_operator> operator_handler::identity();
template product_operator<spin_operator> operator_handler::identity();
template product_operator<boson_operator> operator_handler::identity();

template product_operator<matrix_operator> operator_handler::identity(int target);
template product_operator<spin_operator> operator_handler::identity(int target);
template product_operator<boson_operator> operator_handler::identity(int target);


#ifdef CUDAQ_INSTANTIATE_TEMPLATES
template class product_operator<matrix_operator>;
template class product_operator<spin_operator>;
template class product_operator<boson_operator>;
#endif

} // namespace cudaq