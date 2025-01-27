/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "dynamics/template_declarations.h"
#include "definition.h"
#include "utils/tensor.h"

#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <concepts> 
#include <type_traits>

namespace cudaq {

class scalar_operator {

private:
  // If someone gave us a constant value, we will just return that
  // directly to them when they call `evaluate`.
  std::optional<std::complex<double>> m_constant_value;

  /// @brief The function that generates the value of the scalar operator.
  /// The function can take a vector of complex-valued arguments
  /// and returns a number.
  ScalarCallbackFunction generator;

public:
  scalar_operator(double value) 
    : m_constant_value(value), generator() {}

  /// @brief Constructor that just takes and returns a complex double value.
  /// @NOTE: This replicates the behavior of the python `scalar_operator::const`
  /// without the need for an extra member function.
  scalar_operator(std::complex<double> value) 
    : m_constant_value(value), generator() {}


  scalar_operator(const ScalarCallbackFunction &create) 
    : m_constant_value(), generator(create) {}

  /// @brief Constructor that just takes a callback function with no
  /// arguments.
  scalar_operator(ScalarCallbackFunction &&create)
    : m_constant_value() {
    generator = std::move(create);
  }

  // copy constructor
  scalar_operator(const scalar_operator &other) 
    : m_constant_value(other.m_constant_value), generator(other.generator) {}

  // move constructor
  scalar_operator(scalar_operator &&other) 
    : m_constant_value(other.m_constant_value) {
      generator = std::move(other.generator);
  }

  // assignment operator
  scalar_operator& operator=(const scalar_operator &other) {
    if (this != &other) {
      m_constant_value = other.m_constant_value;
      generator = other.generator;
    }
    return *this;
  }

  // move assignment operator
  scalar_operator& operator=(scalar_operator &&other) {
    if (this != &other) {
      m_constant_value = other.m_constant_value;
      generator = std::move(other.generator);
    }
    return *this;
  }

  /// NOTE: We should revisit these constructors and remove any that have
  /// become unnecessary as the implementation improves.
  // scalar_operator() = default;
  // Copy constructor.
  // scalar_operator(const scalar_operator &other);
  // scalar_operator(scalar_operator &other);

  ~scalar_operator() = default;

  // Need this property for consistency with other inherited types.
  // Particularly, to be used when the scalar operator is held within
  // a variant type next to elementary operators.
  std::vector<int> degrees = {};

  /// @brief Return the scalar operator as a concrete complex value.
  std::complex<double>
  evaluate(const std::map<std::string, std::complex<double>> parameters) const;

  // Return the scalar operator as a 1x1 matrix. This is needed for
  // compatibility with the other inherited classes.
  matrix_2 to_matrix(const std::map<int, int> dimensions,
                     const std::map<std::string, std::complex<double>> parameters) const;

  // Arithmetic overloads against other operator types.
  scalar_operator operator*(double other) const;
  scalar_operator operator/(double other) const;
  scalar_operator operator+(double other) const;
  scalar_operator operator-(double other) const;
  scalar_operator& operator*=(double other);
  scalar_operator& operator/=(double other);
  scalar_operator& operator+=(double other);
  scalar_operator& operator-=(double other);
  scalar_operator operator*(std::complex<double> other) const;
  scalar_operator operator/(std::complex<double> other) const;
  scalar_operator operator+(std::complex<double> other) const;
  scalar_operator operator-(std::complex<double> other) const;
  scalar_operator& operator*=(std::complex<double> other);
  scalar_operator& operator/=(std::complex<double> other);
  scalar_operator& operator+=(std::complex<double> other);
  scalar_operator& operator-=(std::complex<double> other);
  scalar_operator operator*(const scalar_operator &other) const;
  scalar_operator operator/(const scalar_operator &other) const;
  scalar_operator operator+(const scalar_operator &other) const;
  scalar_operator operator-(const scalar_operator &other) const;
  scalar_operator& operator*=(const scalar_operator &other);
  scalar_operator& operator/=(const scalar_operator &other);
  scalar_operator& operator+=(const scalar_operator &other);
  scalar_operator& operator-=(const scalar_operator &other);
  /// TODO: implement and test pow

  friend scalar_operator operator*(double other, const scalar_operator &self);
  friend scalar_operator operator/(double other, const scalar_operator &self);
  friend scalar_operator operator+(double other, const scalar_operator &self);
  friend scalar_operator operator-(double other, const scalar_operator &self);
  friend scalar_operator operator*(std::complex<double> other, const scalar_operator &self);
  friend scalar_operator operator/(std::complex<double> other, const scalar_operator &self);
  friend scalar_operator operator+(std::complex<double> other, const scalar_operator &self);
  friend scalar_operator operator-(std::complex<double> other, const scalar_operator &self);

  // /// @brief Returns true if other is a scalar operator with the same
  // /// generator.
  // bool operator==(scalar_operator other);
};


/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
template <typename HandlerTy> // handler needs to inherit from operation_handler
requires std::derived_from<elementary_operator, HandlerTy>
class operator_sum {

private:
  std::vector<std::tuple<scalar_operator, HandlerTy>>
  canonicalize_product(product_operator<HandlerTy> &prod) const;

  std::vector<std::tuple<scalar_operator, HandlerTy>>
  _canonical_terms() const;

  void aggregate_terms(const product_operator<HandlerTy>& head) {
    terms.push_back(head.terms[0]);
    coefficients.push_back(head.coefficients[0]);
  }

  template <typename ... Args>
  void aggregate_terms(const product_operator<HandlerTy> &head, Args&& ... args) {
    terms.push_back(head.terms[0]);
    coefficients.push_back(head.coefficients[0]);
    aggregate_terms(std::forward<Args>(args)...);
  }

protected:

  operator_sum() = default;
  std::vector<std::vector<HandlerTy>> terms;
  std::vector<scalar_operator> coefficients;

public:

  /// @brief Construct a `cudaq::operator_sum<HandlerTy>` given a sequence of
  /// `cudaq::product_operator<HandlerTy>`'s.
  /// This operator expression represents a sum of terms, where each term
  /// is a product of elementary and scalar operators.
  template<class... Args, class = std::enable_if_t<std::conjunction<std::is_same<product_operator<HandlerTy>, Args>...>::value, void>>
  operator_sum(const Args&... args) {
    terms.reserve(sizeof...(Args));
    coefficients.reserve(sizeof...(Args));
    aggregate_terms(args...);
  }

  operator_sum(const std::vector<product_operator<HandlerTy>>& terms) { 
    this->terms.reserve(terms.size());
    this->coefficients.reserve(terms.size());
    for (const product_operator<HandlerTy>& term : terms) {
      this->terms.push_back(term.terms[0]);
      this->coefficients.push_back(term.coefficients[0]);
    }
  }

  operator_sum(std::vector<product_operator<HandlerTy>>&& terms) { 
    this->terms.reserve(terms.size());
    for (const product_operator<HandlerTy>& term : terms) {
      this->terms.push_back(std::move(term.terms[0]));
      this->coefficients.push_back(std::move(term.coefficients[0]));
    }
  }

  // copy constructor
  operator_sum(const operator_sum &other)
    : coefficients(other.coefficients), terms(other.terms) {}

  // move constructor
  operator_sum(operator_sum &&other) 
    : coefficients(std::move(other.coefficients)), terms(std::move(other.terms)) {}

  // assignment operator
  operator_sum& operator=(const operator_sum& other) {
    if (this != &other) {
      coefficients = other.coefficients;
      terms = other.terms;
    }
    return *this;
  }

  // move assignment operator
  operator_sum& operator=(operator_sum &&other) {
    if (this != &other) {
      coefficients = std::move(other.coefficients);
      terms = std::move(other.terms);
    }
    return *this;
  }

  ~operator_sum() = default;

  operator_sum<HandlerTy> canonicalize() const;

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  bool _is_spinop() const;

  /// TODO: implement
  // template<typename TEval>
  // TEval _evaluate(OperatorArithmetics<TEval> &arithmetics) const;

  /// @brief Return the operator_sum<HandlerTy> as a string.
  std::string to_string() const;

  /// @brief Return the `operator_sum<HandlerTy>` as a matrix.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  matrix_2 to_matrix(
      const std::map<int, int> &dimensions,
      const std::map<std::string, std::complex<double>> &params = {}) const;

  // Arithmetic operators
  operator_sum<HandlerTy> operator*(double other) const;
  operator_sum<HandlerTy> operator+(double other) const;
  operator_sum<HandlerTy> operator-(double other) const;
  operator_sum<HandlerTy>& operator*=(double other);
  operator_sum<HandlerTy>& operator+=(double other);
  operator_sum<HandlerTy>& operator-=(double other);
  operator_sum<HandlerTy> operator*(std::complex<double> other) const;
  operator_sum<HandlerTy> operator+(std::complex<double> other) const;
  operator_sum<HandlerTy> operator-(std::complex<double> other) const;
  operator_sum<HandlerTy>& operator*=(std::complex<double> other);
  operator_sum<HandlerTy>& operator+=(std::complex<double> other);
  operator_sum<HandlerTy>& operator-=(std::complex<double> other);
  operator_sum<HandlerTy> operator*(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const;
  operator_sum<HandlerTy>& operator*=(const scalar_operator &other);
  operator_sum<HandlerTy>& operator+=(const scalar_operator &other);
  operator_sum<HandlerTy>& operator-=(const scalar_operator &other);
  operator_sum<HandlerTy> operator+(const HandlerTy &other) const;
  operator_sum<HandlerTy> operator-(const HandlerTy &other) const;
  operator_sum<HandlerTy> operator*(const HandlerTy &other) const;
  operator_sum<HandlerTy>& operator*=(const HandlerTy &other);
  operator_sum<HandlerTy>& operator+=(const HandlerTy &other);
  operator_sum<HandlerTy>& operator-=(const HandlerTy &other);
  operator_sum<HandlerTy> operator*(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy>& operator*=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy>& operator+=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy>& operator-=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy>& operator*=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy>& operator+=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy>& operator-=(const operator_sum<HandlerTy> &other);

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnon-template-friend"
#endif
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif
/*
  friend operator_sum<HandlerTy> operator*(double other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator+(double other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator-(double other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator*(std::complex<double> other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator+(std::complex<double> other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator-(std::complex<double> other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator*(const scalar_operator &other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator+(const scalar_operator &other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator-(const scalar_operator &other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator*(const HandlerTy &other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator+(const HandlerTy &other, const operator_sum<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator-(const HandlerTy &other, const operator_sum<HandlerTy> &self);
*/
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

  /// @brief Return the number of operator terms that make up this operator sum.
  int term_count() const { return terms.size(); }

  /// @brief  True, if the other value is an operator_sum<HandlerTy> with equivalent terms,
  /// and False otherwise. The equality takes into account that operator
  /// addition is commutative, as is the product of two operators if they
  /// act on different degrees of freedom.
  /// The equality comparison does *not* take commutation relations into
  /// account, and does not try to reorder terms `blockwise`; it may hence
  /// evaluate to False, even if two operators in reality are the same.
  /// If the equality evaluates to True, on the other hand, the operators
  /// are guaranteed to represent the same transformation for all arguments.
  bool operator==(const operator_sum<HandlerTy> &other) const;

  /// FIXME: Protect this once I can do deeper testing in `unittests`.
  // protected:
  std::vector<product_operator<HandlerTy>> get_terms() const { 
    std::vector<product_operator<HandlerTy>> prods;
    prods.reserve(terms.size());
    for (size_t i = 0; i < terms.size(); ++i) {
      prods.push_back(product_operator<HandlerTy>(coefficients[i], terms[i]));
    }
    return prods; }
};

/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
template <typename HandlerTy> // handler needs to inherit from operation_handler
requires std::derived_from<elementary_operator, HandlerTy>
class product_operator : public operator_sum<HandlerTy> {

private:

  void aggregate_terms(const HandlerTy& head) {
    operator_sum<HandlerTy>::terms[0].push_back(head);
  }
  
  template <typename ... Args>
  void aggregate_terms(const HandlerTy &head, Args&& ... args) {
    operator_sum<HandlerTy>::terms[0].push_back(head);
    aggregate_terms(std::forward<Args>(args)...);
  }

public:

  product_operator(scalar_operator coefficient) {
    operator_sum<HandlerTy>::terms.push_back({});
    operator_sum<HandlerTy>::coefficients.push_back(std::move(coefficient));
  }

  // Constructor for an operator expression that represents a product
  // of scalar and elementary operators.
  // arg atomic_operators : The operators of which to compute the product when
  //                         evaluating the operator expression.
  template<class... Args, class = std::enable_if_t<std::conjunction<std::is_same<HandlerTy, Args>...>::value, void>>
  product_operator(scalar_operator coefficient, const Args&... args) {
    operator_sum<HandlerTy>::coefficients.push_back(std::move(coefficient));
    std::vector<HandlerTy> ops = {};
    ops.reserve(sizeof...(Args));
    operator_sum<HandlerTy>::terms.push_back(ops);
    aggregate_terms(args...);
  }

  product_operator(scalar_operator coefficient, const std::vector<HandlerTy>& atomic_operators) { 
    operator_sum<HandlerTy>::terms.push_back(atomic_operators);
    operator_sum<HandlerTy>::coefficients.push_back(std::move(coefficient));
  }

  product_operator(scalar_operator coefficient, std::vector<HandlerTy>&& atomic_operators) {
    operator_sum<HandlerTy>::terms.push_back(std::move(atomic_operators));
    operator_sum<HandlerTy>::coefficients.push_back(std::move(coefficient));
  }

  // copy constructor
  product_operator(const product_operator &other) {
    operator_sum<HandlerTy>::terms = other.terms;
    operator_sum<HandlerTy>::coefficients = other.coefficients;
  }

  // move constructor
  product_operator(product_operator &&other) {
    operator_sum<HandlerTy>::terms = std::move(other.terms);
    operator_sum<HandlerTy>::coefficients = std::move(other.coefficients);
  }

  // assignment operator
  product_operator& operator=(const product_operator& other) {
    if (this != &other) {
      operator_sum<HandlerTy>::terms = other.terms;
      operator_sum<HandlerTy>::coefficients = other.coefficients;
    }
    return *this;
  }

  // move assignment operator
  product_operator& operator=(product_operator &&other) {
    if (this != &other) {
      this->coefficients = std::move(other.coefficients);
      this->terms = std::move(other.terms);
    }
    return *this;
  }

  ~product_operator() = default;

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  /// @brief Return the number of operator terms that make up this product
  /// operator.
  int term_count() const { return operator_sum<HandlerTy>::terms[0].size(); }

  /// @brief Return the `product_operator<HandlerTy>` as a string.
  std::string to_string() const;

  /// @brief Return the `operator_sum<HandlerTy>` as a matrix.
  /// @arg  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  matrix_2
  to_matrix(const std::map<int, int> dimensions,
            const std::map<std::string, std::complex<double>> parameters) const;

  // Arithmetic overloads against all other operator types.
  product_operator<HandlerTy> operator*(double other) const;
  operator_sum<HandlerTy> operator+(double other) const;
  operator_sum<HandlerTy> operator-(double other) const;
  product_operator<HandlerTy>& operator*=(double other);
  product_operator<HandlerTy> operator*(std::complex<double> other) const;
  operator_sum<HandlerTy> operator+(std::complex<double> other) const;
  operator_sum<HandlerTy> operator-(std::complex<double> other) const;
  product_operator<HandlerTy>& operator*=(std::complex<double> other);
  product_operator<HandlerTy> operator*(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const;
  product_operator<HandlerTy>& operator*=(const scalar_operator &other);
  product_operator<HandlerTy> operator*(const HandlerTy &other) const;
  operator_sum<HandlerTy> operator+(const HandlerTy &other) const;
  operator_sum<HandlerTy> operator-(const HandlerTy &other) const;
  product_operator<HandlerTy>& operator*=(const HandlerTy &other);
  product_operator<HandlerTy> operator*(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const product_operator<HandlerTy> &other) const;
  product_operator<HandlerTy>& operator*=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) const;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnon-template-friend"
#endif
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif
/*
  friend product_operator<HandlerTy> operator*(double other, const product_operator<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator+(double other, const product_operator<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator-(double other, const product_operator<HandlerTy> &self);
  friend product_operator<HandlerTy> operator*(std::complex<double> other, const product_operator<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator+(std::complex<double> other, const product_operator<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator-(std::complex<double> other, const product_operator<HandlerTy> &self);
  friend product_operator<HandlerTy> operator*(const scalar_operator &other, const product_operator<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator+(const scalar_operator &other, const product_operator<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator-(const scalar_operator &other, const product_operator<HandlerTy> &self);
  friend product_operator<HandlerTy> operator*(const HandlerTy &other, const product_operator<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator+(const HandlerTy &other, const product_operator<HandlerTy> &self);
  friend operator_sum<HandlerTy> operator-(const HandlerTy &other, const product_operator<HandlerTy> &self);
*/
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

  /// @brief True, if the other value is an operator_sum<HandlerTy> with equivalent terms,
  ///  and False otherwise. The equality takes into account that operator
  ///  addition is commutative, as is the product of two operators if they
  ///  act on different degrees of freedom.
  ///  The equality comparison does *not* take commutation relations into
  ///  account, and does not try to reorder terms `blockwise`; it may hence
  ///  evaluate to False, even if two operators in reality are the same.
  ///  If the equality evaluates to True, on the other hand, the operators
  ///  are guaranteed to represent the same transformation for all arguments.
  bool operator==(product_operator<HandlerTy> other);

  /// FIXME: Protect this once I can do deeper testing in `unittests`.
  // protected:
  std::vector<HandlerTy> get_terms() const { 
    return operator_sum<HandlerTy>::terms[0]; }

  scalar_operator get_coefficient() const { 
    return operator_sum<HandlerTy>::coefficients[0]; }
};


class elementary_operator {

private:
  static std::map<std::string, Definition> m_ops;

protected:
  // FIXME: revise implementation
  /// @brief The number of levels, that is the dimension, for each degree of
  /// freedom in canonical order that the operator acts on. A value of zero or
  /// less indicates that the operator is defined for any dimension of that
  /// degree.
  std::map<int, int> expected_dimensions;

public:
  // The constructor should never be called directly by the user:
  // Keeping it internally documented for now, however.
  /// @brief Constructor.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  elementary_operator(std::string operator_id, const std::vector<int> &degrees)
    : id(operator_id), degrees(degrees) {}

  // constructor
  elementary_operator(std::string operator_id, std::vector<int> &&degrees)
    : id(operator_id), degrees(std::move(degrees)) {}

  // copy constructor
  elementary_operator(const elementary_operator &other)
    : degrees(other.degrees), id(other.id) {}

  // move constructor
  elementary_operator(elementary_operator &&other) 
    : degrees(std::move(other.degrees)), id(other.id) {}

  // assignment operator
  elementary_operator& operator=(const elementary_operator& other) {
    if (this != &other) {
      degrees = other.degrees;
      id = other.id;
    }
    return *this;
  }

  // move assignment operator
  elementary_operator& operator=(elementary_operator &&other) {
    degrees = std::move(other.degrees);
    id = other.id;  
    return *this;
  }

  ~elementary_operator() = default;

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees;
  std::string id;

  /// @brief Return the `elementary_operator` as a string.
  std::string to_string() const;

  /// @brief Return the `elementary_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  matrix_2
  to_matrix(const std::map<int, int> dimensions,
            const std::map<std::string, std::complex<double>> parameters) const;

  // Arithmetic overloads
  product_operator<elementary_operator> operator*(double other) const;
  operator_sum<elementary_operator> operator+(double other) const;
  operator_sum<elementary_operator> operator-(double other) const;
  product_operator<elementary_operator> operator*(std::complex<double> other) const;
  operator_sum<elementary_operator> operator+(std::complex<double> other) const;
  operator_sum<elementary_operator> operator-(std::complex<double> other) const;
  product_operator<elementary_operator> operator*(const scalar_operator &other) const;
  operator_sum<elementary_operator> operator+(const scalar_operator &other) const;
  operator_sum<elementary_operator> operator-(const scalar_operator &other) const;
  product_operator<elementary_operator> operator*(const elementary_operator &other) const;
  operator_sum<elementary_operator> operator+(const elementary_operator &other) const;
  operator_sum<elementary_operator> operator-(const elementary_operator &other) const;

  friend product_operator<elementary_operator> operator*(double other, const elementary_operator &self);
  friend operator_sum<elementary_operator> operator+(double other, const elementary_operator &self);
  friend operator_sum<elementary_operator> operator-(double other, const elementary_operator &self);
  friend product_operator<elementary_operator> operator*(std::complex<double> other, const elementary_operator &self);
  friend operator_sum<elementary_operator> operator+(std::complex<double> other, const elementary_operator &self);
  friend operator_sum<elementary_operator> operator-(std::complex<double> other, const elementary_operator &self);
  friend product_operator<elementary_operator> operator*(const scalar_operator &other, const elementary_operator &self);
  friend operator_sum<elementary_operator> operator+(const scalar_operator &other, const elementary_operator &self);
  friend operator_sum<elementary_operator> operator-(const scalar_operator &other, const elementary_operator &self);

  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(const elementary_operator &other) const {
    return this->id == other.id && this->degrees == other.degrees;
  }

  // Predefined operators.
  static elementary_operator identity(int degree);
  static elementary_operator zero(int degree);
  static elementary_operator annihilate(int degree);
  static elementary_operator create(int degree);
  static elementary_operator momentum(int degree);
  static elementary_operator number(int degree);
  static elementary_operator parity(int degree);
  static elementary_operator position(int degree);
  /// FIXME:
  static elementary_operator squeeze(int degree,
                                     std::complex<double> amplitude);
  static elementary_operator displace(int degree,
                                      std::complex<double> amplitude);

  /// @brief Adds the definition of an elementary operator with the given id to
  /// the class. After definition, an the defined elementary operator can be
  /// instantiated by providing the operator id as well as the degree(s) of
  /// freedom that it acts on. An elementary operator is a parameterized object
  /// acting on certain degrees of freedom. To evaluate an operator, for example
  /// to compute its matrix, the level, that is the dimension, for each degree
  /// of freedom it acts on must be provided, as well as all additional
  /// parameters. Additional parameters must be provided in the form of keyword
  /// arguments. Note: The dimensions passed during operator evaluation are
  /// automatically validated against the expected dimensions specified during
  /// definition - the `create` function does not need to do this.
  /// @arg operator_id : A string that uniquely identifies the defined operator.
  /// @arg expected_dimensions : Defines the number of levels, that is the
  /// dimension,
  ///      for each degree of freedom in canonical (that is sorted) order. A
  ///      negative or zero value for one (or more) of the expected dimensions
  ///      indicates that the operator is defined for any dimension of the
  ///      corresponding degree of freedom.
  /// @arg create : Takes any number of complex-valued arguments and returns the
  ///      matrix representing the operator in canonical order. If the matrix
  ///      can be defined for any number of levels for one or more degree of
  ///      freedom, the `create` function must take an argument called
  ///      `dimension` (or `dim` for short), if the operator acts on a single
  ///      degree of freedom, and an argument called `dimensions` (or `dims` for
  ///      short), if the operator acts
  ///     on multiple degrees of freedom.
  template <typename Func>
  void define(std::string operator_id, std::map<int, int> expected_dimensions,
              Func create) {
    if (elementary_operator::m_ops.find(operator_id) != elementary_operator::m_ops.end()) {
      // todo: make a nice error message to say op already exists
      throw;
    }
    auto defn = Definition();
    defn.create_definition(operator_id, expected_dimensions, create);
    elementary_operator::m_ops[operator_id] = defn;
  }
};

/// @brief Representation of a time-dependent Hamiltonian for Rydberg system
class rydberg_hamiltonian : public operator_sum {
public:
  using Coordinate = std::pair<double, double>;

  /// @brief Constructor.
  /// @param atom_sites List of 2D coordinates for trap sites.
  /// @param amplitude Time-dependent driving amplitude, Omega(t).
  /// @param phase Time-dependent driving phase, phi(t).
  /// @param delta_global Time-dependent driving detuning, Delta_global(t).
  /// @param atom_filling Optional. Marks occupied trap sites (1) and empty
  /// sites (0). Defaults to all sites occupied.
  /// @param delta_local Optional. A tuple of Delta_local(t) and site dependent
  /// local detuning factors.
  rydberg_hamiltonian(
      const std::vector<Coordinate> &atom_sites,
      const scalar_operator &amplitude, const scalar_operator &phase,
      const scalar_operator &delta_global,
      const std::vector<int> &atom_filling = {},
      const std::optional<std::pair<scalar_operator, std::vector<double>>>
          &delta_local = std::nullopt);

  /// @brief Get atom sites.
  const std::vector<Coordinate> &get_atom_sites() const;

  /// @brief Get atom filling.
  const std::vector<int> &get_atom_filling() const;

  /// @brief Get amplitude operator.
  const scalar_operator &get_amplitude() const;

  /// @brief Get phase operator.
  const scalar_operator &get_phase() const;

  /// @brief Get global detuning operator.
  const scalar_operator &get_delta_global() const;

private:
  std::vector<Coordinate> atom_sites;
  std::vector<int> atom_filling;
  scalar_operator amplitude;
  scalar_operator phase;
  scalar_operator delta_global;
  std::optional<std::pair<scalar_operator, std::vector<double>>> delta_local;
};
#ifdef CUDAQ_INSTANTIATE_TEMPLATES
template class product_operator<elementary_operator>;
template class operator_sum<elementary_operator>;
#else
extern template class product_operator<elementary_operator>;
extern template class operator_sum<elementary_operator>;
#endif

} // namespace cudaq