/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "dynamics/templates.h"
#include "dynamics/callback.h"
#include "utils/tensor.h"

#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <type_traits>

namespace cudaq {

class MatrixArithmetics;
class EvaluatedMatrix;


class scalar_operator {

private:
  // If someone gave us a constant value, we will just return that
  // directly to them when they call `evaluate`.
  std::variant<std::complex<double>, ScalarCallbackFunction> value;

public:

  // constructors and destructors

  scalar_operator(double value);

  /// @brief Constructor that just takes and returns a complex double value.
  /// @NOTE: This replicates the behavior of the python `scalar_operator::const`
  /// without the need for an extra member function.
  scalar_operator(std::complex<double> value);

  scalar_operator(const ScalarCallbackFunction &create);

  /// @brief Constructor that just takes a callback function with no
  /// arguments.
  scalar_operator(ScalarCallbackFunction &&create);

  // copy constructor
  scalar_operator(const scalar_operator &other);

  // move constructor
  scalar_operator(scalar_operator &&other);

  ~scalar_operator() = default;

  // assignments

  // assignment operator
  scalar_operator& operator=(const scalar_operator &other);

  // move assignment operator
  scalar_operator& operator=(scalar_operator &&other);

  // evaluations

  /// @brief Return the scalar operator as a concrete complex value.
  std::complex<double>
  evaluate(const std::map<std::string, std::complex<double>> parameters = {}) const;

  // Return the scalar operator as a 1x1 matrix. This is needed for
  // compatibility with the other inherited classes.
  matrix_2 to_matrix(const std::map<int, int> dimensions = {},
                     const std::map<std::string, std::complex<double>> parameters = {}) const;

  // comparisons

  bool operator==(scalar_operator other);

  // unary operators

  scalar_operator operator-() const;
  scalar_operator operator+() const;

  // right-hand arithmetics

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

  friend scalar_operator operator*(scalar_operator &&self, double other);
  friend scalar_operator operator/(scalar_operator &&self, double other);
  friend scalar_operator operator+(scalar_operator &&self, double other);
  friend scalar_operator operator-(scalar_operator &&self, double other);
  friend scalar_operator operator+(scalar_operator &&self, std::complex<double> other);
  friend scalar_operator operator/(scalar_operator &&self, std::complex<double> other);
  friend scalar_operator operator+(scalar_operator &&self, std::complex<double> other);
  friend scalar_operator operator-(scalar_operator &&self, std::complex<double> other);

  // left-hand arithmetics

  friend scalar_operator operator*(double other, const scalar_operator &self);
  friend scalar_operator operator/(double other, const scalar_operator &self);
  friend scalar_operator operator+(double other, const scalar_operator &self);
  friend scalar_operator operator-(double other, const scalar_operator &self);
  friend scalar_operator operator*(std::complex<double> other, const scalar_operator &self);
  friend scalar_operator operator/(std::complex<double> other, const scalar_operator &self);
  friend scalar_operator operator+(std::complex<double> other, const scalar_operator &self);
  friend scalar_operator operator-(std::complex<double> other, const scalar_operator &self);
};


/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
template <typename HandlerTy>
class operator_sum {
template <typename T> friend class operator_sum;
template <typename T> friend class product_operator;

private:

  EvaluatedMatrix m_evaluate(MatrixArithmetics arithmetics, bool pad_terms = true) const;

  void aggregate_terms();

  template <typename ... Args>
  void aggregate_terms(product_operator<HandlerTy> &&head, Args&& ... args);

protected:

  std::vector<std::vector<HandlerTy>> terms;
  std::vector<scalar_operator> coefficients;

  template<typename... Args, std::enable_if_t<std::conjunction<std::is_same<product_operator<HandlerTy>, Args>...>::value, bool> = true>
  operator_sum(Args&&... args);

  operator_sum(const std::vector<product_operator<HandlerTy>> &terms);

  operator_sum(std::vector<product_operator<HandlerTy>> &&terms);

public:

  // read-only properties

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  /// @brief Return the number of operator terms that make up this operator sum.
  int n_terms() const;

  /// FIXME: GET RID OF THIS (MAKE ITERABLE INSTEAD)
  std::vector<product_operator<HandlerTy>> get_terms() const;

  // constructors and destructors

  operator_sum(const product_operator<HandlerTy>& prod);

  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool> = true>
  operator_sum(const operator_sum<T> &other);

  // copy constructor
  operator_sum(const operator_sum<HandlerTy> &other);

  // move constructor
  operator_sum(operator_sum<HandlerTy> &&other);

  ~operator_sum() = default;

  // assignments

  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool> = true>
  operator_sum<HandlerTy>& operator=(const operator_sum<T> &other);

  // assignment operator
  operator_sum<HandlerTy>& operator=(const operator_sum<HandlerTy> &other);

  // move assignment operator
  operator_sum<HandlerTy>& operator=(operator_sum<HandlerTy> &&other);

  // evaluations

  /// @brief Return the operator_sum<HandlerTy> as a string.
  std::string to_string() const;

  /// @brief Return the `operator_sum<HandlerTy>` as a matrix.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  matrix_2 to_matrix(const std::map<int, int> &dimensions = {},
                     const std::map<std::string, std::complex<double>> &parameters = {}) const;

  // comparisons

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

  // unary operators

  operator_sum<HandlerTy> operator-() const;
  operator_sum<HandlerTy> operator+() const;

  // right-hand arithmetics

  operator_sum<HandlerTy> operator*(double other) const;
  operator_sum<HandlerTy> operator+(double other) const;
  operator_sum<HandlerTy> operator-(double other) const;
  operator_sum<HandlerTy> operator*(std::complex<double> other) const;
  operator_sum<HandlerTy> operator+(std::complex<double> other) const;
  operator_sum<HandlerTy> operator-(std::complex<double> other) const;
  operator_sum<HandlerTy> operator*(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator+(const HandlerTy &other) const;
  operator_sum<HandlerTy> operator-(const HandlerTy &other) const;
  operator_sum<HandlerTy> operator*(const HandlerTy &other) const;
  operator_sum<HandlerTy> operator*(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;

  operator_sum<HandlerTy>& operator*=(double other);
  operator_sum<HandlerTy>& operator+=(double other);
  operator_sum<HandlerTy>& operator-=(double other);
  operator_sum<HandlerTy>& operator*=(std::complex<double> other);
  operator_sum<HandlerTy>& operator+=(std::complex<double> other);
  operator_sum<HandlerTy>& operator-=(std::complex<double> other);
  operator_sum<HandlerTy>& operator*=(const scalar_operator &other);
  operator_sum<HandlerTy>& operator+=(const scalar_operator &other);
  operator_sum<HandlerTy>& operator-=(const scalar_operator &other);
  operator_sum<HandlerTy>& operator*=(const HandlerTy &other);
  operator_sum<HandlerTy>& operator+=(const HandlerTy &other);
  operator_sum<HandlerTy>& operator-=(const HandlerTy &other);
  operator_sum<HandlerTy>& operator*=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy>& operator+=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy>& operator-=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy>& operator*=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy>& operator+=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy>& operator-=(const operator_sum<HandlerTy> &other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template instantiation is a nightmare.
  template<typename T>
  friend operator_sum<T> operator*(double other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(double other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(double other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator*(std::complex<double> other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator*(const scalar_operator &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator*(const T &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(const T &other, const operator_sum<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(const T &other, const operator_sum<T> &self); 

  template<typename T>
  friend operator_sum<T> operator+(double other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(double other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(const T &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(const T &other, const product_operator<T> &self);
};


/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
template <typename HandlerTy>
class product_operator {
template <typename T> friend class product_operator;
template <typename T> friend class operator_sum;

private:

  void aggregate_terms();

  template <typename... Args>
  void aggregate_terms(HandlerTy &&head, Args&& ... args);

  EvaluatedMatrix m_evaluate(MatrixArithmetics arithmetics, bool pad_terms = true) const;

protected:

  std::vector<HandlerTy> operators;
  scalar_operator coefficient;

  template<typename... Args, std::enable_if_t<std::conjunction<std::is_same<HandlerTy, Args>...>::value, bool> = true>
  product_operator(scalar_operator coefficient, Args&&... args);

  product_operator(scalar_operator coefficient, const std::vector<HandlerTy> &atomic_operators);

  product_operator(scalar_operator coefficient, std::vector<HandlerTy> &&atomic_operators);

public:

  // read-only properties

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  /// @brief Return the number of operator terms that make up this product
  /// operator.
  int n_terms() const;

  /// FIXME: GET RID OF THIS (MAKE ITERABLE INSTEAD)
  const std::vector<HandlerTy>& get_terms() const;

  scalar_operator get_coefficient() const;

  // constructors and destructors

  product_operator(HandlerTy &&atomic);

  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool> = true>
  product_operator(const product_operator<T> &other);

  // copy constructor
  product_operator(const product_operator<HandlerTy> &other);

  // move constructor
  product_operator(product_operator<HandlerTy> &&other);

  ~product_operator() = default;

  // assignments

  template<typename T, std::enable_if_t<!std::is_same<T, HandlerTy>::value && std::is_constructible<HandlerTy, T>::value, bool> = true>
  product_operator<HandlerTy>& operator=(const product_operator<T> &other);

  // assignment operator
  product_operator<HandlerTy>& operator=(const product_operator<HandlerTy> &other);

  // move assignment operator
  product_operator<HandlerTy>& operator=(product_operator<HandlerTy> &&other);

  // evaluations

  /// @brief Return the `product_operator<HandlerTy>` as a string.
  std::string to_string() const;

  /// @brief Return the `operator_sum<HandlerTy>` as a matrix.
  /// @arg  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  matrix_2 to_matrix(std::map<int, int> dimensions = {},
                     std::map<std::string, std::complex<double>> parameters = {}) const;

  // comparisons

  /// @brief True, if the other value is an operator_sum<HandlerTy> with equivalent terms,
  ///  and False otherwise. The equality takes into account that operator
  ///  addition is commutative, as is the product of two operators if they
  ///  act on different degrees of freedom.
  ///  The equality comparison does *not* take commutation relations into
  ///  account, and does not try to reorder terms `blockwise`; it may hence
  ///  evaluate to False, even if two operators in reality are the same.
  ///  If the equality evaluates to True, on the other hand, the operators
  ///  are guaranteed to represent the same transformation for all arguments.
  bool operator==(const product_operator<HandlerTy> &other) const;

  // unary operators

  product_operator<HandlerTy> operator-() const;
  product_operator<HandlerTy> operator+() const;

  // right-hand arithmetics

  product_operator<HandlerTy> operator*(double other) const;
  operator_sum<HandlerTy> operator+(double other) const;
  operator_sum<HandlerTy> operator-(double other) const;
  product_operator<HandlerTy> operator*(std::complex<double> other) const;
  operator_sum<HandlerTy> operator+(std::complex<double> other) const;
  operator_sum<HandlerTy> operator-(std::complex<double> other) const;
  product_operator<HandlerTy> operator*(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const;
  product_operator<HandlerTy> operator*(const HandlerTy &other) const;
  operator_sum<HandlerTy> operator+(const HandlerTy &other) const;
  operator_sum<HandlerTy> operator-(const HandlerTy &other) const;
  product_operator<HandlerTy> operator*(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) const;

  product_operator<HandlerTy>& operator*=(double other);
  product_operator<HandlerTy>& operator*=(std::complex<double> other);
  product_operator<HandlerTy>& operator*=(const scalar_operator &other);
  product_operator<HandlerTy>& operator*=(const HandlerTy &other);
  product_operator<HandlerTy>& operator*=(const product_operator<HandlerTy> &other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template instantiation is a nightmare.
  template<typename T>
  friend product_operator<T> operator*(double other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(double other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(double other, const product_operator<T> &self);
  template<typename T>
  friend product_operator<T> operator*(std::complex<double> other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(std::complex<double> other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(std::complex<double> other, const product_operator<T> &self);
  template<typename T>
  friend product_operator<T> operator*(const scalar_operator &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(const scalar_operator &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(const scalar_operator &other, const product_operator<T> &self);
  template<typename T>
  friend product_operator<T> operator*(const T &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator+(const T &other, const product_operator<T> &self);
  template<typename T>
  friend operator_sum<T> operator-(const T &other, const product_operator<T> &self);
};


class operator_handler {
public:
  virtual ~operator_handler() = default;

  virtual std::vector<int> degrees() const = 0;

  virtual bool is_identity() const = 0;

  /// @brief Return the `matrix_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual matrix_2 to_matrix(std::map<int, int> &dimensions,
                             std::map<std::string, std::complex<double>> parameters = {}) const = 0;

  virtual std::string to_string(bool include_degrees = true) const = 0;
};

} // namespace cudaq
