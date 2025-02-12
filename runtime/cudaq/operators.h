/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "definition.h"
#include "dynamics/templates.h"
#include "utils/tensor.h"

#include <concepts>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <type_traits>

namespace cudaq {

class MatrixArithmetics;

class scalar_operator {

private:
  // If someone gave us a constant value, we will just return that
  // directly to them when they call `evaluate`.
  std::optional<std::complex<double>> constant_value;

  /// @brief The function that generates the value of the scalar operator.
  /// The function can take a vector of complex-valued arguments
  /// and returns a number.
  ScalarCallbackFunction generator;

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
  scalar_operator &operator=(const scalar_operator &other);

  // move assignment operator
  scalar_operator &operator=(scalar_operator &&other);

  // evaluations

  /// @brief Return the scalar operator as a concrete complex value.
  std::complex<double> evaluate(
      const std::map<std::string, std::complex<double>> parameters = {}) const;

  ScalarCallbackFunction get_generator() const;

  // Return the scalar operator as a 1x1 matrix. This is needed for
  // compatibility with the other inherited classes.
  matrix_2 to_matrix(
      const std::map<int, int> dimensions = {},
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
  scalar_operator &operator*=(double other);
  scalar_operator &operator/=(double other);
  scalar_operator &operator+=(double other);
  scalar_operator &operator-=(double other);
  scalar_operator operator*(std::complex<double> other) const;
  scalar_operator operator/(std::complex<double> other) const;
  scalar_operator operator+(std::complex<double> other) const;
  scalar_operator operator-(std::complex<double> other) const;
  scalar_operator &operator*=(std::complex<double> other);
  scalar_operator &operator/=(std::complex<double> other);
  scalar_operator &operator+=(std::complex<double> other);
  scalar_operator &operator-=(std::complex<double> other);
  scalar_operator operator*(const scalar_operator &other) const;
  scalar_operator operator/(const scalar_operator &other) const;
  scalar_operator operator+(const scalar_operator &other) const;
  scalar_operator operator-(const scalar_operator &other) const;
  scalar_operator &operator*=(const scalar_operator &other);
  scalar_operator &operator/=(const scalar_operator &other);
  scalar_operator &operator+=(const scalar_operator &other);
  scalar_operator &operator-=(const scalar_operator &other);
  /// TODO: implement and test pow

  friend scalar_operator operator*(scalar_operator &&self, double other);
  friend scalar_operator operator/(scalar_operator &&self, double other);
  friend scalar_operator operator+(scalar_operator &&self, double other);
  friend scalar_operator operator-(scalar_operator &&self, double other);
  friend scalar_operator operator+(scalar_operator &&self,
                                   std::complex<double> other);
  friend scalar_operator operator/(scalar_operator &&self,
                                   std::complex<double> other);
  friend scalar_operator operator+(scalar_operator &&self,
                                   std::complex<double> other);
  friend scalar_operator operator-(scalar_operator &&self,
                                   std::complex<double> other);

  // left-hand arithmetics

  friend scalar_operator operator*(double other, const scalar_operator &self);
  friend scalar_operator operator/(double other, const scalar_operator &self);
  friend scalar_operator operator+(double other, const scalar_operator &self);
  friend scalar_operator operator-(double other, const scalar_operator &self);
  friend scalar_operator operator*(std::complex<double> other,
                                   const scalar_operator &self);
  friend scalar_operator operator/(std::complex<double> other,
                                   const scalar_operator &self);
  friend scalar_operator operator+(std::complex<double> other,
                                   const scalar_operator &self);
  friend scalar_operator operator-(std::complex<double> other,
                                   const scalar_operator &self);
};

/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
template <typename HandlerTy> // handler needs to inherit from operation_handler
class operator_sum {

private:
  std::tuple<std::vector<scalar_operator>, std::vector<HandlerTy>>
  m_canonicalize_product(product_operator<HandlerTy> &prod) const;

  std::tuple<std::vector<scalar_operator>, std::vector<HandlerTy>>
  m_canonical_terms() const;

  matrix_2 m_evaluate(MatrixArithmetics arithmetics,
                      bool pad_terms = true) const;

  void aggregate_terms();

  template <typename... Args>
  void aggregate_terms(const product_operator<HandlerTy> &head, Args &&...args);

protected:
  operator_sum() = default;
  std::vector<std::vector<HandlerTy>> terms;
  std::vector<scalar_operator> coefficients;

public:
  // read-only properties

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  /// @brief Return the number of operator terms that make up this operator sum.
  int n_terms() const;

  std::vector<product_operator<HandlerTy>> get_terms() const;

  // constructors and destructors

  template <class... Args,
            class = std::enable_if_t<
                std::conjunction<
                    std::is_same<product_operator<HandlerTy>, Args>...>::value,
                void>>
  operator_sum(const Args &...args);

  operator_sum(const std::vector<product_operator<HandlerTy>> &terms);

  operator_sum(std::vector<product_operator<HandlerTy>> &&terms);

  // copy constructor
  operator_sum(const operator_sum<HandlerTy> &other);

  // move constructor
  operator_sum(operator_sum<HandlerTy> &&other);

  ~operator_sum() = default;

  // assignments

  // assignment operator
  operator_sum<HandlerTy> &operator=(const operator_sum<HandlerTy> &other);

  // move assignment operator
  operator_sum<HandlerTy> &operator=(operator_sum<HandlerTy> &&other);

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
  matrix_2 to_matrix(
      const std::map<int, int> &dimensions = {},
      const std::map<std::string, std::complex<double>> &parameters = {}) const;

  // comparisons

  /// @brief  True, if the other value is an operator_sum<HandlerTy> with
  /// equivalent terms, and False otherwise. The equality takes into account
  /// that operator addition is commutative, as is the product of two operators
  /// if they act on different degrees of freedom. The equality comparison does
  /// *not* take commutation relations into account, and does not try to reorder
  /// terms `blockwise`; it may hence evaluate to False, even if two operators
  /// in reality are the same. If the equality evaluates to True, on the other
  /// hand, the operators are guaranteed to represent the same transformation
  /// for all arguments.
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
  operator_sum<HandlerTy>
  operator*(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy>
  operator+(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy>
  operator-(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;

  operator_sum<HandlerTy> &operator*=(double other);
  operator_sum<HandlerTy> &operator+=(double other);
  operator_sum<HandlerTy> &operator-=(double other);
  operator_sum<HandlerTy> &operator*=(std::complex<double> other);
  operator_sum<HandlerTy> &operator+=(std::complex<double> other);
  operator_sum<HandlerTy> &operator-=(std::complex<double> other);
  operator_sum<HandlerTy> &operator*=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator+=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator-=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator*=(const HandlerTy &other);
  operator_sum<HandlerTy> &operator+=(const HandlerTy &other);
  operator_sum<HandlerTy> &operator-=(const HandlerTy &other);
  operator_sum<HandlerTy> &operator*=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> &operator+=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> &operator-=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> &operator*=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> &operator+=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> &operator-=(const operator_sum<HandlerTy> &other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template
  // instantiation is a nightmare.
  template <typename T>
  friend operator_sum<T> operator*(double other, const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(double other, const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(double other, const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator*(std::complex<double> other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(std::complex<double> other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(std::complex<double> other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator*(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator*(const T &other, const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const T &other, const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const T &other, const operator_sum<T> &self);

  template <typename T>
  friend operator_sum<T>
  product_operator<T>::operator*(const operator_sum<T> &other) const;
  template <typename T>
  friend operator_sum<T>
  product_operator<T>::operator+(const operator_sum<T> &other) const;
  template <typename T>
  friend operator_sum<T>
  product_operator<T>::operator-(const operator_sum<T> &other) const;
};

/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
template <typename HandlerTy> // handler needs to inherit from operation_handler
class product_operator : public operator_sum<HandlerTy> {
  friend class operator_sum<HandlerTy>; // FIXME: explicitly list members
                                        // instead?

private:
  void aggregate_terms();

  template <typename... Args>
  void aggregate_terms(const HandlerTy &head, Args &&...args);

  matrix_2 m_evaluate(MatrixArithmetics arithmetics,
                      bool pad_terms = true) const;

public:
  // read-only properties

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  /// @brief Return the number of operator terms that make up this product
  /// operator.
  int n_terms() const;

  std::vector<HandlerTy> get_terms() const;

  scalar_operator get_coefficient() const;

  // constructors and destructors

  template <
      class... Args,
      class = std::enable_if_t<
          std::conjunction<std::is_same<HandlerTy, Args>...>::value, void>>
  product_operator(scalar_operator coefficient, const Args &...args);

  product_operator(scalar_operator coefficient,
                   const std::vector<HandlerTy> &atomic_operators);

  product_operator(scalar_operator coefficient,
                   std::vector<HandlerTy> &&atomic_operators);

  // copy constructor
  product_operator(const product_operator<HandlerTy> &other);

  // move constructor
  product_operator(product_operator<HandlerTy> &&other);

  ~product_operator() = default;

  // assignments

  // assignment operator
  product_operator<HandlerTy> &
  operator=(const product_operator<HandlerTy> &other);

  // move assignment operator
  product_operator<HandlerTy> &operator=(product_operator<HandlerTy> &&other);

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
  matrix_2
  to_matrix(std::map<int, int> dimensions = {},
            std::map<std::string, std::complex<double>> parameters = {}) const;

  // comparisons

  /// @brief True, if the other value is an operator_sum<HandlerTy> with
  /// equivalent terms,
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
  product_operator<HandlerTy>
  operator*(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy>
  operator+(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy>
  operator-(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) const;

  product_operator<HandlerTy> &operator*=(double other);
  product_operator<HandlerTy> &operator*=(std::complex<double> other);
  product_operator<HandlerTy> &operator*=(const scalar_operator &other);
  product_operator<HandlerTy> &operator*=(const HandlerTy &other);
  product_operator<HandlerTy> &
  operator*=(const product_operator<HandlerTy> &other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template
  // instantiation is a nightmare.
  template <typename T>
  friend product_operator<T> operator*(double other,
                                       const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(double other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(double other,
                                   const product_operator<T> &self);
  template <typename T>
  friend product_operator<T> operator*(std::complex<double> other,
                                       const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(std::complex<double> other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(std::complex<double> other,
                                   const product_operator<T> &self);
  template <typename T>
  friend product_operator<T> operator*(const scalar_operator &other,
                                       const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend product_operator<T> operator*(const T &other,
                                       const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const T &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const T &other,
                                   const product_operator<T> &self);
};

class matrix_operator {

private:
  static std::map<std::string, Definition> m_ops;

public:
  // The constructor should never be called directly by the user:
  // Keeping it internally documented for now, however.
  /// @brief Constructor.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  matrix_operator(std::string operator_id, const std::vector<int> &degrees)
      : id(operator_id), degrees(degrees) {}

  // constructor
  matrix_operator(std::string operator_id, std::vector<int> &&degrees)
      : id(operator_id), degrees(std::move(degrees)) {}

  // copy constructor
  matrix_operator(const matrix_operator &other)
      : degrees(other.degrees), id(other.id) {}

  // move constructor
  matrix_operator(matrix_operator &&other)
      : degrees(std::move(other.degrees)), id(other.id) {}

  // assignment operator
  matrix_operator &operator=(const matrix_operator &other) {
    if (this != &other) {
      degrees = other.degrees;
      id = other.id;
    }
    return *this;
  }

  // move assignment operator
  matrix_operator &operator=(matrix_operator &&other) {
    degrees = std::move(other.degrees);
    id = other.id;
    return *this;
  }

  virtual ~matrix_operator() = default;

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees;
  std::string id;

  /// @brief Return the `matrix_operator` as a string.
  std::string to_string() const;

  /// @brief Return the `matrix_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  matrix_2
  to_matrix(std::map<int, int> dimensions = {},
            std::map<std::string, std::complex<double>> parameters = {}) const;

  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(const matrix_operator &other) const {
    return this->id == other.id && this->degrees == other.degrees;
  }

  // Predefined operators.
  static product_operator<matrix_operator> identity(int degree);
  static product_operator<matrix_operator> zero(int degree);
  static product_operator<matrix_operator> annihilate(int degree);
  static product_operator<matrix_operator> create(int degree);
  static product_operator<matrix_operator> momentum(int degree);
  static product_operator<matrix_operator> number(int degree);
  static product_operator<matrix_operator> parity(int degree);
  static product_operator<matrix_operator> position(int degree);
  /// Operators that accept parameters at runtime.
  static product_operator<matrix_operator> squeeze(int degree);
  static product_operator<matrix_operator> displace(int degree);

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
  static void define(std::string operator_id,
                     std::vector<int> expected_dimensions,
                     CallbackFunction &&create) {
    auto defn = Definition(operator_id, expected_dimensions,
                           std::forward<CallbackFunction>(create));
    auto result = matrix_operator::m_ops.insert({operator_id, std::move(defn)});
    if (!result.second) {
      // todo: make a nice error message to say op already exists
      throw;
    }
  }
};

/// @brief Representation of a time-dependent Hamiltonian for Rydberg system
class rydberg_hamiltonian {
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
template class product_operator<matrix_operator>;
template class operator_sum<matrix_operator>;
#else
extern template class product_operator<matrix_operator>;
extern template class operator_sum<matrix_operator>;
#endif

template <typename TEval>
class OperatorArithmetics {
public:
  /// @brief Accesses the relevant data to evaluate an operator expression
  /// in the leaf nodes, that is in elementary and scalar operators.
  TEval evaluate(product_operator<matrix_operator> &op);

  /// @brief Adds two operators that act on the same degrees of freedom.
  TEval add(TEval val1, TEval val2);

  /// @brief Multiplies two operators that act on the same degrees of freedom.
  TEval mul(TEval val1, TEval val2);

  /// @brief Computes the tensor product of two operators that act on different
  /// degrees of freedom.
  TEval tensor(TEval val1, TEval val2);
};

class EvaluatedMatrix {
  friend class MatrixArithmetics;

private:
  std::vector<int> m_degrees;
  matrix_2 m_matrix;

public:
  EvaluatedMatrix() = default;
  EvaluatedMatrix(std::vector<int> degrees, matrix_2 matrix)
      : m_degrees(degrees), m_matrix(matrix) {}

  /// @brief The degrees of freedom that the matrix of the evaluated value
  /// applies to.
  std::vector<int> degrees() { return m_degrees; }

  /// @brief The matrix representation of an evaluated operator, according
  /// to the sequence of degrees of freedom associated with the evaluated
  /// value.
  matrix_2 matrix() { return m_matrix; }
};

/// Encapsulates the functions needed to compute the matrix representation
/// of an operator expression.
class MatrixArithmetics : public OperatorArithmetics<EvaluatedMatrix> {
private:
  std::vector<int> _compute_permutation(std::vector<int> op_degrees,
                                        std::vector<int> canon_degrees);
  std::tuple<matrix_2, std::vector<int>>
  _canonicalize(matrix_2 &op_matrix, std::vector<int> op_degrees);

public:
  std::map<int, int> &m_dimensions; // fixme: make const
  std::map<std::string, std::complex<double>>
      &m_parameters; // fixme: make const

  MatrixArithmetics(std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> parameters)
      : m_dimensions(dimensions), m_parameters(parameters) {}

  // Computes the tensor product of two evaluate operators that act on
  // different degrees of freedom using the kronecker product.
  EvaluatedMatrix tensor(EvaluatedMatrix op1, EvaluatedMatrix op2);
  // Multiplies two evaluated operators that act on the same degrees
  // of freedom.
  EvaluatedMatrix mul(EvaluatedMatrix op1, EvaluatedMatrix op2);
  // Adds two evaluated operators that act on the same degrees
  // of freedom.
  EvaluatedMatrix add(EvaluatedMatrix op1, EvaluatedMatrix op2);
  // Computes the matrix of an ElementaryOperator or ScalarOperator using its
  // `to_matrix` method.
  EvaluatedMatrix evaluate(std::variant<scalar_operator, matrix_operator,
                                        product_operator<matrix_operator>>
                               op);
};

} // namespace cudaq