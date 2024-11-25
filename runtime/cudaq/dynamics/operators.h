/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/matrix.h"
#include "cudaq/qis/state.h"
#include "definition.h"

#include <functional>
#include <iostream>
#include <map>
#include <set>

namespace cudaq {

template <typename TEval>
class operator_arithmetics;

class operator_sum;
class product_operator;
class scalar_operator;
class elementary_operator;

/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
class operator_sum {
private:
  std::vector<product_operator> m_terms;

  std::vector<std::tuple<scalar_operator, elementary_operator>>
  canonicalize_product(product_operator &prod) const;

  std::vector<std::tuple<scalar_operator, elementary_operator>>
  _canonical_terms() const;

public:
  /// @brief Empty constructor that a user can aggregate terms into.
  operator_sum() = default;

  /// @brief Construct a `cudaq::operator_sum` given a sequence of
  /// `cudaq::product_operator`'s.
  /// This operator expression represents a sum of terms, where each term
  /// is a product of elementary and scalar operators.
  operator_sum(const std::vector<product_operator> &terms);

  operator_sum canonicalize() const;

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  std::map<std::string, std::complex<double>> parameters() const;

  bool _is_spinop() const;

  /// TODO: implement
  template <typename TEval>
  TEval _evaluate(operator_arithmetics<TEval> &arithmetics) const;

  /// @brief Return the `operator_sum` as a matrix.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  complex_matrix
  to_matrix(const std::map<int, int> &dimensions,
            const std::map<std::string, double> &params = {}) const;

  // Arithmetic operators
  operator_sum operator+(const operator_sum &other) const;
  operator_sum operator-(const operator_sum &other) const;
  operator_sum operator*(const operator_sum &other) const;
  operator_sum operator/(const operator_sum &other) const;
  operator_sum operator+=(const operator_sum &other);
  operator_sum operator-=(const operator_sum &other);

  operator_sum operator+(const scalar_operator &other) const;
  operator_sum operator-(const scalar_operator &other) const;
  operator_sum operator+=(const scalar_operator &other);
  operator_sum operator-=(const scalar_operator &other);

  operator_sum operator+(const product_operator &other) const;
  operator_sum operator-(const product_operator &other) const;
  operator_sum operator+=(const product_operator &other);
  operator_sum operator-=(const product_operator &other);

  operator_sum operator+(const elementary_operator &other) const;
  operator_sum operator-(const elementary_operator &other) const;
  operator_sum operator+=(const elementary_operator &other);
  operator_sum operator-=(const elementary_operator &other);

  /// @brief Return the operator_sum as a string.
  std::string to_string() const;

  /// @brief  True, if the other value is an operator_sum with equivalent terms,
  /// and False otherwise. The equality takes into account that operator
  /// addition is commutative, as is the product of two operators if they
  /// act on different degrees of freedom.
  /// The equality comparison does *not* take commutation relations into
  /// account, and does not try to reorder terms block-wise; it may hence
  /// evaluate to False, even if two operators in reality are the same.
  /// If the equality evaluates to True, on the other hand, the operators
  /// are guaranteed to represent the same transformation for all arguments.
  bool operator==(const operator_sum &other) const;

  const std::vector<product_operator> &get_terms() const { return m_terms; }

  std::complex<double>
  evaluate(std::map<std::string, std::complex<double>> parameters);
};

/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
class product_operator : public operator_sum {
private:
  /// FIXME: Not sure if this is quite correct but using it as a dummy
  /// type for now.
  std::vector<std::variant<scalar_operator, elementary_operator>> m_terms;

public:
  product_operator() = default;
  ~product_operator() = default;

  /// @brief Constructor for an operator expression that represents a product
  /// of scalar and elementary operators.
  /// @arg atomic_operators : The operators of which to compute the product when
  ///                         evaluating the operator expression.
  product_operator(
      std::vector<std::variant<scalar_operator, elementary_operator>>
          atomic_operators);

  // Arithmetic overloads against all other operator types.
  operator_sum operator+(std::complex<double> other);
  operator_sum operator-(std::complex<double> other);
  product_operator operator*(std::complex<double> other);
  product_operator operator*=(std::complex<double> other);
  product_operator operator/(const std::complex<double> scalar) const;
  operator_sum operator+(operator_sum other);
  operator_sum operator-(operator_sum other);
  product_operator operator*(operator_sum other);
  product_operator operator*=(operator_sum other);
  operator_sum operator+(const scalar_operator &other) const;
  operator_sum operator-(const scalar_operator &other) const;
  product_operator operator*(const scalar_operator &other) const;
  product_operator &operator*=(const scalar_operator &other);
  operator_sum operator+(const product_operator &other) const;
  operator_sum operator-(const product_operator &other) const;
  product_operator operator*(const product_operator &other) const;
  product_operator operator*=(product_operator other);
  operator_sum operator+(elementary_operator other);
  operator_sum operator-(elementary_operator other);
  product_operator operator*(elementary_operator other);
  product_operator operator*=(elementary_operator other);
  /// @brief True, if the other value is an operator_sum with equivalent terms,
  ///  and False otherwise. The equality takes into account that operator
  ///  addition is commutative, as is the product of two operators if they
  ///  act on different degrees of freedom.
  ///  The equality comparison does *not* take commutation relations into
  ///  account, and does not try to reorder terms block-wise; it may hence
  ///  evaluate to False, even if two operators in reality are the same.
  ///  If the equality evaluates to True, on the other hand, the operators
  ///  are guaranteed to represent the same transformation for all arguments.
  bool operator==(product_operator other);

  /// @brief Converts a Pauli word to a product operator
  product_operator _from_word(const std::string &word);

  /// @brief Return the `product_operator` as a string.
  std::string to_string() const;

  /// @brief Return the `operator_sum` as a matrix.
  /// @arg  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  complex_matrix
  to_matrix(const std::map<int, int> dimensions,
            const std::map<std::string, std::complex<double>> parameters) const;

  /// @brief Creates a representation of the operator as a `cudaq::pauli_word`
  /// that can be passed as an argument to quantum kernels.
  // pauli_word to_pauli_word();

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  std::complex<double>
  evaluate(std::map<std::string, std::complex<double>> parameters);

  template <typename TEval>
  TEval _evaluate(operator_arithmetics<TEval> &arithmetics) const;

  /// @brief A map of the parameter names to their concrete, complex values.
  std::map<std::string, std::complex<double>> parameters;

  const std::vector<std::variant<scalar_operator, elementary_operator>> &
  get_terms() const {
    return m_terms;
  };
};

class elementary_operator : public product_operator {
private:
  std::map<std::string, Definition> m_ops;

public:
  // The constructor should never be called directly by the user:
  // Keeping it internally documentd for now, however.
  /// @brief Constructor.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  elementary_operator(std::string operator_id, std::vector<int> degrees);

  // Copy constructor.
  elementary_operator(const elementary_operator &other);
  elementary_operator(elementary_operator &other);

  // Arithmetic overloads against all other operator types.
  /// NOTE: All of below arithmetic implemented except for the division between
  ///       two elementary ops. Further testing at the matrix level is needed.
  operator_sum operator+(std::complex<double> other);
  operator_sum operator-(std::complex<double> other);
  product_operator operator*(std::complex<double> other);
  product_operator operator/(std::complex<double> other);
  operator_sum operator+(double other);
  operator_sum operator-(double other);
  product_operator operator*(double other);
  product_operator operator/(double other);
  operator_sum operator+(scalar_operator other);
  operator_sum operator-(scalar_operator other);
  product_operator operator*(scalar_operator other);
  product_operator operator/(scalar_operator other);
  operator_sum operator+(elementary_operator other);
  operator_sum operator-(elementary_operator other);
  product_operator operator*(elementary_operator other);
  product_operator operator/(elementary_operator other);

  /// TODO: implement and test the below
  operator_sum operator+(operator_sum other);
  operator_sum operator-(operator_sum other);
  operator_sum operator+=(operator_sum other);
  operator_sum operator-=(operator_sum other);
  /// FIXME: Correct signature?
  product_operator operator*(operator_sum other);
  //
  operator_sum operator+(product_operator other);
  operator_sum operator-(product_operator other);
  product_operator operator*(product_operator other);
  product_operator operator/(product_operator other);

  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(elementary_operator other);

  /// @brief Return the `elementary_operator` as a string.
  std::string to_string() const;

  /// @brief Return the `elementary_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  complex_matrix
  to_matrix(const std::map<int, int> dimensions,
            const std::map<std::string, std::complex<double>> parameters) const;

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
    if (m_ops.find(operator_id) != m_ops.end()) {
      // todo: make a nice error message to say op already exists
      throw;
    }
    auto defn = Definition();
    defn.create_definition(operator_id, expected_dimensions, create);
    m_ops[operator_id] = defn;
  }

  // Attributes.

  /// @brief The number of levels, that is the dimension, for each degree of
  /// freedom in canonical order that the operator acts on. A value of zero or
  /// less indicates that the operator is defined for any dimension of that
  /// degree.
  std::map<int, int> expected_dimensions;
  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees;
  /// @brief A map of the parameter names to their concrete, complex values.
  /// This will be enabled once we can handle generalized callback function
  /// arguments.
  /// @FIXME: Not needed until generalizing the function arguments.
  std::map<std::string, std::complex<double>> parameters;
  std::string id;

  // /// @brief Creates a representation of the operator as `pauli_word` that
  // can be passed as an argument to quantum kernels.
  // pauli_word to_pauli_word ovveride();

  std::complex<double>
  evaluate(std::map<std::string, std::complex<double>> parameters);

  template <typename TEval>
  TEval _evaluate(operator_arithmetics<TEval> &arithmetics) const;
};
// Reverse order arithmetic for elementary operators against pure scalars.
operator_sum operator+(std::complex<double> other, elementary_operator self);
operator_sum operator-(std::complex<double> other, elementary_operator self);
product_operator operator*(std::complex<double> other,
                           elementary_operator self);
product_operator operator/(std::complex<double> other,
                           elementary_operator self);
operator_sum operator+(double other, elementary_operator self);
operator_sum operator-(double other, elementary_operator self);
product_operator operator*(double other, elementary_operator self);
product_operator operator/(double other, elementary_operator self);

class scalar_operator : public product_operator {
private:
  // If someone gave us a constant value, we will just return that
  // directly to them when they call `evaluate`.
  std::complex<double> m_constant_value;

public:
  /// @brief Constructor that just takes a callback function with no
  /// arguments.

  scalar_operator(ScalarCallbackFunction &&create) {
    generator = ScalarCallbackFunction(create);
  }

  /// @brief Constructor that just takes and returns a complex double value.
  /// @NOTE: This replicates the behavior of the python `scalar_operator::const`
  /// without the need for an extra member function.
  scalar_operator(std::complex<double> value);
  scalar_operator(double value);

  // Arithmetic overloads against other operator types.
  scalar_operator operator+(scalar_operator other);
  scalar_operator operator-(scalar_operator other);
  scalar_operator operator*(scalar_operator other);
  scalar_operator operator/(scalar_operator other);
  /// TODO:
  scalar_operator pow(scalar_operator other);

  /// TODO: All of the below need deeper testing but are implemented.
  operator_sum operator+(elementary_operator other);
  operator_sum operator-(elementary_operator other);
  product_operator operator*(elementary_operator other);
  product_operator operator/(elementary_operator other);
  // TODO:
  operator_sum operator+(operator_sum other);
  operator_sum operator-(operator_sum other);
  operator_sum operator*(operator_sum other);
  operator_sum operator/(operator_sum other);
  operator_sum operator+(product_operator other);
  operator_sum operator-(product_operator other);
  product_operator operator*(product_operator other);
  product_operator operator/(product_operator other);

  /// @brief Return the scalar operator as a concrete complex value.
  std::complex<double>
  evaluate(const std::map<std::string, std::complex<double>> parameters) const;

  // Return the scalar operator as a 1x1 matrix. This is needed for
  // compatability with the other inherited classes.
  complex_matrix
  to_matrix(const std::map<int, int> dimensions,
            const std::map<std::string, std::complex<double>> parameters) const;

  // /// @brief Returns true if other is a scalar operator with the same
  // /// generator.
  // bool operator==(scalar_operator other);

  /// @brief The function that generates the value of the scalar operator.
  /// The function can take a vector of complex-valued arguments
  /// and returns a number.
  ScalarCallbackFunction generator;

  // Only populated when we've performed arithmetic between various
  // scalar operators.
  std::vector<scalar_operator> _operators_to_compose;

  /// NOTE: We should revisit these constructors and remove any that have
  /// become unnecessary as the implementation improves.
  scalar_operator() = default;
  // Copy constructor.
  scalar_operator(const scalar_operator &other);
  scalar_operator(scalar_operator &other);

  ~scalar_operator() = default;

  template <typename TEval>
  TEval _evaluate(operator_arithmetics<TEval> &arithmetics) const;

  // Need this property for consistency with other inherited types.
  // Particularly, to be used when the scalar operator is held within
  // a variant type next to elementary operators.
  std::vector<int> degrees = {-1};

  // REMOVEME: just using this as a temporary patch:
  std::complex<double> get_val() { return m_constant_value; };

  bool has_val() {
    return m_constant_value.real() != 0.0 && m_constant_value.imag() != 0.0;
  };
};

scalar_operator operator+(scalar_operator self, std::complex<double> other);
scalar_operator operator-(scalar_operator self, std::complex<double> other);
scalar_operator operator*(scalar_operator self, std::complex<double> other);
scalar_operator operator/(scalar_operator self, std::complex<double> other);
scalar_operator operator+(std::complex<double> other, scalar_operator self);
scalar_operator operator-(std::complex<double> other, scalar_operator self);
scalar_operator operator*(std::complex<double> other, scalar_operator self);
scalar_operator operator/(std::complex<double> other, scalar_operator self);
scalar_operator operator+(scalar_operator self, double other);
scalar_operator operator-(scalar_operator self, double other);
scalar_operator operator*(scalar_operator self, double other);
scalar_operator operator/(scalar_operator self, double other);
scalar_operator operator+(double other, scalar_operator self);
scalar_operator operator-(double other, scalar_operator self);
scalar_operator operator*(double other, scalar_operator self);
scalar_operator operator/(double other, scalar_operator self);
void operator+=(scalar_operator &self, std::complex<double> other);
void operator-=(scalar_operator &self, std::complex<double> other);
void operator*=(scalar_operator &self, std::complex<double> other);
void operator/=(scalar_operator &self, std::complex<double> other);
void operator+=(scalar_operator &self, scalar_operator other);
void operator-=(scalar_operator &self, scalar_operator other);
void operator*=(scalar_operator &self, scalar_operator other);
void operator/=(scalar_operator &self, scalar_operator other);

using Operator = std::variant<std::monostate, std::unique_ptr<operator_sum>,
                              std::unique_ptr<product_operator>,
                              std::unique_ptr<elementary_operator>,
                              std::unique_ptr<scalar_operator>>;
} // namespace cudaq