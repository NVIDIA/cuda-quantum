/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "definition.h"
#include "utils/tensor.h"

#include <functional>
#include <iostream>
#include <map>
#include <set>

namespace cudaq {

template <typename HandlerTy> 
class operator_sum;

template <typename HandlerTy> 
class product_operator;

template <typename HandlerTy> 
class elementary_operator;

class scalar_operator;

/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
template <typename HandlerTy> // handler needs to inherit from operation_handler
class operator_sum {

private:
  std::vector<product_operator<HandlerTy>> m_terms;

  std::vector<std::tuple<scalar_operator, elementary_operator<HandlerTy>>>
  canonicalize_product(product_operator<HandlerTy> &prod) const;

  std::vector<std::tuple<scalar_operator, elementary_operator<HandlerTy>>>
  _canonical_terms() const;

public:
  /// @brief Empty constructor that a user can aggregate terms into.
  // operator_sum<HandlerTy>() = default; // FIXME: NEEDED?

  /// @brief Construct a `cudaq::operator_sum<HandlerTy>` given a sequence of
  /// `cudaq::product_operator<HandlerTy>`'s.
  /// This operator expression represents a sum of terms, where each term
  /// is a product of elementary and scalar operators.
  operator_sum(const std::vector<product_operator<HandlerTy>> &terms)
    : m_terms(terms) {}

  operator_sum<HandlerTy> canonicalize() const;

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  bool _is_spinop() const;

  /// TODO: implement
  // template<typename TEval>
  // TEval _evaluate(OperatorArithmetics<TEval> &arithmetics) const;

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
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator*(operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator*=(operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> operator+=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> operator-=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> operator*(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const;
  operator_sum<HandlerTy> operator*=(const scalar_operator &other);
  operator_sum<HandlerTy> operator+=(const scalar_operator &other);
  operator_sum<HandlerTy> operator-=(const scalar_operator &other);
  operator_sum<HandlerTy> operator*(std::complex<double> other) const;
  operator_sum<HandlerTy> operator+(std::complex<double> other) const;
  operator_sum<HandlerTy> operator-(std::complex<double> other) const;
  operator_sum<HandlerTy> operator*=(std::complex<double> other);
  operator_sum<HandlerTy> operator+=(std::complex<double> other);
  operator_sum<HandlerTy> operator-=(std::complex<double> other);
  operator_sum<HandlerTy> operator*(double other) const;
  operator_sum<HandlerTy> operator+(double other) const;
  operator_sum<HandlerTy> operator-(double other) const;
  operator_sum<HandlerTy> operator*=(double other);
  operator_sum<HandlerTy> operator+=(double other);
  operator_sum<HandlerTy> operator-=(double other);
  operator_sum<HandlerTy> operator*(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator+(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator*=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> operator+=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> operator-=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> operator+(const elementary_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator-(const elementary_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator*(const elementary_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy> operator*=(const elementary_operator<HandlerTy> &other);
  operator_sum<HandlerTy> operator+=(const elementary_operator<HandlerTy> &other);
  operator_sum<HandlerTy> operator-=(const elementary_operator<HandlerTy> &other);

  /// @brief Return the operator_sum<HandlerTy> as a string.
  std::string to_string() const;

  /// @brief Return the number of operator terms that make up this operator sum.
  int term_count() const { return m_terms.size(); }

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
  std::vector<product_operator<HandlerTy>> get_terms() { return m_terms; }
};

template <typename HandlerTy>
operator_sum<HandlerTy> operator*(std::complex<double> other, operator_sum<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(std::complex<double> other, operator_sum<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(std::complex<double> other, operator_sum<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator*(double other, operator_sum<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(double other, operator_sum<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(double other, operator_sum<HandlerTy> self);

/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
template <typename HandlerTy> // handler needs to inherit from operation_handler
class product_operator : public operator_sum<HandlerTy> {

private:
  std::vector<std::variant<scalar_operator, elementary_operator<HandlerTy>>> m_terms;

public:
  // product_operator<HandlerTy>() = default; // FIXME: needed?
  ~product_operator() = default;

  // Constructor for an operator expression that represents a product
  // of scalar and elementary operators.
  // arg atomic_operators : The operators of which to compute the product when
  //                         evaluating the operator expression.
  product_operator(
      std::vector<std::variant<scalar_operator, elementary_operator<HandlerTy>>>
          atomic_operators)
      : m_terms(atomic_operators) {}

  // Arithmetic overloads against all other operator types.
  operator_sum<HandlerTy> operator+(std::complex<double> other);
  operator_sum<HandlerTy> operator-(std::complex<double> other);
  product_operator<HandlerTy> operator*(std::complex<double> other);
  product_operator<HandlerTy> operator*=(std::complex<double> other);
  operator_sum<HandlerTy> operator+(double other);
  operator_sum<HandlerTy> operator-(double other);
  product_operator<HandlerTy> operator*(double other);
  product_operator<HandlerTy> operator*=(double other);
  operator_sum<HandlerTy> operator+(scalar_operator other);
  operator_sum<HandlerTy> operator-(scalar_operator other);
  product_operator<HandlerTy> operator*(scalar_operator other);
  product_operator<HandlerTy> operator*=(scalar_operator other);
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> other);
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> other);
  product_operator<HandlerTy> operator*(product_operator<HandlerTy> other);
  product_operator<HandlerTy> operator*=(product_operator<HandlerTy> other);
  operator_sum<HandlerTy> operator+(elementary_operator<HandlerTy> other);
  operator_sum<HandlerTy> operator-(elementary_operator<HandlerTy> other);
  product_operator<HandlerTy> operator*(elementary_operator<HandlerTy> other);
  product_operator<HandlerTy> operator*=(elementary_operator<HandlerTy> other);
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> other);
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> other);
  operator_sum<HandlerTy> operator*(operator_sum<HandlerTy> other);

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

  /// @brief Creates a representation of the operator as a `cudaq::pauli_word`
  /// that can be passed as an argument to quantum kernels.
  // pauli_word to_pauli_word();

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  /// @brief Return the number of operator terms that make up this product
  /// operator.
  int term_count() const { return m_terms.size(); }

  /// FIXME: Protect this once I can do deeper testing in `unittests`.
  // protected:
  std::vector<std::variant<scalar_operator, elementary_operator<HandlerTy>>> get_terms() {
    return m_terms;
  };
};

template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(std::complex<double> other, product_operator<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(std::complex<double> other, product_operator<HandlerTy> self);
template <typename HandlerTy> 
product_operator<HandlerTy> operator*(std::complex<double> other, product_operator<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(double other, product_operator<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(double other, product_operator<HandlerTy> self);
template <typename HandlerTy> 
product_operator<HandlerTy> operator*(double other, product_operator<HandlerTy> self);

template <typename HandlerTy> 
class elementary_operator : public product_operator<HandlerTy> {

private:
  std::map<std::string, Definition> m_ops;

public:
  // The constructor should never be called directly by the user:
  // Keeping it internally documented for now, however.
  /// @brief Constructor.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  elementary_operator(std::string operator_id, std::vector<int> degrees)
    : id(operator_id), degrees(degrees) {}

  // Copy constructor. FIXME: NEEDED?
  elementary_operator(const elementary_operator<HandlerTy> &other)
    : m_ops(other.m_ops), expected_dimensions(other.expected_dimensions),
      degrees(other.degrees), id(other.id) {}

  elementary_operator(elementary_operator<HandlerTy> &other)
    : m_ops(other.m_ops), expected_dimensions(other.expected_dimensions),
      degrees(other.degrees), id(other.id) {}

  // Arithmetic overloads against all other operator types.
  operator_sum<HandlerTy> operator+(std::complex<double> other);
  operator_sum<HandlerTy> operator-(std::complex<double> other);
  product_operator<HandlerTy> operator*(std::complex<double> other);
  operator_sum<HandlerTy> operator+(double other);
  operator_sum<HandlerTy> operator-(double other);
  product_operator<HandlerTy> operator*(double other);
  operator_sum<HandlerTy> operator+(scalar_operator other);
  operator_sum<HandlerTy> operator-(scalar_operator other);
  product_operator<HandlerTy> operator*(scalar_operator other);
  // big question regarding templates is whether coefficients are part of product or are part of elem ops - 
  // part of product seems more intuitive to me, and probably simplifies code;
  // that would mean even for eager prod/add, we still get a prod/sum as result
  operator_sum<HandlerTy> operator+(elementary_operator<HandlerTy> other);
  operator_sum<HandlerTy> operator-(elementary_operator<HandlerTy> other);
  product_operator<HandlerTy> operator*(elementary_operator<HandlerTy> other);
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> other);
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> other);
  product_operator<HandlerTy> operator*(product_operator<HandlerTy> other);
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> other);
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> other);
  operator_sum<HandlerTy> operator+=(operator_sum<HandlerTy> other);
  operator_sum<HandlerTy> operator-=(operator_sum<HandlerTy> other);
  operator_sum<HandlerTy> operator*(operator_sum<HandlerTy> other);

  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(elementary_operator<HandlerTy> other);

  /// @brief Return the `elementary_operator<HandlerTy>` as a string.
  std::string to_string() const;

  /// @brief Return the `elementary_operator<HandlerTy>` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  matrix_2
  to_matrix(const std::map<int, int> dimensions,
            const std::map<std::string, std::complex<double>> parameters) const;

  // Predefined operators.
  static elementary_operator<HandlerTy> identity(int degree);
  static elementary_operator<HandlerTy> zero(int degree);
  static elementary_operator<HandlerTy> annihilate(int degree);
  static elementary_operator<HandlerTy> create(int degree);
  static elementary_operator<HandlerTy> momentum(int degree);
  static elementary_operator<HandlerTy> number(int degree);
  static elementary_operator<HandlerTy> parity(int degree);
  static elementary_operator<HandlerTy> position(int degree);
  /// FIXME:
  static elementary_operator<HandlerTy> squeeze(int degree,
                                     std::complex<double> amplitude);
  static elementary_operator<HandlerTy> displace(int degree,
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
  std::string id;

  // /// @brief Creates a representation of the operator as `pauli_word` that
  // can be passed as an argument to quantum kernels.
  // pauli_word to_pauli_word ovveride();
};
// Reverse order arithmetic for elementary operators against pure scalars.
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(std::complex<double> other, elementary_operator<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(std::complex<double> other, elementary_operator<HandlerTy> self);
template <typename HandlerTy> 
product_operator<HandlerTy> operator*(std::complex<double> other,
                                      elementary_operator<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(double other, elementary_operator<HandlerTy> self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(double other, elementary_operator<HandlerTy> self);
template <typename HandlerTy> 
product_operator<HandlerTy> operator*(double other, elementary_operator<HandlerTy> self);

// FIXME: check if we really need the inheritance from prod operator, and if so what it should be
// (check if this really should be its own class?)
class scalar_operator {

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
  /// TODO: implement and test pow
  scalar_operator pow(scalar_operator other);
  template <typename HandlerTy> 
  operator_sum<HandlerTy> operator+(elementary_operator<HandlerTy> other);
  template <typename HandlerTy> 
  operator_sum<HandlerTy> operator-(elementary_operator<HandlerTy> other);
  template <typename HandlerTy> 
  product_operator<HandlerTy> operator*(elementary_operator<HandlerTy> other);
  template <typename HandlerTy> 
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> other);
  template <typename HandlerTy> 
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> other);
  template <typename HandlerTy> 
  product_operator<HandlerTy> operator*(product_operator<HandlerTy> other);
  template <typename HandlerTy> 
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> other);
  template <typename HandlerTy> 
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> other);
  template <typename HandlerTy> 
  operator_sum<HandlerTy> operator*(operator_sum<HandlerTy> other);

  /// @brief Return the scalar operator as a concrete complex value.
  std::complex<double>
  evaluate(std::map<std::string, std::complex<double>> parameters) const;

  // Return the scalar operator as a 1x1 matrix. This is needed for
  // compatibility with the other inherited classes.
  matrix_2 to_matrix(std::map<int, int> dimensions,
                     std::map<std::string, std::complex<double>> parameters);

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

  // Need this property for consistency with other inherited types.
  // Particularly, to be used when the scalar operator is held within
  // a variant type next to elementary operators.
  std::vector<int> degrees = {-1};

  // REMOVEME: just using this as a temporary patch:
  std::complex<double> get_val() { return m_constant_value; };
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

} // namespace cudaq