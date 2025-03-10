/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <unordered_map>
#include <vector>

#include "cudaq/operators.h"
#include "cudaq/utils/tensor.h"

namespace cudaq {

class matrix_operator : public operator_handler {
public:
  struct commutation_behavior {
    commutation_relations group =
        operator_handler::default_commutation_relations;
    bool commutes_across_degrees = true;

    commutation_behavior() {
      this->group = operator_handler::default_commutation_relations;
      this->commutes_across_degrees = true;
    }

    commutation_behavior(commutation_relations commutation_group,
                         bool commutes_across_degrees) {
      this->group = commutation_group;
      this->commutes_across_degrees = commutes_across_degrees;
    }
  };

private:
  static std::unordered_map<std::string, Definition> defined_ops;

  // used when converting other operators to matrix operators
  template <typename T>
  static std::string type_prefix();

  virtual std::string
  op_code_to_string(std::unordered_map<int, int> &dimensions) const override;

protected:
  std::string op_code;
  commutation_relations group;
  bool commutes;
  std::vector<int> targets;

  matrix_operator(
      std::string operator_id, const std::vector<int> &degrees,
      const commutation_behavior &behavior = commutation_behavior());
  matrix_operator(
      std::string operator_id, std::vector<int> &&degrees,
      const commutation_behavior &behavior = commutation_behavior());

public:
#if !defined(NDEBUG)
  static bool
      can_be_canonicalized; // needs to be false; no canonical order can be
                            // defined for matrix operator expressions
#endif

  // tools for custom operators

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
  ///      matrix representing the operator. The matrix must be ordered such
  ///      that the value returned by `op.degrees()` matches the order of the
  ///      matrix, where `op` is the instantiated the operator defined here. The
  ///      `create` function must take a vector of integers that specifies the
  ///      "number of levels" (the dimension) for each degree of freedom that
  ///      the operator acts on, and an unordered map from string to complex
  ///      double that contains additional parameters the operator may use.
  static void define(std::string operator_id,
                     std::vector<int> expected_dimensions,
                     matrix_callback &&create);

  /// @brief Instantiates a custom operator.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  static product_operator<matrix_operator>
  instantiate(std::string operator_id, const std::vector<int> &degrees,
              const commutation_behavior &behavior = commutation_behavior());

  /// @brief Instantiates a custom operator.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  static product_operator<matrix_operator>
  instantiate(std::string operator_id, std::vector<int> &&degrees,
              const commutation_behavior &behavior = commutation_behavior());

  // read-only properties

  const commutation_relations &commutation_group = this->group;
  const bool &commutes_across_degrees = this->commutes;

  virtual std::string unique_id() const override;

  virtual std::vector<int> degrees() const override;

  // constructors and destructors

  matrix_operator(int target);

  template <typename T, std::enable_if_t<std::is_base_of_v<operator_handler, T>,
                                         bool> = true>
  matrix_operator(const T &other);

  template <typename T, std::enable_if_t<std::is_base_of_v<operator_handler, T>,
                                         bool> = true>
  matrix_operator(const T &other, const commutation_behavior &behavior);

  // copy constructor
  matrix_operator(const matrix_operator &other);

  // move constructor
  matrix_operator(matrix_operator &&other);

  ~matrix_operator() = default;

  // assignments

  matrix_operator &operator=(matrix_operator &&other);

  matrix_operator &operator=(const matrix_operator &other);

  template <typename T,
            std::enable_if_t<!std::is_same<T, matrix_operator>::value &&
                                 std::is_base_of_v<operator_handler, T>,
                             bool> = true>
  matrix_operator &operator=(const T &other);

  // evaluations

  /// @brief Return the `matrix_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual matrix_2
  to_matrix(std::unordered_map<int, int> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const override;

  virtual std::string to_string(bool include_degrees) const override;

  // comparisons

  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(const matrix_operator &other) const;

  // predefined operators

  static operator_sum<matrix_operator> empty();
  static product_operator<matrix_operator> identity();

  static product_operator<matrix_operator> identity(int degree);
  static product_operator<matrix_operator> number(int degree);
  static product_operator<matrix_operator> parity(int degree);
  static product_operator<matrix_operator> position(int degree);
  static product_operator<matrix_operator> momentum(int degree);
  /// Operators that accept parameters at runtime.
  static product_operator<matrix_operator> squeeze(int degree);
  static product_operator<matrix_operator> displace(int degree);
};

} // namespace cudaq