/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <unordered_map>
#include <vector>

#include "cudaq/operators/operator_leafs.h"
#include "cudaq/utils/matrix.h"

namespace cudaq {
/// @brief The matrix_handler class manages matrix-based quantum operators.
/// It derives from operator_handler and facilitates the definition,
/// instantiation, evaluation, and manipulation of elementary as well as custom
/// operators that can be represented in matrix form.
class matrix_handler : public operator_handler {
public:
  /// @brief The commutation_behavior struct encapsulates the commutation
  /// properties for a matrix operator:
  ///   - group: Specifies the commutation relations group (using
  ///   commutation_relations enum),
  ///            defaulting to the operator_handler's
  ///            default_commutation_relations.
  ///   - commutes_across_degrees: A flag indicating whether the operator
  ///   commutes across
  ///            different degrees of freedom.
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

  // internal only string encoding
  virtual std::string
  canonical_form(std::unordered_map<std::size_t, std::int64_t> &dimensions,
                 std::vector<std::int64_t> &relevant_dims) const override;

protected:
  std::string op_code;
  commutation_relations group;
  bool commutes;
  std::vector<std::size_t> targets;

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
  /// @param operator_id : A string that uniquely identifies the defined
  /// operator.
  /// @param expected_dimensions : Defines the number of levels, that is the
  /// dimension,
  ///      for each degree of freedom in canonical (that is sorted) order. A
  ///      negative or zero value for one (or more) of the expected dimensions
  ///      indicates that the operator is defined for any dimension of the
  ///      corresponding degree of freedom.
  /// @param create : Takes any number of complex-valued arguments and returns
  /// the
  ///      matrix representing the operator. The matrix must be ordered such
  ///      that the value returned by `op.degrees()` matches the order of the
  ///      matrix, where `op` is the instantiated the operator defined here. The
  ///      `create` function must take a vector of integers that specifies the
  ///      "number of levels" (the dimension) for each degree of freedom that
  ///      the operator acts on, and an unordered map from string to complex
  ///      double that contains additional parameters the operator may use.
  static void define(std::string operator_id,
                     std::vector<std::int64_t> expected_dimensions,
                     matrix_callback &&create,
                     const std::unordered_map<std::string, std::string>
                         &parameter_descriptions = {});

  /// @brief Adds the definition of an elementary operator with the given id.
  // The definition also includes a multi-diagonal matrix generator.
  static void define(std::string operator_id,
                     std::vector<std::int64_t> expected_dimensions,
                     matrix_callback &&create,
                     diag_matrix_callback &&diag_create,
                     const std::unordered_map<std::string, std::string>
                         &parameter_descriptions = {});

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
  /// @param operator_id : A string that uniquely identifies the defined
  /// operator.
  /// @param expected_dimensions : Defines the number of levels, that is the
  /// dimension,
  ///      for each degree of freedom in canonical (that is sorted) order. A
  ///      negative or zero value for one (or more) of the expected dimensions
  ///      indicates that the operator is defined for any dimension of the
  ///      corresponding degree of freedom.
  /// @param create : Takes any number of complex-valued arguments and returns
  /// the
  ///      matrix representing the operator. The matrix must be ordered such
  ///      that the value returned by `op.degrees()` matches the order of the
  ///      matrix, where `op` is the instantiated the operator defined here. The
  ///      `create` function must take a vector of integers that specifies the
  ///      "number of levels" (the dimension) for each degree of freedom that
  ///      the operator acts on, and an unordered map from string to complex
  ///      double that contains additional parameters the operator may use.
  static void
  define(std::string operator_id, std::vector<std::int64_t> expected_dimensions,
         matrix_callback &&create,
         std::unordered_map<std::string, std::string> &&parameter_descriptions);

  /// Removes any definition for an operator with the given id.
  /// Returns true if the definition was removed, returns false
  /// if no definition of an operator with the given id existed
  /// in the first place.
  static bool remove_definition(const std::string &operator_id);

  /// @brief Instantiates a custom operator.
  /// @param operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @param degrees : the degrees of freedom that the operator acts upon.
  static product_op<matrix_handler>
  instantiate(std::string operator_id, const std::vector<std::size_t> &degrees,
              const commutation_behavior &behavior = commutation_behavior());

  /// @brief Instantiates a custom operator.
  /// @param operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @param degrees : the degrees of freedom that the operator acts upon.
  static product_op<matrix_handler>
  instantiate(std::string operator_id, std::vector<std::size_t> &&degrees,
              const commutation_behavior &behavior = commutation_behavior());

  /// @brief Returns a map with parameter names and their description
  /// if such a map was provided when the operator was defined.
  const std::unordered_map<std::string, std::string> &
  get_parameter_descriptions() const;

  /// @brief Returns a vector of integers representing the expected dimension
  /// for each degree of freedom. A negative value indicates that the operator
  /// is defined for any dimension of that degree.
  const std::vector<std::int64_t> &get_expected_dimensions() const;

  // read-only properties

  const commutation_relations &commutation_group = this->group;
  const bool &commutes_across_degrees = this->commutes;

  /// @brief Returns a unique identifier string for this operator instance.
  /// @return A string representing the unique ID of the operator.
  virtual std::string unique_id() const override;

  /// @brief Returns the degrees of freedom on which the operator acts.
  /// @return A vector of std::size_t representing the degrees of freedom.
  virtual std::vector<std::size_t> degrees() const override;

  // constructors and destructors

  /// @brief Constructs a matrix_handler that applies an identity operator to
  /// the given target.
  /// @param target A std::size_t representing the target index relevant for the
  /// operator.
  matrix_handler(std::size_t target);

  /// @brief Instantiates a matrix_handler.
  /// @param operator_id A string identifying the operator definition.
  /// @param degrees A vector defining the degrees of freedom that the operator
  /// acts on.
  /// @param behavior An optional argument to define the commutation behavior.
  matrix_handler(std::string operator_id,
                 const std::vector<std::size_t> &degrees,
                 const commutation_behavior &behavior = commutation_behavior());

  /// @brief Instantiates a matrix_handler.
  /// @param operator_id A string identifying the operator definition.
  /// @param degrees A vector defining the degrees of freedom that the operator
  /// acts on.
  /// @param behavior An optional argument to define the commutation behavior.
  matrix_handler(std::string operator_id, std::vector<std::size_t> &&degrees,
                 const commutation_behavior &behavior = commutation_behavior());

  /// @brief Constructs a matrix_handler by copying from an
  /// operator_handler-derived instance.
  /// @tparam T A type derived from operator_handler.
  /// @param other A constant reference to the source operator instance.
  template <typename T, std::enable_if_t<std::is_base_of_v<operator_handler, T>,
                                         bool> = true>
  matrix_handler(const T &other);

  /// @brief Constructs a matrix_handler from an operator_handler-derived
  /// instance with specified commutation behavior.
  /// @tparam T A type derived from operator_handler.
  /// @param other A constant reference to the source operator instance.
  /// @param behavior A commutation_behavior object specifying the commutation
  /// properties.
  template <typename T, std::enable_if_t<std::is_base_of_v<operator_handler, T>,
                                         bool> = true>
  matrix_handler(const T &other, const commutation_behavior &behavior);

  /// @brief Copy constructs a matrix_handler from another matrix_handler
  /// instance.
  /// @param other A constant reference to the other matrix_handler instance.
  matrix_handler(const matrix_handler &other);

  /// @brief Move constructs a matrix_handler by transferring resources from
  /// another instance.
  /// @param other An rvalue reference to the other matrix_handler instance.
  matrix_handler(matrix_handler &&other);

  /// @brief Default destructor for matrix_handler.
  ~matrix_handler() = default;

  // assignments

  /// @brief Move assigns a matrix_handler instance by transferring resources
  /// from another instance.
  /// @param other An rvalue reference to the other matrix_handler instance.
  /// @return A reference to the assigned matrix_handler.
  matrix_handler &operator=(matrix_handler &&other);

  /// @brief Copy assigns a matrix_handler instance from another matrix_handler.
  /// @param other A constant reference to the other matrix_handler instance.
  /// @return A reference to the assigned matrix_handler.
  matrix_handler &operator=(const matrix_handler &other);

  /// @brief Assigns a base operator to a matrix_handler instance.
  /// @tparam T A type derived from operator_handler and not matrix_handler.
  /// @param other A constant reference to the operator_handler-derived
  /// instance.
  /// @return A reference to the assigned matrix_handler.
  template <typename T,
            std::enable_if_t<!std::is_same<T, matrix_handler>::value &&
                                 std::is_base_of_v<operator_handler, T>,
                             bool> = true>
  matrix_handler &operator=(const T &other);

  // evaluations

  /// @brief Return the `matrix_handler` as a matrix.
  /// @param  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual complex_matrix
  to_matrix(std::unordered_map<std::size_t, std::int64_t> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const override;

  /// @brief Return the `matrix_handler` as a multi-diagonal matrix.
  /// @param  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  /// @param  `parameters` : A map specifying runtime parameter values.
  /// If the multi-diagonal matrix representation is not available, it will
  /// return empty.
  mdiag_sparse_matrix
  to_diagonal_matrix(std::unordered_map<std::size_t, std::int64_t> &dimensions,
                     const std::unordered_map<std::string, std::complex<double>>
                         &parameters = {}) const;

  /// @brief Generates a string representation of the matrix_handler.
  /// @param include_degrees A flag indicating whether to include degree
  /// information in the string.
  /// @return A string describing the operator.
  virtual std::string to_string(bool include_degrees) const override;

  // comparisons

  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(const matrix_handler &other) const;

  // predefined operators

  /// @brief Constructs a operator representing a number operator for
  /// the given degree.
  /// @param degree : The degree of freedom that the parity operator acts on.
  /// @return A matrix_handler instance that encapsulates the
  /// constructed number operator matrix.
  static matrix_handler number(std::size_t degree);
  /// @brief Creates a parity operator with
  /// matrix_handler.
  /// @param degree : The degree of freedom that the parity operator acts on.
  /// @return A matrix_handler that encapsulates the parity transformation.
  static matrix_handler parity(std::size_t degree);
  /// @brief Constructs a position operator representing the position operator.
  /// @param degree : The degree of freedom that the parity operator acts on.
  /// @return A matrix_handler that embodies
  /// the position operator.
  static matrix_handler position(std::size_t degree);
  /// @brief Constructs a momentum operator based on the specified degree.
  /// @param degree : The degree of freedom that the parity operator acts on.
  /// @return A matrix_handler that represents
  /// the momentum operator.
  static matrix_handler momentum(std::size_t degree);
  /// @brief Creates a squeeze operator with a specific degree.
  /// @param degree : The degree of freedom that the parity operator acts on.
  /// @return A matrix_handler representing the squeeze
  /// transformation.
  static matrix_handler squeeze(std::size_t degree);
  /// @brief Creates a displacement operator based on a specified degree.
  /// @param degree : The degree of freedom that the parity operator acts on.
  /// @return A matrix handler representing the
  /// displacement.
  static matrix_handler displace(std::size_t degree);
};
} // namespace cudaq

// needs to be down here such that the handler is defined
// before we include the template declarations that depend on it
#include "cudaq/operators.h"

namespace cudaq::operators {
product_op<matrix_handler> number(std::size_t target);
product_op<matrix_handler> parity(std::size_t target);
product_op<matrix_handler> position(std::size_t target);
product_op<matrix_handler> momentum(std::size_t target);
product_op<matrix_handler> squeeze(std::size_t target);
product_op<matrix_handler> displace(std::size_t target);
} // namespace cudaq::operators
