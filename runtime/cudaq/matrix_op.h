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

  virtual std::string op_code_to_string(
      std::unordered_map<std::size_t, int64_t> &dimensions) const override;

protected:
  std::string op_code;
  commutation_relations group;
  bool commutes;
  std::vector<std::size_t> targets;

  matrix_handler(std::string operator_id,
                 const std::vector<std::size_t> &degrees,
                 const commutation_behavior &behavior = commutation_behavior());
  matrix_handler(std::string operator_id, std::vector<std::size_t> &&degrees,
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
                     std::vector<int64_t> expected_dimensions,
                     matrix_callback &&create);

  /// @brief Instantiates a custom operator.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  static product_op<matrix_handler>
  instantiate(std::string operator_id, const std::vector<std::size_t> &degrees,
              const commutation_behavior &behavior = commutation_behavior());

  /// @brief Instantiates a custom operator.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  static product_op<matrix_handler>
  instantiate(std::string operator_id, std::vector<std::size_t> &&degrees,
              const commutation_behavior &behavior = commutation_behavior());

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

  /// @brief Constructs a matrix_handler from a given target index.
  /// @arg target A std::size_t representing the target index relevant for the
  /// operator.
  matrix_handler(std::size_t target);

  /// @brief Constructs a matrix_handler by copying from an
  /// operator_handler-derived instance.
  /// @targ T A type derived from operator_handler.
  /// @arg other A constant reference to the source operator instance.
  template <typename T, std::enable_if_t<std::is_base_of_v<operator_handler, T>,
                                         bool> = true>
  matrix_handler(const T &other);

  /// @brief Constructs a matrix_handler from an operator_handler-derived
  /// instance with specified commutation behavior.
  /// @targ T A type derived from operator_handler.
  /// @arg other A constant reference to the source operator instance.
  /// @arg behavior A commutation_behavior object specifying the commutation
  /// properties.
  template <typename T, std::enable_if_t<std::is_base_of_v<operator_handler, T>,
                                         bool> = true>
  matrix_handler(const T &other, const commutation_behavior &behavior);

  /// @brief Copy constructs a matrix_handler from another matrix_handler
  /// instance.
  /// @arg other A constant reference to the other matrix_handler instance.
  matrix_handler(const matrix_handler &other);

  /// @brief Move constructs a matrix_handler by transferring resources from
  /// another instance.
  /// @arg other An rvalue reference to the other matrix_handler instance.
  matrix_handler(matrix_handler &&other);

  /// @brief Default destructor for matrix_handler.
  ~matrix_handler() = default;

  // assignments

  /// @brief Move assigns a matrix_handler instance by transferring resources
  /// from another instance.
  /// @arg other An rvalue reference to the other matrix_handler instance.
  /// @return A reference to the assigned matrix_handler.
  matrix_handler &operator=(matrix_handler &&other);

  /// @brief Copy assigns a matrix_handler instance from another matrix_handler.
  /// @arg other A constant reference to the other matrix_handler instance.
  /// @return A reference to the assigned matrix_handler.
  matrix_handler &operator=(const matrix_handler &other);

  /// @brief Assigns a base operator to a matrix_handler instance.
  /// @targ T A type derived from operator_handler and not matrix_handler.
  /// @arg other A constant reference to the operator_handler-derived instance.
  /// @return A reference to the assigned matrix_handler.
  template <typename T,
            std::enable_if_t<!std::is_same<T, matrix_handler>::value &&
                                 std::is_base_of_v<operator_handler, T>,
                             bool> = true>
  matrix_handler &operator=(const T &other);

  // evaluations

  /// @brief Return the `matrix_handler` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual complex_matrix
  to_matrix(std::unordered_map<std::size_t, int64_t> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const override;

  /// @brief Generates a string representation of the matrix_handler.
  /// @arg include_degrees A flag indicating whether to include degree
  /// information in the string.
  /// @return A string describing the operator.
  virtual std::string to_string(bool include_degrees) const override;

  // comparisons

  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(const matrix_handler &other) const;

  // predefined operators

  /// @brief Constructs a product operator representing a number operator for
  /// the given degree.
  /// @arg degree : The degree or power to which the number operator is
  /// constructed.
  /// @return A matrix_handler instance that encapsulates the
  /// constructed number operator matrix.
  static matrix_handler number(std::size_t degree);
  /// @brief Creates a parity operator using a product operator with
  /// matrix_handler.
  /// @arg degree : The degree of the parity transformation, which may
  /// correspond to the number of qubits or the order of the operation.
  /// @return A product operator that encapsulates the parity transformation.
  static matrix_handler parity(std::size_t degree);
  /// @brief Constructs a product operator representing the position operator.
  /// @arg degree : Specifies the operator's degree, which may determine the
  /// approximation order or related numerical properties.
  /// @return A product operator constructed with a matrix handler that embodies
  /// the position operator.
  static matrix_handler position(std::size_t degree);
  /// @brief Constructs a momentum operator based on the specified degree.
  /// @arg degree : The degree of the momentum operator, influencing its
  /// construction.
  /// @return A product_op object containing a matrix_handler that represents
  /// the momentum operator.
  static matrix_handler momentum(std::size_t degree);
  /// Operators that accept parameters at runtime.
  /// @brief Creates a squeeze operator with a specific degree.
  /// @arg degree : The degree indicating the intensity of the squeeze
  /// transformation.
  /// @return matrix_handler An operator representing the squeeze
  /// transformation.
  static matrix_handler squeeze(std::size_t degree);
  /// @brief Creates a displacement operator based on a specified degree.
  /// @arg degree : The magnitude or extent of the displacement.
  /// @return A product operator (with a matrix handler) representing the
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
