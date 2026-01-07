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

#include "cudaq/operators/helpers.h"
#include "cudaq/operators/operator_leafs.h"
#include "cudaq/utils/matrix.h"

namespace cudaq {

enum class pauli { I, X, Y, Z };

class spin_handler : public operator_handler, mdiag_operator_handler {
  template <typename T>
  friend class product_op;
  template <typename T>
  friend class sum_op;

private:
  // I = 0, Z = 1, X = 2, Y = 3
  int op_code;
  std::size_t degree;

  spin_handler(std::size_t target, int op_code);

  // private helpers

  // user friendly string encoding
  std::string op_code_to_string() const;
  // internal only string encoding
  virtual std::string
  canonical_form(std::unordered_map<std::size_t, std::int64_t> &dimensions,
                 std::vector<std::int64_t> &relevant_dims) const override;

  std::complex<double> inplace_mult(const spin_handler &other);

  // helper function for matrix creations
  static void create_matrix(
      const std::string &pauli_word,
      const std::function<void(std::size_t, std::size_t, std::complex<double>)>
          &process_element,
      bool invert_order);

  // overload for consistency with other operator classes
  // that support in-place multiplication
  static cudaq::detail::EigenSparseMatrix
  to_sparse_matrix(const std::string &pauli,
                   const std::vector<std::int64_t> &dimensions,
                   std::complex<double> coeff = 1., bool invert_order = false);

  // overload for consistency with other operator classes
  // that support in-place multiplication
  static complex_matrix to_matrix(const std::string &pauli,
                                  const std::vector<std::int64_t> &dimensions,
                                  std::complex<double> coeff = 1.,
                                  bool invert_order = false);

  // helper function for multi-diagonal matrix creations
  static mdiag_sparse_matrix
  to_diagonal_matrix(const std::string &fermi_word,
                     const std::vector<std::int64_t> &dimensions = {},
                     std::complex<double> coeff = 1.,
                     bool invert_order = false);

public:
  // read-only properties

  pauli as_pauli() const;

  virtual std::string unique_id() const override;

  virtual std::vector<std::size_t> degrees() const override;

  std::size_t target() const;

  // constructors and destructors

  spin_handler(std::size_t target);

  spin_handler(pauli p, std::size_t target);

  ~spin_handler() = default;

  // evaluations

  /// @brief Computes the sparse matrix representation of a Pauli string.
  /// By default, the ordering of the matrix matches the ordering of the Pauli
  /// string.
  static cudaq::detail::EigenSparseMatrix
  to_sparse_matrix(const std::string &pauli, std::complex<double> coeff = 1.,
                   bool invert_order = false);

  /// @brief Computes the matrix representation of a Pauli string.
  /// By default, the ordering of the matrix matches the ordering of the Pauli
  /// string.
  static complex_matrix to_matrix(const std::string &pauli,
                                  std::complex<double> coeff = 1.,
                                  bool invert_order = false);

  /// @brief Return the `matrix_handler` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual complex_matrix
  to_matrix(std::unordered_map<std::size_t, std::int64_t> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const override;

  /// @brief Return the `spin_handler` as a multi-diagonal matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  /// @param  `parameters` : A map specifying runtime parameter values.
  virtual mdiag_sparse_matrix
  to_diagonal_matrix(std::unordered_map<std::size_t, std::int64_t> &dimensions,
                     const std::unordered_map<std::string, std::complex<double>>
                         &parameters = {}) const override;
  virtual std::string to_string(bool include_degrees) const override;

  // comparisons

  bool operator==(const spin_handler &other) const;

  // defined operators

  static spin_handler z(std::size_t degree);
  static spin_handler x(std::size_t degree);
  static spin_handler y(std::size_t degree);
};
} // namespace cudaq

// needs to be down here such that the handler is defined
// before we include the template declarations that depend on it
#include "cudaq/operators.h"

namespace cudaq::spin {
product_op<spin_handler> i(std::size_t target);
product_op<spin_handler> x(std::size_t target);
product_op<spin_handler> y(std::size_t target);
product_op<spin_handler> z(std::size_t target);
sum_op<spin_handler> plus(std::size_t target);
sum_op<spin_handler> minus(std::size_t target);
} // namespace cudaq::spin
