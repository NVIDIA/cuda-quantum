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

class fermion_handler : public operator_handler, mdiag_operator_handler {
  template <typename T>
  friend class product_op;
  template <typename T>
  friend class sum_op;

private:
  // Given that the dimension for fermion operators has to be 2,
  // we effectively may just as well store a 2 x 2 matrix.
  // Since we only ever need the operator Ad, A, N, (1-N), I, 0,
  // we choose to store this as a single integer whose bits
  // correspond to the quadrant entry.
  // That is:
  // 0 = 0000 = 0,
  // 1 = 0001 = (1-N),
  // 2 = 0010 = A,
  // 4 = 0100 = Ad
  // 8 = 1000 = N
  // 9 = 1001 = I
  int8_t op_code;
  bool commutes;
  std::size_t degree;

  // Note that this constructor is chosen to be independent
  // on the internal encoding; to be less critic, we here use the usual
  // 0 = I, Ad = 1, A = 2, AdA = 3
  fermion_handler(std::size_t target, int op_id);

  // user friendly string encoding
  std::string op_code_to_string() const;
  // internal only string encoding
  virtual std::string
  canonical_form(std::unordered_map<std::size_t, std::int64_t> &dimensions,
                 std::vector<std::int64_t> &relevant_dims) const override;

#if !defined(NDEBUG)
  // Here to check if my reasoning regarding only ever needing the operators
  // above were correct.
  void validate_opcode() const;
#endif

  void inplace_mult(const fermion_handler &other);

  // helper function for matrix creations
  static void create_matrix(
      const std::string &fermi_word,
      const std::function<void(std::size_t, std::size_t, std::complex<double>)>
          &process_element,
      bool invert_order);

  /// @brief Computes the sparse matrix representation of the string encoding
  /// of a fermionic product operator. Private method since this encoding is
  /// not very user friendly.
  static cudaq::detail::EigenSparseMatrix
  to_sparse_matrix(const std::string &fermi_word,
                   const std::vector<std::int64_t> &dimensions = {},
                   std::complex<double> coeff = 1., bool invert_order = false);

  /// @brief Computes the sparse matrix representation of the string encoding
  /// of a fermionic product operator. Private method since this encoding is
  /// not very user friendly.
  static complex_matrix
  to_matrix(const std::string &fermi_word,
            const std::vector<std::int64_t> &dimensions = {},
            std::complex<double> coeff = 1., bool invert_order = false);

  /// @brief Computes the multi-diagonal matrix representation of the string
  /// encoding of a fermionic product operator. Private method since this
  /// encoding is not very user friendly.
  static mdiag_sparse_matrix
  to_diagonal_matrix(const std::string &fermi_word,
                     const std::vector<std::int64_t> &dimensions = {},
                     std::complex<double> coeff = 1.,
                     bool invert_order = false);

public:
  static constexpr commutation_relations commutation_group =
      operator_handler::fermion_commutation_relations;

  // read-only properties

  const bool &commutes_across_degrees = this->commutes;

  virtual std::string unique_id() const override;

  virtual std::vector<std::size_t> degrees() const override;

  std::size_t target() const;

  // constructors and destructors

  fermion_handler(std::size_t target);

  fermion_handler(const fermion_handler &other);

  ~fermion_handler() = default;

  // assignments

  fermion_handler &operator=(const fermion_handler &other);

  // evaluations

  /// @brief Return the matrix representation of the operator in the eigenbasis
  /// of the number operator.
  /// @param  `dimensions` : A map specifying the dimension, that is the number
  /// of eigenstates, for each degree of freedom.
  virtual complex_matrix
  to_matrix(std::unordered_map<std::size_t, std::int64_t> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const override;
  /// @brief Return the multi-diagonal matrix representation of the operator in
  /// the eigenbasis of the number operator.
  /// @param  `dimensions` : A map specifying the dimension, that is the number
  /// of eigenstates, for each degree of freedom.
  /// @param  `parameters` : A map specifying runtime parameter values.
  virtual mdiag_sparse_matrix
  to_diagonal_matrix(std::unordered_map<std::size_t, std::int64_t> &dimensions,
                     const std::unordered_map<std::string, std::complex<double>>
                         &parameters = {}) const override;
  virtual std::string to_string(bool include_degrees) const override;

  // comparisons

  /// @returns True if, and only if, the two operators have the same effect on
  /// any state.
  bool operator==(const fermion_handler &other) const;

  // defined operators

  static fermion_handler create(std::size_t degree);
  static fermion_handler annihilate(std::size_t degree);
  static fermion_handler number(std::size_t degree);

  // Note that we don't define position and momentum here, since physically they
  // do not make much sense; see e.g.
  // https://physics.stackexchange.com/questions/319296/why-does-a-fermionic-hamiltonian-always-obey-fermionic-parity-symmetry
};
} // namespace cudaq

// needs to be down here such that the handler is defined
// before we include the template declarations that depend on it
#include "cudaq/operators.h"

namespace cudaq::fermion {
product_op<fermion_handler> create(std::size_t target);
product_op<fermion_handler> annihilate(std::size_t target);
product_op<fermion_handler> number(std::size_t target);
} // namespace cudaq::fermion
