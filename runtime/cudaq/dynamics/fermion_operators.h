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

#include "cudaq/utils/tensor.h"
#include "cudaq/operators.h"

namespace cudaq {

// FIXME: rename?
class fermion_operator : public operator_handler{
template <typename T> friend class product_operator;

private:

  // Given that the dimension for fermion operators has to be 2,
  // we effectively may just as well store a 2 x 2 matrix.
  // Since we only ever need the operator Ad, A, N, (1-N), I, 0
  // as well as their negatives (except for I and 0, which should
  // never become negative), we choose to store this merely as a 
  // single integer whose bits correspond to the quadrant entry.
  // That is: 
  // 0 = 0000 = 0,
  // 1 = 0001 = (1-N),
  // 2 = 0010 = A,
  // 4 = 0100 = Ad
  // 8 = 1000 = N
  // 9 = 1001 = I
  // The sign bit indicates the sign of the operator.
  int op_code;
  int target;

  // Note that this constructor is chosen to be independent
  // on the internal encoding; to be less critic, we here use the usual
  // 0 = I, Ad = 1, A = 2, AdA = 3
  fermion_operator(int target, int op_id);

  std::string op_code_to_string() const;
  virtual std::string op_code_to_string(std::unordered_map<int, int> &dimensions) const;

  #if !defined(NDEBUG)
  // Here to check if my reasoning regarding only ever needing the operators 
  // above were correct.
  void validate_opcode() const;
  #endif
  
  void inplace_mult(const fermion_operator &other);

  void flip_phase();

public:

  // read-only properties

  virtual std::string unique_id() const;

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  virtual std::vector<int> degrees() const;

  // constructors and destructors

  fermion_operator(int target);

  ~fermion_operator() = default;

  // evaluations

  /// @brief Return the matrix representation of the operator in the eigenbasis of the number operator.
  /// @arg  `dimensions` : A map specifying the dimension, that is the number of eigenstates, for each degree of freedom.
  virtual matrix_2 to_matrix(std::unordered_map<int, int> &dimensions,
                             const std::unordered_map<std::string, std::complex<double>> &parameters = {}) const;

  virtual std::string to_string(bool include_degrees) const;

  // comparisons

  /// @returns True if, and only if, the two operators have the same effect on any state.
  bool operator==(const fermion_operator &other) const;

  // defined operators

  static operator_sum<fermion_operator> empty();
  static product_operator<fermion_operator> identity();

  static product_operator<fermion_operator> identity(int degree);
  static product_operator<fermion_operator> create(int degree);
  static product_operator<fermion_operator> annihilate(int degree);
  static product_operator<fermion_operator> number(int degree);

  // Note that we don't define position and momentum here, since physically they do not make much sense; see e.g.
  // https://physics.stackexchange.com/questions/319296/why-does-a-fermionic-hamiltonian-always-obey-fermionic-parity-symmetry
};

} // namespace cudaq