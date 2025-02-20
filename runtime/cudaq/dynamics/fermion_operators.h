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

template <typename HandlerTy> 
class product_operator;

template <typename HandlerTy> 
class operator_sum;

// FIXME: rename?
class fermion_operator : public operator_handler{
template <typename T> friend class product_operator;

private:

  int op_code;
  int target;

  // 0 = I, 1 = Cd, 2 = C, 3 = Z
  fermion_operator(int target, int op_code);

  std::string op_code_to_string() const;

  void inplace_mult(const fermion_operator &other);

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

  //static operator_sum<fermion_operator> position(int degree);
  //static operator_sum<fermion_operator> momentum(int degree);

  // FIXME: position, momentum may not make sense, but parity instead?
  // see also https://physics.stackexchange.com/questions/319296/why-does-a-fermionic-hamiltonian-always-obey-fermionic-parity-symmetry
};

} // namespace cudaq