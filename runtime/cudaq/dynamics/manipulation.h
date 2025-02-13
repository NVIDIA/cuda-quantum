/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once 

#include <unordered_map>
#include <vector>
#include "cudaq/utils/tensor.h"

namespace cudaq {

template <typename TEval>
class OperatorArithmetics {
public:
  /// @brief Accesses the relevant data to evaluate an operator expression
  /// in the leaf nodes, that is in elementary and scalar operators.
  // template <typename HandlerTy>
  //TEval evaluate(HandlerTy &op);

  /// @brief Adds two operators that act on the same degrees of freedom.
  TEval add(TEval val1, TEval val2);

  /// @brief Multiplies two operators that act on the same degrees of freedom.
  TEval mul(TEval val1, TEval val2);

  /// @brief Computes the tensor product of two operators that act on different
  /// degrees of freedom.
  TEval tensor(TEval val1, TEval val2);
};

class EvaluatedMatrix {
private:

  std::vector<int> targets;
  matrix_2 value;

public:
  const std::vector<int>& degrees() const;

  const matrix_2& matrix() const;

  EvaluatedMatrix(const std::vector<int> &degrees, const matrix_2 &matrix);
  EvaluatedMatrix(EvaluatedMatrix &&other);

  // delete copy constructor and copy assignment to avoid unnecessary copies
  EvaluatedMatrix(const EvaluatedMatrix &other) = delete;
  EvaluatedMatrix& operator=(const EvaluatedMatrix &other) = delete;

  EvaluatedMatrix& operator=(EvaluatedMatrix &&other);
};

/// Encapsulates the functions needed to compute the matrix representation
/// of an operator expression.
class MatrixArithmetics : public OperatorArithmetics<EvaluatedMatrix> {
private:
  std::vector<int> compute_permutation(const std::vector<int> &op_degrees,
                                       const std::vector<int> &canon_degrees);
  
  void canonicalize(matrix_2 &op_matrix, std::vector<int> &op_degrees);

public:
  std::unordered_map<int, int> m_dimensions; // may be updated during evaluation
  const std::unordered_map<std::string, std::complex<double>> m_parameters;

  MatrixArithmetics(std::unordered_map<int, int> &dimensions,
                    const std::unordered_map<std::string, std::complex<double>> &parameters);

  // Computes the tensor product of two evaluate operators that act on
  // different degrees of freedom using the kronecker product.
  EvaluatedMatrix tensor(EvaluatedMatrix op1, EvaluatedMatrix op2);
  // Multiplies two evaluated operators that act on the same degrees
  // of freedom.
  EvaluatedMatrix mul(EvaluatedMatrix op1, EvaluatedMatrix op2);
  // Adds two evaluated operators that act on the same degrees
  // of freedom.
  EvaluatedMatrix add(EvaluatedMatrix op1, EvaluatedMatrix op2);
};

}