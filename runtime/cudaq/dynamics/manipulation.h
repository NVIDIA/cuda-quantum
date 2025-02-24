/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/tensor.h"
#include <unordered_map>
#include <vector>

namespace cudaq {
  
class operator_handler;
class scalar_operator;

template <typename EvalTy>
class OperatorArithmetics {
public:
  OperatorArithmetics(std::unordered_map<int, int> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters);

  /// @brief Accesses the relevant data to evaluate an operator expression
  /// in the leaf nodes, that is in elementary and scalar operators.
  EvalTy evaluate(const operator_handler &op);
  EvalTy evaluate(const scalar_operator &op);

  /// @brief Computes the tensor product of two operators that act on different
  /// degrees of freedom.
  EvalTy tensor(const scalar_operator &scalar, EvalTy &&op);
  EvalTy tensor(EvalTy &&val1, EvalTy &&val2);

  /// @brief Multiplies two operators that act on the same degrees of freedom.
  EvalTy mul(EvalTy &&val1, EvalTy &&val2);

  /// @brief Adds two operators that act on the same degrees of freedom.
  EvalTy add(EvalTy &&val1, EvalTy &&val2);
};

class EvaluatedMatrix {
private:
  std::vector<int> targets;
  matrix_2 value;

public:
  const std::vector<int> &degrees() const;

  const matrix_2 &matrix() const;

  EvaluatedMatrix(std::vector<int> &&degrees, matrix_2 &&matrix);
  EvaluatedMatrix(EvaluatedMatrix &&other);

  // delete copy constructor and copy assignment to avoid unnecessary copies
  EvaluatedMatrix(const EvaluatedMatrix &other) = delete;
  EvaluatedMatrix &operator=(const EvaluatedMatrix &other) = delete;

  EvaluatedMatrix &operator=(EvaluatedMatrix &&other);
};

#ifndef CUDAQ_INSTANTIATE_TEMPLATES
extern template class OperatorArithmetics<EvaluatedMatrix>;
#endif  

} // namespace cudaq