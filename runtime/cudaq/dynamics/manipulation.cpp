/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <set>
#include <unordered_map>

#include "helpers.h"
#include "manipulation.h"
#include "operator_leafs.h"

namespace cudaq {

// EvaluatedMatrix class

const std::vector<int>& EvaluatedMatrix::degrees() const {
  return this->targets;
}

const matrix_2& EvaluatedMatrix::matrix() const {
  return this->value;
}

EvaluatedMatrix::EvaluatedMatrix(std::vector<int> &&degrees, matrix_2 &&matrix)
  : targets(std::move(degrees)), value(std::move(matrix)) {
#if !defined(NDEBUG)
    std::set<int> unique_degrees;
    for (auto d : degrees)
      unique_degrees.insert(d);
    assert(unique_degrees.size() == degrees.size());
#endif
  }

EvaluatedMatrix::EvaluatedMatrix(EvaluatedMatrix &&other)
  : targets(std::move(other.targets)), value(std::move(other.value)) {}

EvaluatedMatrix& EvaluatedMatrix::operator=(EvaluatedMatrix &&other) {
  if (this != &other) {
    this->targets = std::move(other.targets);
    this->value = std::move(other.value);
  }
  return *this;
}

// OperatorArithmetics<EvaluatedMatrix>

template<>
class OperatorArithmetics<EvaluatedMatrix> {

private:

  std::unordered_map<int, int> dimensions; // may be updated during evaluation
  const std::unordered_map<std::string, std::complex<double>> parameters;

  std::vector<int> compute_permutation(const std::vector<int> &op_degrees,
                                          const std::vector<int> &canon_degrees) {
    assert(op_degrees.size() == canon_degrees.size());
    auto states = cudaq::detail::generate_all_states(canon_degrees, this->dimensions);
  
    std::vector<int> reordering;
    for (auto degree : op_degrees) {
      auto it = std::find(canon_degrees.cbegin(), canon_degrees.cend(), degree);
      reordering.push_back(it - canon_degrees.cbegin());
    }
  
    std::vector<std::string> op_states = 
        cudaq::detail::generate_all_states(op_degrees, this->dimensions);
  
    std::vector<int> permutation;
    for (auto state : states) {
      std::string term;
      for (auto i : reordering) {
        term += state[i];
      }
      auto it = std::find(op_states.cbegin(), op_states.cend(), term);
      permutation.push_back(it - op_states.cbegin());
    }
  
    return permutation;
  }
  
  // Given a matrix representation that acts on the given degrees or freedom,
  // sorts the degrees and permutes the matrix to match that canonical order.
  void canonicalize(matrix_2 &matrix, std::vector<int> &degrees) {
    auto current_degrees = degrees;
    cudaq::detail::canonicalize_degrees(degrees);
    if (current_degrees != degrees) {
      auto permutation = this->compute_permutation(current_degrees, degrees);
      cudaq::detail::permute_matrix(matrix, permutation);   
    }
  }

public:

  OperatorArithmetics(std::unordered_map<int, int> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
  : dimensions(dimensions), parameters(parameters) {}
  
  EvaluatedMatrix evaluate(const operator_handler &op) {
    return EvaluatedMatrix(op.degrees(), op.to_matrix(this->dimensions, this->parameters));    
  }

  EvaluatedMatrix evaluate(const scalar_operator &op) {
    return EvaluatedMatrix({}, op.to_matrix(this->parameters));
  }

  EvaluatedMatrix tensor(const scalar_operator &scalar, EvaluatedMatrix op) {
    auto degrees = op.degrees();
    auto matrix = scalar.evaluate(this->parameters) * op.matrix();
    return EvaluatedMatrix(std::move(degrees), std::move(matrix));
  }
  
  EvaluatedMatrix tensor(EvaluatedMatrix op1,
                                            EvaluatedMatrix op2) {
    std::vector<int> degrees;
    auto op1_degrees = op1.degrees();
    auto op2_degrees = op2.degrees();
    degrees.reserve(op1_degrees.size() + op2_degrees.size());
    for (auto d : op1_degrees)
      degrees.push_back(d);
    for (auto d : op2_degrees) {
      assert(std::find(degrees.cbegin(), degrees.cend(), d) == degrees.cend());
      degrees.push_back(d);
    }
    auto matrix = cudaq::kronecker(op1.matrix(), op2.matrix());
    this->canonicalize(matrix, degrees);
    return EvaluatedMatrix(std::move(degrees), std::move(matrix));
  }
  
  EvaluatedMatrix mul(EvaluatedMatrix op1,
                                         EvaluatedMatrix op2) {
    // Elementary operators have sorted degrees such that we have a unique
    // convention for how to define the matrix. Tensor products permute the
    // computed matrix if necessary to guarantee that all operators always have
    // sorted degrees.
    auto degrees = op1.degrees();
    assert(degrees == op2.degrees());
    return EvaluatedMatrix(std::move(degrees), (op1.matrix() * op2.matrix()));
  }
  
  EvaluatedMatrix add(EvaluatedMatrix op1,
                                         EvaluatedMatrix op2) {
    // Elementary operators have sorted degrees such that we have a unique
    // convention for how to define the matrix. Tensor products permute the
    // computed matrix if necessary to guarantee that all operators always have
    // sorted degrees.
    auto degrees = op1.degrees();
    assert(degrees == op2.degrees());
    return EvaluatedMatrix(std::move(degrees), op1.matrix() + op2.matrix());
  }
};

#ifdef CUDAQ_INSTANTIATE_TEMPLATES
template class OperatorArithmetics<EvaluatedMatrix>;
#endif

} // namespace cudaq