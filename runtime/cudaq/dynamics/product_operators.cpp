/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "cudaq/operators.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>

namespace cudaq {

/// Product Operator constructors.
product_operator::product_operator(
    std::vector<std::variant<scalar_operator, elementary_operator>>
        atomic_operators)
    : m_terms(atomic_operators) {}

// tensor kroneckerHelper(std::vector<tensor> &matrices) {
//   // essentially we pass in the list of elementary operators to
//   // this function -- with lowest degree being leftmost -- then it computes
//   the
//   // kronecker product of all of them.
//   auto kronecker = [](tensor self, tensor other) {
//     return self.kronecker(other);
//   };

//   return std::accumulate(begin(matrices), end(matrices),
//                          tensor::identity(1, 1), kronecker);
// }

// /// IMPLEMENT:
// tensor product_operator::to_matrix(
//     std::map<int, int> dimensions,
//     std::map<std::string, std::complex<double>> parameters) {

//   /// TODO: This initial logic may not be needed.
//   // std::vector<int> degrees, levels;
//   // for(std::map<int,int>::iterator it = dimensions.begin(); it !=
//   // dimensions.end(); ++it) {
//   //   degrees.push_back(it->first);
//   //   levels.push_back(it->second);
//   // }
//   // // Calculate the size of the full Hilbert space of the given product
//   // operator. int fullSize = std::accumulate(begin(levels), end(levels), 1,
//   // std::multiplies<int>());
//   std::cout << "here 49\n";
//   auto getDegrees = [](auto &&t) { return t.degrees; };
//   auto getMatrix = [&](auto &&t) {
//     auto outMatrix = t.to_matrix(dimensions, parameters);
//     std::cout << "dumping the outMatrix : \n";
//     outMatrix.dump();
//     return outMatrix;
//   };
//   std::vector<tensor> matricesFullVectorSpace;
//   for (auto &term : m_terms) {
//     auto op_degrees = std::visit(getDegrees, term);
//     std::cout << "here 58\n";
//     // Keeps track of if we've already inserted the operator matrix
//     // into the full list of matrices.
//     bool alreadyInserted = false;
//     std::vector<tensor> matrixWithIdentities;
//     /// General procedure for inserting identities:
//     // * check if the operator acts on this degree by looking through
//     // `op_degrees`
//     // * if not, insert an identity matrix of the proper level size
//     // * if so, insert the matrix itself
//     for (auto [degree, level] : dimensions) {
//       std::cout << "here 68\n";
//       auto it = std::find(op_degrees.begin(), op_degrees.end(), degree);
//       if (it != op_degrees.end() && !alreadyInserted) {
//         std::cout << "here 71\n";
//         auto matrix = std::visit(getMatrix, term);
//         std::cout << "here 75\n";
//         matrixWithIdentities.push_back(matrix);
//         std::cout << "here 77\n";
//       } else {
//         std::cout << "here 80\n";
//         matrixWithIdentities.push_back(tensor::identity(level,
//         level));
//       }
//     }
//     std::cout << "here 84\n";
//     matricesFullVectorSpace.push_back(kroneckerHelper(matrixWithIdentities));
//   }
//   // Now just need to accumulate with matrix multiplication all of the
//   // matrices in `matricesFullVectorSpace` -- they should all be the same
//   size
//   // already.
//   std::cout << "here 89\n";

//   // temporary
//   auto out = tensor::identity(1, 1);
//   std::cout << "here 93\n";
//   return out;
// }

// Degrees property
std::vector<int> product_operator::degrees() const {
  std::set<int> unique_degrees;
  // The variant type makes it difficult
  auto beginFunc = [](auto &&t) { return t.degrees.begin(); };
  auto endFunc = [](auto &&t) { return t.degrees.end(); };
  for (const auto &term : m_terms) {
    unique_degrees.insert(std::visit(beginFunc, term),
                          std::visit(endFunc, term));
  }
  // Erase any `-1` degree values that may have come from scalar operators.
  auto it = unique_degrees.find(-1);
  if (it != unique_degrees.end()) {
    unique_degrees.erase(it);
  }
  return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
}

operator_sum product_operator::operator+(scalar_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>> _other = {
      other};
  return operator_sum({*this, product_operator(_other)});
}

operator_sum product_operator::operator-(scalar_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>> _other = {
      other};
  return operator_sum({*this, -1. * product_operator(_other)});
}

product_operator product_operator::operator*(scalar_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>>
      combined_terms = m_terms;
  combined_terms.push_back(other);
  return product_operator(combined_terms);
}

product_operator product_operator::operator*=(scalar_operator other) {
  *this = *this * other;
  return *this;
}

operator_sum product_operator::operator+(std::complex<double> other) {
  return *this + scalar_operator(other);
}

operator_sum product_operator::operator-(std::complex<double> other) {
  return *this - scalar_operator(other);
}

product_operator product_operator::operator*(std::complex<double> other) {
  return *this * scalar_operator(other);
}

product_operator product_operator::operator*=(std::complex<double> other) {
  *this = *this * scalar_operator(other);
  return *this;
}

operator_sum operator+(std::complex<double> other, product_operator self) {
  return operator_sum({scalar_operator(other), self});
}

operator_sum operator-(std::complex<double> other, product_operator self) {
  return scalar_operator(other) - self;
}

product_operator operator*(std::complex<double> other, product_operator self) {
  return scalar_operator(other) * self;
}

operator_sum product_operator::operator+(double other) {
  return *this + scalar_operator(other);
}

operator_sum product_operator::operator-(double other) {
  return *this - scalar_operator(other);
}

product_operator product_operator::operator*(double other) {
  return *this * scalar_operator(other);
}

product_operator product_operator::operator*=(double other) {
  *this = *this * scalar_operator(other);
  return *this;
}

operator_sum operator+(double other, product_operator self) {
  return operator_sum({scalar_operator(other), self});
}

operator_sum operator-(double other, product_operator self) {
  return scalar_operator(other) - self;
}

product_operator operator*(double other, product_operator self) {
  return scalar_operator(other) * self;
}

operator_sum product_operator::operator+(product_operator other) {
  return operator_sum({*this, other});
}

operator_sum product_operator::operator-(product_operator other) {
  return operator_sum({*this, (-1. * other)});
}

product_operator product_operator::operator*(product_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>>
      combined_terms = m_terms;
  combined_terms.insert(combined_terms.end(),
                        std::make_move_iterator(other.m_terms.begin()),
                        std::make_move_iterator(other.m_terms.end()));
  return product_operator(combined_terms);
}

product_operator product_operator::operator*=(product_operator other) {
  *this = *this * other;
  return *this;
}

operator_sum product_operator::operator+(elementary_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>> _other = {
      other};
  return operator_sum({*this, product_operator(_other)});
}

operator_sum product_operator::operator-(elementary_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>> _other = {
      other};
  return operator_sum({*this, -1. * product_operator(_other)});
}

product_operator product_operator::operator*(elementary_operator other) {
  std::vector<std::variant<scalar_operator, elementary_operator>>
      combined_terms = m_terms;
  combined_terms.push_back(other);
  return product_operator(combined_terms);
}

product_operator product_operator::operator*=(elementary_operator other) {
  *this = *this * other;
  return *this;
}

operator_sum product_operator::operator+(operator_sum other) {
  std::vector<product_operator> other_terms = other.get_terms();
  other_terms.insert(other_terms.begin(), *this);
  return operator_sum(other_terms);
}

operator_sum product_operator::operator-(operator_sum other) {
  auto negative_other = (-1. * other);
  std::vector<product_operator> other_terms = negative_other.get_terms();
  other_terms.insert(other_terms.begin(), *this);
  return operator_sum(other_terms);
}

operator_sum product_operator::operator*(operator_sum other) {
  std::vector<product_operator> other_terms = other.get_terms();
  for (auto &term : other_terms) {
    term = *this * term;
  }
  return operator_sum(other_terms);
}

} // namespace cudaq