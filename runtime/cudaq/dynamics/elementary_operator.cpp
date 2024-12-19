/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "common/Logger.h"
#include "operators.h"
#include <iostream>
#include <set>

namespace cudaq {

/// Elementary Operator constructor.
elementary_operator::elementary_operator(std::string operator_id,
                                         std::vector<int> degrees)
    : id(operator_id), degrees(degrees) {}
elementary_operator::elementary_operator(const elementary_operator &other)
    : m_ops(other.m_ops), expected_dimensions(other.expected_dimensions),
      degrees(other.degrees), id(other.id) {}
elementary_operator::elementary_operator(elementary_operator &other)
    : m_ops(other.m_ops), expected_dimensions(other.expected_dimensions),
      degrees(other.degrees), id(other.id) {}

elementary_operator elementary_operator::identity(int degree) {
  std::string op_id = "identity";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      int degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      // Build up the identity matrix.
      for (std::size_t i = 0; i < dimension; i++) {
        mat(i, i) = 1.0 + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "done\n\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::zero(int degree) {
  std::string op_id = "zero";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      // Need to set the degree via the op itself because the
      // argument to the outer function goes out of scope when
      // the user invokes this later on via, e.g, `to_matrix()`.
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      mat.set_zero();
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::annihilate(int degree) {
  std::string op_id = "annihilate";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat(i, i + 1) = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::create(int degree) {
  std::string op_id = "create";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat(i + 1, i) = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::position(int degree) {
  std::string op_id = "position";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = cudaq::complex_matrix(dimension, dimension);
      // position = 0.5 * (create + annihilate)
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat(i + 1, i) = 0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        mat(i, i + 1) = 0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::momentum(int degree) {
  std::string op_id = "momentum";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = cudaq::complex_matrix(dimension, dimension);
      // momentum = 0.5j * (create - annihilate)
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat(i + 1, i) =
            (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        mat(i, i + 1) =
            -1. * (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::number(int degree) {
  std::string op_id = "number";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat(i, i) = static_cast<double>(i) + 0.0j;
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "done\n\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator elementary_operator::parity(int degree) {
  std::string op_id = "parity";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto mat = complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat(i, i) = std::pow(-1., static_cast<double>(i)) + 0.0j;
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "done\n\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return op;
}

elementary_operator
elementary_operator::displace(int degree, std::complex<double> amplitude) {
  std::string op_id = "displace";
  std::vector<int> degrees = {degree};
  auto op = elementary_operator(op_id, degrees);
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      auto degree = op.degrees[0];
      int dimension = dimensions[degree];
      auto temp_mat = cudaq::complex_matrix(dimension, dimension);
      // displace = exp[ (amplitude * create) - (conj(amplitude) * annihilate) ]
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        temp_mat(i + 1, i) =
            amplitude * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        temp_mat(i, i + 1) =
            -1. * std::conj(amplitude) * std::sqrt(static_cast<double>(i + 1)) +
            0.0 * 'j';
      }
      // Not ideal that our method of computing the matrix exponential
      // requires copies here. Maybe we can just use eigen directly here
      // to limit to one copy, but we can address that later.
      auto mat = temp_mat.exp();
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  throw std::runtime_error("currently have a bug in implementation.");
  return op;
}

elementary_operator
elementary_operator::squeeze(int degree, std::complex<double> amplitude) {
  throw std::runtime_error("Not yet implemented.");
}

complex_matrix elementary_operator::to_matrix(
    const std::map<int, int> dimensions,
    const std::map<std::string, std::complex<double>> parameters) const {
  if (m_ops.find(id) == m_ops.end())
    throw std::runtime_error(
        fmt::format("No operator found with this ID '{}'", id));

  return m_ops.at(id).generator(dimensions, parameters);
}

template <typename TEval>
TEval elementary_operator::_evaluate(
    operator_arithmetics<TEval> &arithmetics) const {
  std::cout << "In ElementaryOp _evaluate" << std::endl;
  return arithmetics.evaluate();
}
/// Elementary Operator Arithmetic.

operator_sum elementary_operator::operator+(scalar_operator other) {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  return operator_sum({product_operator({*this}), product_operator({other})});
}

operator_sum elementary_operator::operator-(scalar_operator other) {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  return operator_sum(
      {product_operator({*this}), product_operator({-1. * other})});
}

product_operator elementary_operator::operator*(scalar_operator other) {
  return product_operator({*this, other});
}

product_operator elementary_operator::operator/(scalar_operator other) {
  return product_operator({*this, (1. / other)});
}

operator_sum elementary_operator::operator+(std::complex<double> other) {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  auto other_scalar = scalar_operator(other);
  return operator_sum(
      {product_operator({*this}), product_operator({other_scalar})});
}

operator_sum elementary_operator::operator-(std::complex<double> other) {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  auto other_scalar = scalar_operator((-1. * other));
  return operator_sum(
      {product_operator({*this}), product_operator({other_scalar})});
}

product_operator elementary_operator::operator*(std::complex<double> other) {
  auto other_scalar = scalar_operator(other);
  return product_operator({*this, other_scalar});
}

product_operator elementary_operator::operator/(std::complex<double> other) {
  auto other_scalar = scalar_operator((1. / other));
  return product_operator({*this, other_scalar});
}

operator_sum elementary_operator::operator+(double other) {
  std::complex<double> value(other, 0.0);
  return *this + value;
}

operator_sum elementary_operator::operator-(double other) {
  std::complex<double> value(other, 0.0);
  return *this - value;
}

product_operator elementary_operator::operator*(double other) {
  std::complex<double> value(other, 0.0);
  return *this * value;
}

product_operator elementary_operator::operator/(double other) {
  std::complex<double> value(other, 0.0);
  return *this / value;
}

operator_sum operator+(std::complex<double> other, elementary_operator self) {
  auto other_scalar = scalar_operator(other);
  return operator_sum(
      {product_operator({other_scalar}), product_operator({self})});
}

operator_sum operator-(std::complex<double> other, elementary_operator self) {
  auto other_scalar = scalar_operator(other);
  return operator_sum({product_operator({other_scalar}), (-1. * self)});
}

product_operator operator*(std::complex<double> other,
                           elementary_operator self) {
  auto other_scalar = scalar_operator(other);
  return product_operator({other_scalar, self});
}

product_operator operator/(std::complex<double> other,
                           elementary_operator self) {
  auto other_scalar = scalar_operator((1. / other));
  return product_operator({other_scalar, self});
}

operator_sum operator+(double other, elementary_operator self) {
  auto other_scalar = scalar_operator(other);
  return operator_sum(
      {product_operator({other_scalar}), product_operator({self})});
}

operator_sum operator-(double other, elementary_operator self) {
  auto other_scalar = scalar_operator(other);
  return operator_sum({product_operator({other_scalar}), (-1. * self)});
}

product_operator operator*(double other, elementary_operator self) {
  auto other_scalar = scalar_operator(other);
  return product_operator({other_scalar, self});
}

product_operator operator/(double other, elementary_operator self) {
  auto other_scalar = scalar_operator((1. / other));
  return product_operator({other_scalar, self});
}

product_operator elementary_operator::operator*(elementary_operator other) {
  return product_operator({*this, other});
}

operator_sum elementary_operator::operator+(elementary_operator other) {
  return operator_sum({product_operator({*this}), product_operator({other})});
}

operator_sum elementary_operator::operator-(elementary_operator other) {
  return operator_sum({product_operator({*this}), (-1. * other)});
}

/// FIXME:
// product_operator elementary_operator::operator/(elementary_operator other) {
//   return product_operator({*this, (1./other)});
// }

// /// temporary helper
// std::vector<int> reverseArange(int start, int stop, int step = 1) {
//   std::vector<int> values;
//   if (start == stop) {
//     values = {start};
//   } else {
//     for (int value = stop; value > start; value -= step)
//       values.push_back(value);
//   }
//   return values;
// }

// /// Temporary helper function.
// /// FIXME: Needs a bunch of performance improvement but the initial goal
// /// is just to get something working.
// complex_matrix _cast_to_full_hilbert(complex_matrix &mat, int degree,
//                                      std::vector<int> allDegreesReverseOrder,
//                                      std::map<int, int> dimensions) {
//   std::cout << "\nL343\n";
//   std::vector<int> degreesSubset;
//   for (auto degree_index : allDegreesReverseOrder) {
//     // If we don't have a dimension defined for this degree of freedom,
//     // skip it.
//     if (dimensions.find(degree_index) == dimensions.end()) {
//       continue;
//     } else {
//       degreesSubset.push_back(degree_index);
//     }
//   }
//   std::cout << "\nL354\n";

//   // Create an identity matrix appropriately sized for the initial degree of
//   // freedom in our reveresed subset.
//   complex_matrix returnMatrix = complex_matrix::identity(
//       dimensions[degreesSubset[0]], dimensions[degreesSubset[0]]);
//   // Now we can just loop through the degrees of freedom that we have levels
//   // defined for.
//   // If the initial degree of freedom is for the current `mat`, then we will
//   // start with that instead of an identity matrix.
//   if (degreesSubset[0] == degree) {
//     returnMatrix = mat;
//   }
//   for (auto degree_index : degreesSubset) {
//     if (degree != degree_index) {
//       auto identity = complex_matrix::identity(dimensions[degree_index],
//                                                dimensions[degree_index]);
//       returnMatrix = returnMatrix.kronecker(identity);
//     }
//     // If this is the degree of our operator, no identity padding is needed.
//     else {
//       continue;
//     }
//   }
//   std::cout << "\nL379\n";
//   return returnMatrix;
// }

/// FIXME: Assuming each operator only acts on a single degree of freedom
/// for right now.
// product_operator elementary_operator::operator*(elementary_operator other) {
//   /// FIXME: See above about assuming only 1 DOF per operator for right now.
//   int thisDegree = this->degrees[0];
//   int otherDegree = other.degrees[0];
//   auto allDegrees = reverseArange(std::min(thisDegree, otherDegree),
//                                   std::max(thisDegree, otherDegree));

// auto returnOperator = product_operator({*this, other});

// if (thisDegree == otherDegree) {
//   auto func = [&](std::map<int, int> dimensions,
//                   std::map<std::string, std::complex<double>> parameters) {
//     /// FIXME: Making everything its own explicit variable for now while
//     /// I work out testing bugs.
//     auto thisMatrix = this->to_matrix(dimensions, parameters);
//     auto otherMatrix = other.to_matrix(dimensions, parameters);
//     auto returnMatrix = thisMatrix * otherMatrix;
//     return returnMatrix;
//   };
//   returnOperator.m_generator = CallbackFunction(func);
// }

// else if (thisDegree != otherDegree) {
//   std::cout << "defining function.\n";
//   auto func = [&](std::map<int, int> dimensions,
//                   std::map<std::string, std::complex<double>> parameters) {
//     std::cout << "inside function.\n";
//     auto thisMatrix = this->to_matrix(dimensions, parameters);
//     std::cout << "calling _to_full_hilbert.\n";
//     // auto thisFullSpace =
//     //     _cast_to_full_hilbert(thisMatrix, thisDegree, allDegrees,
//     dimensions);
//     // std::cout << "back from _to_full_hilbert.\n";
//     // thisFullSpace.dump();

//     // auto otherMatrix = other.to_matrix(dimensions, parameters);
//     // auto otherFullSpace =
//     //     _cast_to_full_hilbert(otherMatrix, otherDegree, allDegrees,
//     dimensions);
//     // otherFullSpace.dump();

//     // auto returnMatrix = thisFullSpace * otherFullSpace;
//     // return returnMatrix;
//     return thisFullSpace;
//   };
//   std::cout << "function defined.\n";
//   returnOperator.m_generator = CallbackFunction(func);
// }

//   // else {
//   //   throw std::runtime_error("unexpected error encountered.");
//   // }

//   return returnOperator;
// }

// operator_sum
// elementary_operator::operator+(const elementary_operator &other) const {
// std::map<std::string, std::string> merged_params =
//     aggregate_parameters(this->m_ops, other.m_ops);

// auto merged_func =
//     [this, other](std::vector<int> degrees,
//                   std::vector<VariantArg> parameters) -> complex_matrix {
//   complex_matrix result1 = this->func(degrees, parameters);
//   complex_matrix result2 = other.func(degrees, parameters);

//   for (size_t i = 0; i < result1.rows(); i++) {
//     for (size_t j = 0; j < result1.cols(); j++) {
//       result1(i, j) += result2(i, j);
//     }
//   }

//   return result1;
// };

// return operator_sum(merged_func);
// }

} // namespace cudaq