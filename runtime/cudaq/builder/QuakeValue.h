/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"
#include <functional>
#include <map>
#include <memory>
#include <optional>

namespace mlir {
class Value;
class ImplicitLocOpBuilder;
} // namespace mlir

namespace cudaq {
class QuakeValue;

/// @brief A QuakeValue is meant to provide a thin
/// wrapper around an mlir::Value instance. These QuakeValues
/// represent handles to function arguments and return values from
/// MLIR Operations, specifically Quake Dialect operations. The QuakeValue
/// also exposes and algebraic API enabling one to negate, add, subtract,
/// and multiply QuakeValues with each other as well as with primitive
/// arithmetic types, e.g. double.
class QuakeValue {

protected:
  // Pimpl Idiom for mlir Value
  class ValueHolder;
  std::shared_ptr<ValueHolder> value;

  /// @brief Pointer to the OpBuilder we are using
  mlir::ImplicitLocOpBuilder &opBuilder;

  /// @brief For Values of StdVecType, we might be able
  /// to validate that the number of required unique elements
  /// is equal to the number provided as input at runtime.
  bool canValidateVectorNumElements = true;

public:
  /// @brief Return the actual MLIR Value
  mlir::Value getValue() const;

  /// @brief The constructor, takes the builder and the value to wrap
  QuakeValue(mlir::ImplicitLocOpBuilder &builder, mlir::Value v);

  /// @brief The constructor, takes the builder and a constant double
  /// value, which will map to an arith::ConstantFloatOp Value.
  QuakeValue(mlir::ImplicitLocOpBuilder &builder, double v);

  QuakeValue(QuakeValue &&);
  QuakeValue(QuakeValue &);
  QuakeValue(const QuakeValue &);
  ~QuakeValue();

  /// @brief Dump the QuakeValue to standard out.
  void dump();

  /// @brief Dump the QuakeValue to the given output stream.
  void dump(std::ostream &);

  /// @brief Return true if this QuakeValue of StdVecType can
  /// validate its number of unique elements. We cannot do this in the
  /// case of QuakeValue extractions within for loops where we do not know
  /// the bounds of the loop.
  bool canValidateNumElements() { return canValidateVectorNumElements; }

  /// @brief For a subscriptable QuakeValue, extract a sub set of the elements
  /// starting at the given startIdx and including the following count elements.
  QuakeValue slice(const std::size_t startIdx, const std::size_t count);

  /// @brief For a QuakeValue with type StdVec or Veq, return
  /// the size QuakeValue.
  QuakeValue size();

  /// @brief Return the constant size of this QuakeValue
  /// if it is of Veq type.
  std::optional<std::size_t> constantSize();

  /// @brief Return true if this QuakeValue is of type StdVec.
  /// @return
  bool isStdVec();

  /// @brief For a QuakeValue of type StdVec, return the
  /// number of required elements, i.e. the number of unique
  /// extractions observed.
  std::size_t getRequiredElements();

  /// @brief Return a new QuakeValue when the current value
  /// is indexed, specifically for QuakeValues of type StdVecType
  /// and VeqType.
  QuakeValue operator[](const std::size_t idx);

  /// @brief Return a new QuakeValue when the current value
  /// is indexed, specifically for QuakeValues of type StdVecType
  /// and VeqType.
  QuakeValue operator[](const QuakeValue &idx);

  /// @brief Return the negation of this QuakeValue
  QuakeValue operator-() const;

  /// @brief Multiply this QuakeValue by the given double.
  QuakeValue operator*(const double);

  /// @brief Multiply this QuakeValue by the given QuakeValue
  QuakeValue operator*(QuakeValue other);

  /// @brief Divide this QuakeValue by the given double.
  QuakeValue operator/(const double);

  /// @brief Divide this QuakeValue by the given QuakeValue
  QuakeValue operator/(QuakeValue other);

  /// @brief Add this QuakeValue with the given double.
  QuakeValue operator+(const double);

  /// @brief Add this QuakeValue with the given int.
  QuakeValue operator+(const int);

  /// @brief Add this QuakeValue with the given QuakeValue
  QuakeValue operator+(QuakeValue other);

  /// @brief Subtract the given double from this QuakeValue
  QuakeValue operator-(const double);

  /// @brief Subtract the given int from this QuakeValue
  QuakeValue operator-(const int);

  /// @brief Subtract the given QuakeValue from this QuakeValue
  QuakeValue operator-(QuakeValue other);

  /// @brief Return the inverse (1/x) of this QuakeValue
  QuakeValue inverse() const;
};

#if CUDAQ_USE_STD20
/// @brief Concept constraining the input type below to be a QuakeValue
template <typename ValueType>
concept IsQuakeValue = std::is_convertible_v<ValueType, QuakeValue>;

/// Concept constraining the LHS arguments to be numeric below.
template <typename T>
concept IsNumericType = requires(T param) { std::is_convertible_v<T, double>; };

QuakeValue operator*(IsNumericType auto &&d, IsQuakeValue auto &&q) {
  return q * d;
}

QuakeValue operator*(IsQuakeValue auto &&q, IsNumericType auto &&d) {
  return q * d;
}

QuakeValue operator-(IsNumericType auto &&d, IsQuakeValue auto &&q) {
  return -q + d;
}
QuakeValue operator+(IsNumericType auto &&d, IsQuakeValue auto &&q) {
  return q + d;
}
QuakeValue operator/(IsNumericType auto &&d, IsQuakeValue auto &&q) {
  return q.inverse() * d;
}

#else
// C++ 2011 compatible definitions.
template <typename N, typename Q,
          typename = std::enable_if_t<std::is_convertible_v<N, double>>,
          typename = std::enable_if_t<std::is_convertible_v<Q, QuakeValue>>>
QuakeValue operator*(N &&d, Q &&q) {
  return q * d;
}

template <typename N, typename Q,
          typename = std::enable_if_t<std::is_convertible_v<N, double>>,
          typename = std::enable_if_t<std::is_convertible_v<Q, QuakeValue>>>
QuakeValue operator*(Q &&q, N &&d) {
  return q * d;
}

template <typename N, typename Q,
          typename = std::enable_if_t<std::is_convertible_v<N, double>>,
          typename = std::enable_if_t<std::is_convertible_v<Q, QuakeValue>>>
QuakeValue operator-(N &&d, Q &&q) {
  return -q + d;
}

template <typename N, typename Q,
          typename = std::enable_if_t<std::is_convertible_v<N, double>>,
          typename = std::enable_if_t<std::is_convertible_v<Q, QuakeValue>>>
QuakeValue operator+(N &&d, Q &&q) {
  return q + d;
}

template <typename N, typename Q,
          typename = std::enable_if_t<std::is_convertible_v<N, double>>,
          typename = std::enable_if_t<std::is_convertible_v<Q, QuakeValue>>>
QuakeValue operator/(N &&d, Q &&q) {
  return q.inverse() * d;
}
#endif

} // namespace cudaq
