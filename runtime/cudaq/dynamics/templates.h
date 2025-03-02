/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <type_traits>

#include "boson_operators.h"
#include "fermion_operators.h"
#include "matrix_operators.h"
#include "operator_leafs.h"
#include "spin_operators.h"

namespace cudaq {

template <typename HandlerTy>
class product_operator;

template <typename HandlerTy>
class operator_sum;

#define TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype)                             \
  std::enable_if_t<!std::is_same<LHtype, RHtype>::value &&                     \
                       !std::is_same<matrix_operator, LHtype>::value &&        \
                       std::is_base_of<operator_handler, LHtype>::value &&     \
                       std::is_base_of<operator_handler, RHtype>::value,       \
                   bool>

template <typename HandlerTy>
product_operator<HandlerTy> operator*(const scalar_operator &other,
                                      const product_operator<HandlerTy> &self);
template <typename HandlerTy>
product_operator<HandlerTy> operator*(const scalar_operator &other,
                                      product_operator<HandlerTy> &&self);
template <typename HandlerTy>
operator_sum<HandlerTy> operator+(const scalar_operator &other,
                                  const product_operator<HandlerTy> &self);
template <typename HandlerTy>
operator_sum<HandlerTy> operator+(const scalar_operator &other,
                                  product_operator<HandlerTy> &&self);
template <typename HandlerTy>
operator_sum<HandlerTy> operator-(const scalar_operator &other,
                                  const product_operator<HandlerTy> &self);
template <typename HandlerTy>
operator_sum<HandlerTy> operator-(const scalar_operator &other,
                                  product_operator<HandlerTy> &&self);

template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
product_operator<matrix_operator>
operator*(const product_operator<LHtype> &other,
          const product_operator<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator+(const product_operator<LHtype> &other,
                                        const product_operator<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator-(const product_operator<LHtype> &other,
                                        const product_operator<RHtype> &self);

template <typename HandlerTy>
operator_sum<HandlerTy> operator*(const scalar_operator &other,
                                  const operator_sum<HandlerTy> &self);
template <typename HandlerTy>
operator_sum<HandlerTy> operator*(const scalar_operator &other,
                                  operator_sum<HandlerTy> &&self);
template <typename HandlerTy>
operator_sum<HandlerTy> operator+(const scalar_operator &other,
                                  const operator_sum<HandlerTy> &self);
template <typename HandlerTy>
operator_sum<HandlerTy> operator+(const scalar_operator &other,
                                  operator_sum<HandlerTy> &&self);
template <typename HandlerTy>
operator_sum<HandlerTy> operator-(const scalar_operator &other,
                                  const operator_sum<HandlerTy> &self);
template <typename HandlerTy>
operator_sum<HandlerTy> operator-(const scalar_operator &other,
                                  operator_sum<HandlerTy> &&self);

template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator*(const operator_sum<LHtype> &other,
                                        const product_operator<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator+(const operator_sum<LHtype> &other,
                                        const product_operator<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator-(const operator_sum<LHtype> &other,
                                        const product_operator<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator*(const product_operator<LHtype> &other,
                                        const operator_sum<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator+(const product_operator<LHtype> &other,
                                        const operator_sum<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator-(const product_operator<LHtype> &other,
                                        const operator_sum<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator*(const operator_sum<LHtype> &other,
                                        const operator_sum<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator+(const operator_sum<LHtype> &other,
                                        const operator_sum<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator-(const operator_sum<LHtype> &other,
                                        const operator_sum<RHtype> &self);

#ifndef CUDAQ_INSTANTIATE_TEMPLATES
#define EXTERN_TEMPLATE_SPECIALIZATIONS(HandlerTy)                             \
                                                                               \
  extern template product_operator<HandlerTy> operator*(                       \
      const scalar_operator &other, const product_operator<HandlerTy> &self);  \
  extern template product_operator<HandlerTy> operator*(                       \
      const scalar_operator &other, product_operator<HandlerTy> &&self);       \
  extern template operator_sum<HandlerTy> operator+(                           \
      const scalar_operator &other, const product_operator<HandlerTy> &self);  \
  extern template operator_sum<HandlerTy> operator+(                           \
      const scalar_operator &other, product_operator<HandlerTy> &&self);       \
  extern template operator_sum<HandlerTy> operator-(                           \
      const scalar_operator &other, const product_operator<HandlerTy> &self);  \
  extern template operator_sum<HandlerTy> operator-(                           \
      const scalar_operator &other, product_operator<HandlerTy> &&self);       \
                                                                               \
  extern template operator_sum<HandlerTy> operator*(                           \
      const scalar_operator &other, const operator_sum<HandlerTy> &self);      \
  extern template operator_sum<HandlerTy> operator*(                           \
      const scalar_operator &other, operator_sum<HandlerTy> &&self);           \
  extern template operator_sum<HandlerTy> operator+(                           \
      const scalar_operator &other, const operator_sum<HandlerTy> &self);      \
  extern template operator_sum<HandlerTy> operator+(                           \
      const scalar_operator &other, operator_sum<HandlerTy> &&self);           \
  extern template operator_sum<HandlerTy> operator-(                           \
      const scalar_operator &other, const operator_sum<HandlerTy> &self);      \
  extern template operator_sum<HandlerTy> operator-(                           \
      const scalar_operator &other, operator_sum<HandlerTy> &&self);

EXTERN_TEMPLATE_SPECIALIZATIONS(matrix_operator);
EXTERN_TEMPLATE_SPECIALIZATIONS(spin_operator);
EXTERN_TEMPLATE_SPECIALIZATIONS(boson_operator);

#define EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(op, returnTy)               \
                                                                               \
  extern template returnTy<matrix_operator> operator op(                       \
      const product_operator<spin_operator> &other,                            \
      const product_operator<matrix_operator> &self);                          \
  extern template returnTy<matrix_operator> operator op(                       \
      const product_operator<boson_operator> &other,                           \
      const product_operator<matrix_operator> &self);                          \
  extern template returnTy<matrix_operator> operator op(                       \
      const product_operator<fermion_operator> &other,                         \
      const product_operator<matrix_operator> &self);                          \
  extern template returnTy<matrix_operator> operator op(                       \
      const product_operator<spin_operator> &other,                            \
      const product_operator<boson_operator> &self);                           \
  extern template returnTy<matrix_operator> operator op(                       \
      const product_operator<boson_operator> &other,                           \
      const product_operator<spin_operator> &self);                            \
  extern template returnTy<matrix_operator> operator op(                       \
      const product_operator<spin_operator> &other,                            \
      const product_operator<fermion_operator> &self);                         \
  extern template returnTy<matrix_operator> operator op(                       \
      const product_operator<fermion_operator> &other,                         \
      const product_operator<spin_operator> &self);                            \
  extern template returnTy<matrix_operator> operator op(                       \
      const product_operator<boson_operator> &other,                           \
      const product_operator<fermion_operator> &self);                         \
  extern template returnTy<matrix_operator> operator op(                       \
      const product_operator<fermion_operator> &other,                         \
      const product_operator<boson_operator> &self);                           \
                                                                               \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<spin_operator> &other,                                \
      const product_operator<matrix_operator> &self);                          \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<boson_operator> &other,                               \
      const product_operator<matrix_operator> &self);                          \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<fermion_operator> &other,                             \
      const product_operator<matrix_operator> &self);                          \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<spin_operator> &other,                                \
      const product_operator<boson_operator> &self);                           \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<boson_operator> &other,                               \
      const product_operator<spin_operator> &self);                            \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<spin_operator> &other,                                \
      const product_operator<fermion_operator> &self);                         \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<fermion_operator> &other,                             \
      const product_operator<spin_operator> &self);                            \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<boson_operator> &other,                               \
      const product_operator<fermion_operator> &self);                         \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<fermion_operator> &other,                             \
      const product_operator<boson_operator> &self);                           \
                                                                               \
  extern template operator_sum<matrix_operator> operator op(                   \
      const product_operator<spin_operator> &other,                            \
      const operator_sum<matrix_operator> &self);                              \
  extern template operator_sum<matrix_operator> operator op(                   \
      const product_operator<boson_operator> &other,                           \
      const operator_sum<matrix_operator> &self);                              \
  extern template operator_sum<matrix_operator> operator op(                   \
      const product_operator<fermion_operator> &other,                         \
      const operator_sum<matrix_operator> &self);                              \
  extern template operator_sum<matrix_operator> operator op(                   \
      const product_operator<spin_operator> &other,                            \
      const operator_sum<boson_operator> &self);                               \
  extern template operator_sum<matrix_operator> operator op(                   \
      const product_operator<boson_operator> &other,                           \
      const operator_sum<spin_operator> &self);                                \
  extern template operator_sum<matrix_operator> operator op(                   \
      const product_operator<spin_operator> &other,                            \
      const operator_sum<fermion_operator> &self);                             \
  extern template operator_sum<matrix_operator> operator op(                   \
      const product_operator<fermion_operator> &other,                         \
      const operator_sum<spin_operator> &self);                                \
  extern template operator_sum<matrix_operator> operator op(                   \
      const product_operator<boson_operator> &other,                           \
      const operator_sum<fermion_operator> &self);                             \
  extern template operator_sum<matrix_operator> operator op(                   \
      const product_operator<fermion_operator> &other,                         \
      const operator_sum<boson_operator> &self);                               \
                                                                               \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<spin_operator> &other,                                \
      const operator_sum<matrix_operator> &self);                              \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<boson_operator> &other,                               \
      const operator_sum<matrix_operator> &self);                              \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<fermion_operator> &other,                             \
      const operator_sum<matrix_operator> &self);                              \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<spin_operator> &other,                                \
      const operator_sum<boson_operator> &self);                               \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<boson_operator> &other,                               \
      const operator_sum<spin_operator> &self);                                \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<spin_operator> &other,                                \
      const operator_sum<fermion_operator> &self);                             \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<fermion_operator> &other,                             \
      const operator_sum<spin_operator> &self);                                \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<boson_operator> &other,                               \
      const operator_sum<fermion_operator> &self);                             \
  extern template operator_sum<matrix_operator> operator op(                   \
      const operator_sum<fermion_operator> &other,                             \
      const operator_sum<boson_operator> &self);

EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(*, product_operator);
EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(+, operator_sum);
EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(-, operator_sum);
#endif

// templates for arithmetics with callables

#define PRODUCT_FUNCTION_ARITHMETICS_RHS(op, returnTy)                         \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(product_operator<HandlerTy> &&prod,          \
                                  Callable &&fct) {                            \
    return std::move(prod)                                                     \
        op scalar_operator(scalar_callback(std::forward<Callable>(fct)));      \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(const product_operator<HandlerTy> &prod,     \
                                  Callable &&fct) {                            \
    return prod op scalar_operator(                                            \
        scalar_callback(std::forward<Callable>(fct)));                         \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(product_operator<HandlerTy> &&prod,          \
                                  scalar value) {                              \
    return std::move(prod) op scalar_operator(value);                          \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(const product_operator<HandlerTy> &prod,     \
                                  scalar value) {                              \
    return prod op scalar_operator(value);                                     \
  }

#define PRODUCT_FUNCTION_ARITHMETICS_LHS(op, returnTy)                         \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(Callable &&fct,                              \
                                  product_operator<HandlerTy> &&prod) {        \
    return scalar_operator(scalar_callback(std::forward<Callable>(fct)))       \
        op std::move(prod);                                                    \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(Callable &&fct,                              \
                                  const product_operator<HandlerTy> &prod) {   \
    return scalar_operator(scalar_callback(std::forward<Callable>(fct)))       \
        op prod;                                                               \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(scalar value,                                \
                                  product_operator<HandlerTy> &&prod) {        \
    return scalar_operator(value) op std::move(prod);                          \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(scalar value,                                \
                                  const product_operator<HandlerTy> &prod) {   \
    return scalar_operator(value) op prod;                                     \
  }

#define PRODUCT_FUNCTION_ARITHMETICS_ASSIGNMENT(op)                            \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  product_operator<HandlerTy> &operator op(product_operator<HandlerTy> &prod,  \
                                           Callable &&fct) {                   \
    return prod op scalar_operator(                                            \
        scalar_callback(std::forward<Callable>(fct)));                         \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  product_operator<HandlerTy> &operator op(product_operator<HandlerTy> &prod,  \
                                           scalar value) {                     \
    return prod op scalar_operator(value);                                     \
  }

PRODUCT_FUNCTION_ARITHMETICS_RHS(*, product_operator);
PRODUCT_FUNCTION_ARITHMETICS_RHS(/, product_operator);
PRODUCT_FUNCTION_ARITHMETICS_RHS(+, operator_sum);
PRODUCT_FUNCTION_ARITHMETICS_RHS(-, operator_sum);
PRODUCT_FUNCTION_ARITHMETICS_LHS(*, product_operator);
PRODUCT_FUNCTION_ARITHMETICS_LHS(+, operator_sum);
PRODUCT_FUNCTION_ARITHMETICS_LHS(-, operator_sum);
PRODUCT_FUNCTION_ARITHMETICS_ASSIGNMENT(*=);
PRODUCT_FUNCTION_ARITHMETICS_ASSIGNMENT(/=);

#define SUM_FUNCTION_ARITHMETICS_RHS(op)                                       \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  operator_sum<HandlerTy> operator op(operator_sum<HandlerTy> &&sum,           \
                                      Callable &&fct) {                        \
    return std::move(sum)                                                      \
        op scalar_operator(scalar_callback(std::forward<Callable>(fct)));      \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  operator_sum<HandlerTy> operator op(const operator_sum<HandlerTy> &sum,      \
                                      Callable &&fct) {                        \
    return sum op scalar_operator(                                             \
        scalar_callback(std::forward<Callable>(fct)));                         \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  operator_sum<HandlerTy> operator op(operator_sum<HandlerTy> &&sum,           \
                                      scalar value) {                          \
    return std::move(sum) op scalar_operator(value);                           \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  operator_sum<HandlerTy> operator op(const operator_sum<HandlerTy> &sum,      \
                                      scalar value) {                          \
    return sum op scalar_operator(value);                                      \
  }

#define SUM_FUNCTION_ARITHMETICS_LHS(op)                                       \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  operator_sum<HandlerTy> operator op(Callable &&fct,                          \
                                      operator_sum<HandlerTy> &&sum) {         \
    return scalar_operator(scalar_callback(std::forward<Callable>(fct)))       \
        op std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  operator_sum<HandlerTy> operator op(Callable &&fct,                          \
                                      const operator_sum<HandlerTy> &sum) {    \
    return scalar_operator(scalar_callback(std::forward<Callable>(fct)))       \
        op sum;                                                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  operator_sum<HandlerTy> operator op(scalar value,                            \
                                      operator_sum<HandlerTy> &&sum) {         \
    return scalar_operator(value) op std::move(sum);                           \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  operator_sum<HandlerTy> operator op(scalar value,                            \
                                      const operator_sum<HandlerTy> &sum) {    \
    return scalar_operator(value) op sum;                                      \
  }

#define SUM_FUNCTION_ARITHMETICS_ASSIGNMENT(op)                                \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  operator_sum<HandlerTy> &operator op(operator_sum<HandlerTy> &prod,          \
                                       Callable &&fct) {                       \
    return prod op scalar_operator(                                            \
        scalar_callback(std::forward<Callable>(fct)));                         \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  operator_sum<HandlerTy> &operator op(operator_sum<HandlerTy> &prod,          \
                                       scalar value) {                         \
    return prod op scalar_operator(value);                                     \
  }

SUM_FUNCTION_ARITHMETICS_RHS(*);
SUM_FUNCTION_ARITHMETICS_RHS(/);
SUM_FUNCTION_ARITHMETICS_RHS(+);
SUM_FUNCTION_ARITHMETICS_RHS(-);
SUM_FUNCTION_ARITHMETICS_LHS(*);
SUM_FUNCTION_ARITHMETICS_LHS(+);
SUM_FUNCTION_ARITHMETICS_LHS(-);
SUM_FUNCTION_ARITHMETICS_ASSIGNMENT(*=);
SUM_FUNCTION_ARITHMETICS_ASSIGNMENT(/=);
SUM_FUNCTION_ARITHMETICS_ASSIGNMENT(+=);
SUM_FUNCTION_ARITHMETICS_ASSIGNMENT(-=);

} // namespace cudaq